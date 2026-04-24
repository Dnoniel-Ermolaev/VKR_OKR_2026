from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


DEFAULT_COLLECTION_NAME = "acs_guidelines"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


@dataclass
class _Chunk:
    chunk_id: str
    text: str
    source: str
    title: str
    position: int
    page_start: int | None
    page_end: int | None


@dataclass
class RetrievalHit:
    chunk_id: str
    text: str
    source: str
    title: str
    chunk_position: int
    page_start: int | None
    page_end: int | None
    semantic_distance: float | None = None
    lexical_score: float = 0.0
    combined_score: float = 0.0

    @property
    def citation(self) -> str:
        page_part = ""
        if self.page_start is not None and self.page_end is not None:
            if self.page_start == self.page_end:
                page_part = f"page={self.page_start}, "
            else:
                page_part = f"pages={self.page_start}-{self.page_end}, "
        elif self.page_start is not None:
            page_part = f"page={self.page_start}, "
        return f"{self.source} ({page_part}chunk={self.chunk_position})"

    def formatted_text(self, rank: int) -> str:
        distance_hint = ""
        if self.semantic_distance is not None:
            distance_hint = f", distance={self.semantic_distance:.4f}"
        return (
            f"[RAG hit #{rank}] source={self.source}, title={self.title}, "
            f"page={self.page_start if self.page_start is not None else 'na'}, "
            f"chunk={self.chunk_position}, score={self.combined_score:.4f}{distance_hint}\n"
            f"{self.text.strip()}"
        )


class GuidelinesRetriever:
    """Local Chroma-based retriever with chunk cleaning, citations, and hybrid reranking."""

    def __init__(
        self,
        guidelines_dir: Path,
        *,
        persist_dir: Path | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
        chunk_size: int = 900,
        chunk_overlap: int = 180,
        cross_encoder_top_n: int = 12,
        use_cross_encoder: bool = True,
    ) -> None:
        self.guidelines_dir = guidelines_dir
        self.guidelines_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir = persist_dir or guidelines_dir.parent / "chroma"
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.cross_encoder_model = cross_encoder_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cross_encoder_top_n = cross_encoder_top_n
        self.use_cross_encoder = use_cross_encoder
        self.manifest_path = self.persist_dir / f"{collection_name}_manifest.json"

        self._client = None
        self._collection = None
        self._embedding_function = None
        self._cross_encoder = None
        self._init_error: str | None = None
        self._cross_encoder_error: str | None = None

        self._initialize_backend()
        self._ensure_index()

    def clean_guideline_files(self) -> int:
        cleaned = 0
        for doc in self._guideline_documents():
            parsed = self._parse_document(doc)
            metadata = dict(parsed.get("metadata", {}))
            sections = self._split_page_sections(str(parsed.get("body", "")))
            if sections:
                blocks = []
                for page_start, _, section_text in sections:
                    if page_start is not None:
                        blocks.append(f"[Page {page_start}]")
                    blocks.append(section_text.strip())
                body = "\n\n".join(block for block in blocks if block.strip())
            else:
                body = str(parsed.get("body", "")).strip()

            serialized = self._serialize_document(metadata=metadata, body=body)
            current = doc.read_text(encoding="utf-8", errors="ignore")
            if serialized != current:
                doc.write_text(serialized, encoding="utf-8")
                cleaned += 1
        return cleaned

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        hits = self.retrieve_hits(query=query, top_k=top_k)
        return [hit.formatted_text(rank=idx + 1) for idx, hit in enumerate(hits)]

    def retrieve_hits(self, query: str, top_k: int = 3) -> List[RetrievalHit]:
        documents = self._guideline_documents()
        if not documents:
            return []

        candidate_count = max(top_k * 4, 8)
        if self._collection is not None:
            try:
                result = self._collection.query(
                    query_texts=[query],
                    n_results=candidate_count,
                    include=["documents", "metadatas", "distances"],
                )
                docs = result.get("documents", [[]])[0]
                metas = result.get("metadatas", [[]])[0]
                distances = result.get("distances", [[]])[0]
                ids = result.get("ids", [[]])[0]
                if docs:
                    candidates = [
                        RetrievalHit(
                            chunk_id=str(ids[idx]) if idx < len(ids) else str((meta or {}).get("chunk_id", "")),
                            text=doc,
                            source=str((meta or {}).get("source", "unknown")),
                            title=str((meta or {}).get("title", "guideline")),
                            chunk_position=int((meta or {}).get("position", idx)),
                            page_start=self._to_int((meta or {}).get("page_start")),
                            page_end=self._to_int((meta or {}).get("page_end")),
                            semantic_distance=distances[idx] if idx < len(distances) else None,
                        )
                        for idx, (doc, meta) in enumerate(zip(docs, metas))
                    ]
                    reranked = self._rerank_hits(query=query, hits=candidates)
                    return reranked[:top_k]
            except Exception as exc:
                self._init_error = f"Chroma retrieval failed: {exc}"

        return self._fallback_retrieve(query=query, top_k=top_k)

    def index_documents(self, *, force: bool = False) -> int:
        if self._collection is None or self._client is None:
            return 0

        fingerprint = self._documents_fingerprint()
        manifest = self._load_manifest()
        if not force and manifest.get("fingerprint") == fingerprint and self._collection.count() > 0:
            return int(manifest.get("chunks", 0))

        chunks = self._collect_chunks()
        if not chunks:
            self._save_manifest({"fingerprint": fingerprint, "chunks": 0})
            return 0

        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        self._collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[self._metadata_for_chunk(chunk) for chunk in chunks],
        )
        self._save_manifest({"fingerprint": fingerprint, "chunks": len(chunks)})
        return len(chunks)

    def _initialize_backend(self) -> None:
        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

            self._embedding_function = SentenceTransformerEmbeddingFunction(model_name=self.embedding_model)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            self._init_error = f"Chroma unavailable: {exc}"
            self._client = None
            self._collection = None
            self._embedding_function = None

    def _ensure_index(self) -> None:
        if self._collection is None:
            return
        try:
            self.index_documents()
        except Exception as exc:
            self._init_error = f"RAG index build failed: {exc}"

    def _guideline_documents(self) -> List[Path]:
        return sorted(self.guidelines_dir.glob("*.txt"))

    def _documents_fingerprint(self) -> str:
        hasher = hashlib.sha256()
        for doc in self._guideline_documents():
            stat = doc.stat()
            hasher.update(str(doc.relative_to(self.guidelines_dir)).encode("utf-8"))
            hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
            hasher.update(str(stat.st_size).encode("utf-8"))
        return hasher.hexdigest()

    def _collect_chunks(self) -> List[_Chunk]:
        chunks: List[_Chunk] = []
        for doc in self._guideline_documents():
            parsed = self._parse_document(doc)
            title = parsed["title"]
            body = str(parsed["body"]).strip()
            if not body:
                continue
            chunk_index = 0
            for page_start, page_end, section_text in self._split_page_sections(body):
                for chunk_text in self._chunk_text(section_text):
                    clean_chunk = self._clean_chunk_text(chunk_text)
                    if not clean_chunk or self._is_low_signal_chunk(clean_chunk):
                        continue
                    chunk_id = hashlib.sha1(
                        f"{doc.as_posix()}::{page_start}:{page_end}::{chunk_index}::{clean_chunk}".encode("utf-8")
                    ).hexdigest()
                    chunks.append(
                        _Chunk(
                            chunk_id=chunk_id,
                            text=clean_chunk,
                            source=doc.name,
                            title=title,
                            position=chunk_index,
                            page_start=page_start,
                            page_end=page_end,
                        )
                    )
                    chunk_index += 1
        return chunks

    def _chunk_text(self, text: str) -> List[str]:
        normalized = re.sub(r"\r\n?", "\n", text)
        blocks = [block.strip() for block in re.split(r"\n\s*\n", normalized) if block.strip()]
        units = blocks if blocks else [normalized.strip()]

        chunks: List[str] = []
        current = ""
        for unit in units:
            candidate = unit if not current else f"{current}\n\n{unit}"
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current.strip())
            if len(unit) <= self.chunk_size:
                current = unit
                continue

            start = 0
            step = max(1, self.chunk_size - self.chunk_overlap)
            while start < len(unit):
                piece = unit[start : start + self.chunk_size].strip()
                if piece:
                    chunks.append(piece)
                start += step
            current = ""

        if current:
            chunks.append(current.strip())
        return chunks

    def _fallback_retrieve(self, query: str, top_k: int) -> List[RetrievalHit]:
        query_tokens = {token for token in re.findall(r"\w+", query.lower()) if len(token) > 1}
        hits: List[RetrievalHit] = []
        for chunk in self._collect_chunks():
            lower = chunk.text.lower()
            lexical_score = self._lexical_score(query_tokens, lower)
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    title=chunk.title,
                    chunk_position=chunk.position,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    semantic_distance=None,
                    lexical_score=lexical_score,
                    combined_score=lexical_score,
                )
            )

        reranked = self._rerank_hits(query=query, hits=hits)
        return reranked[:top_k]

    def _parse_document(self, path: Path) -> Dict[str, str]:
        raw_text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not raw_text:
            return {"title": path.stem.replace("_", " "), "body": "", "metadata": {}}

        metadata: Dict[str, str] = {}
        body_lines: List[str] = []
        in_metadata = True
        for line in raw_text.splitlines():
            if in_metadata and not line.strip():
                in_metadata = False
                continue
            if in_metadata and ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip().upper()] = value.strip()
                continue
            in_metadata = False
            body_lines.append(line)

        body = self._clean_document_body("\n".join(body_lines))
        title = metadata.get("TITLE", path.stem.replace("_", " "))
        return {"title": title, "body": body, "metadata": metadata}

    def _clean_document_body(self, text: str) -> str:
        normalized = re.sub(r"\r\n?", "\n", text)
        lines = normalized.splitlines()
        cleaned_lines: List[str] = []
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                if cleaned_lines and cleaned_lines[-1] != "":
                    cleaned_lines.append("")
                continue
            if line.startswith("[Page ") or line.startswith("[Slide "):
                cleaned_lines.append(line)
                continue
            if self._is_noise_line(line):
                continue
            cleaned_lines.append(line)
        cleaned = "\n".join(cleaned_lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _split_page_sections(self, body: str) -> List[tuple[int | None, int | None, str]]:
        sections: List[tuple[int | None, int | None, str]] = []
        current_page: int | None = None
        buffer: List[str] = []

        def flush() -> None:
            nonlocal buffer, current_page
            text = "\n".join(buffer).strip()
            if text and not self._is_noise_page(text):
                sections.append((current_page, current_page, text))
            buffer = []

        for line in body.splitlines():
            page_match = re.fullmatch(r"\[(?:Page|Slide)\s+(\d+)\]", line.strip())
            if page_match:
                flush()
                current_page = int(page_match.group(1))
                continue
            buffer.append(line)
        flush()

        if sections:
            return sections
        fallback_text = body.strip()
        return [(None, None, fallback_text)] if fallback_text else []

    def _clean_chunk_text(self, text: str) -> str:
        cleaned = re.sub(r"\n{3,}", "\n\n", text.strip())
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        return cleaned.strip()

    def _is_noise_page(self, text: str) -> bool:
        lower = text.lower()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        dotted_lines = sum(1 for line in lines if "." * 5 in line or re.search(r"\.{4,}\s*\d+$", line))
        appendix_lines = sum(1 for line in lines if line.lower().startswith("приложение"))
        heading_like_lines = sum(
            1
            for line in lines
            if len(line) < 120 and (
                re.match(r"^\d+(\.\d+)*", line) or
                line.lower().startswith("приложение") or
                line.lower().startswith("список ") or
                line.lower().startswith("термины ") or
                line.lower().startswith("keywords")
            )
        )
        if "оглавление" in lower or "contents" in lower:
            return True
        if dotted_lines >= 3:
            return True
        if appendix_lines >= 3:
            return True
        if len(lines) >= 5 and heading_like_lines >= max(4, int(len(lines) * 0.6)):
            return True
        author_markers = (
            "authors/task force members",
            "task force co-ordinator",
            "scientific document group",
            "corresponding authors",
            "document reviewers",
            "состав рабочей группы",
        )
        if any(marker in lower for marker in author_markers):
            return True
        return False

    def _is_noise_line(self, line: str) -> bool:
        lower = line.lower()
        if re.fullmatch(r"\d+", line):
            return True
        if re.fullmatch(r"--\s*\d+\s+of\s+\d+\s*--", line):
            return True
        if "." * 8 in line or re.search(r"\.{4,}\s*\d+$", line):
            return True
        noise_markers = (
            "downloaded from",
            "document reviewers",
            "authors/task force members",
            "task force co-ordinator",
            "scientific document group",
            "corresponding authors",
            "all rights reserved",
            "journals.permissions@oup.com",
            "european heart journal",
            "guest on",
        )
        return any(marker in lower for marker in noise_markers)

    def _is_low_signal_chunk(self, text: str) -> bool:
        lower = text.lower()
        if len(lower) < 80:
            return True
        banned = (
            "оглавление",
            "authors/task force members",
            "document reviewers",
            "all rights reserved",
        )
        return any(marker in lower for marker in banned)

    def _rerank_hits(self, query: str, hits: List[RetrievalHit]) -> List[RetrievalHit]:
        query_tokens = {token for token in re.findall(r"\w+", query.lower()) if len(token) > 1}
        reranked: List[RetrievalHit] = []
        for hit in hits:
            lexical = self._lexical_score(query_tokens, hit.text.lower())
            semantic = 0.0
            if hit.semantic_distance is not None:
                semantic = max(0.0, 1.0 - min(hit.semantic_distance, 1.5) / 1.5)
            combined = 0.65 * semantic + 0.35 * lexical
            if hit.page_start is not None:
                combined += 0.03
            if self._is_low_signal_chunk(hit.text):
                combined -= 0.25
            hit.lexical_score = lexical
            hit.combined_score = combined
            reranked.append(hit)
        reranked.sort(key=lambda item: item.combined_score, reverse=True)
        reranked = self._cross_encoder_rerank(query=query, hits=reranked)
        return reranked

    def _lexical_score(self, query_tokens: set[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        matches = sum(1 for token in query_tokens if token in text)
        return matches / len(query_tokens)

    def _to_int(self, value: object) -> int | None:
        try:
            if value is None or value == "":
                return None
            return int(value)
        except Exception:
            return None

    def _load_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_manifest(self, payload: dict) -> None:
        self.manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _metadata_for_chunk(self, chunk: _Chunk) -> dict:
        metadata: dict = {
            "source": chunk.source,
            "title": chunk.title,
            "position": chunk.position,
            "chunk_id": chunk.chunk_id,
        }
        if chunk.page_start is not None:
            metadata["page_start"] = chunk.page_start
        if chunk.page_end is not None:
            metadata["page_end"] = chunk.page_end
        return metadata

    def _serialize_document(self, *, metadata: Dict[str, str], body: str) -> str:
        title = metadata.get("TITLE", "guideline")
        source_file = metadata.get("SOURCE_FILE", "")
        doc_type = metadata.get("DOC_TYPE", "prepared_text")
        lang = metadata.get("LANG", "ru_or_en")
        topic = metadata.get("TOPIC", "ACS, OKS, cardiology, triage")
        return (
            f"TITLE: {title}\n"
            f"SOURCE_FILE: {source_file}\n"
            f"DOC_TYPE: {doc_type}\n"
            f"LANG: {lang}\n"
            f"TOPIC: {topic}\n\n"
            f"{body.strip()}\n"
        )

    def _ensure_cross_encoder(self) -> bool:
        if not self.use_cross_encoder:
            return False
        if self._cross_encoder is not None:
            return True
        if self._cross_encoder_error is not None:
            return False
        try:
            from sentence_transformers.cross_encoder import CrossEncoder

            self._cross_encoder = CrossEncoder(self.cross_encoder_model)
            return True
        except Exception as exc:
            self._cross_encoder_error = str(exc)
            return False

    def _cross_encoder_rerank(self, *, query: str, hits: List[RetrievalHit]) -> List[RetrievalHit]:
        if not hits or not self._ensure_cross_encoder():
            return hits

        limit = min(self.cross_encoder_top_n, len(hits))
        rerank_hits = hits[:limit]
        try:
            scores = self._cross_encoder.predict([(query, hit.text) for hit in rerank_hits])
        except Exception as exc:
            self._cross_encoder_error = str(exc)
            return hits

        for hit, score in zip(rerank_hits, scores):
            ce_score = 1.0 / (1.0 + math.exp(-float(score)))
            hit.combined_score = 0.55 * hit.combined_score + 0.45 * ce_score

        rerank_hits.sort(key=lambda item: item.combined_score, reverse=True)
        return rerank_hits + hits[limit:]
