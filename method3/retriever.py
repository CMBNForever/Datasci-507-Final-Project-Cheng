"""
Helper functions for the retrieval system.
Chunk scraped text, embed, write index; load index and cosine top-K search.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _sentence_transformer(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)
MAX_WORDS_PER_CHUNK = 400
MIN_CHUNK_WORDS = 30


def generate_query(row: Mapping[str, Any]) -> str:
    """Map a behavioral profile row to a natural-language query for retrieval."""
    parts: list[str] = []

    if float(row["usage_score"]) > 0.5:
        parts.append("excessive usage")
    if float(row["interaction_score"]) > 0.5:
        parts.append("frequent scrolling")

    srr = float(row["self_reg_risk"])
    if srr > 0.5:
        parts.append("low self-control")
    elif srr < -0.5:
        parts.append("high self-control")

    if row.get("Watch_Reason_Procrastination", 0) == 1:
        parts.append("procrastination")
    if row.get("Watch_Reason_Habit", 0) == 1:
        parts.append("compulsive usage")
    if row.get("Watch_Reason_Boredom", 0) == 1:
        parts.append("idle usage")

    for key, val in row.items():
        if isinstance(key, str) and key.startswith("Video_Category_") and val == 1:
            category = key.replace("Video_Category_", "")
            parts.append(f"{category.lower()} content")

    return ", ".join(parts)


def chunk_text(text: str, max_words: int = MAX_WORDS_PER_CHUNK) -> list[str]:
    words = text.split()
    if not words:
        return []
    out: list[str] = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words])
        if len(chunk.split()) >= MIN_CHUNK_WORDS:
            out.append(chunk)
    return out if out else [" ".join(words)]


def load_manifest(root: Path) -> dict[str, dict[str, Any]]:
    p = root / "data" / "knowledge" / "scraped" / "manifest.json"
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {m["id"]: m for m in data if m.get("ok") and m.get("text_path")}


def build_index(root: Optional[Path] = None) -> Path:
    root = root or Path(__file__).resolve().parent.parent
    text_dir = root / "data" / "knowledge" / "scraped" / "text"
    index_dir = root / "data" / "knowledge" / "index"
    if not text_dir.is_dir():
        raise FileNotFoundError(f"Missing corpus: {text_dir}")

    manifest = load_manifest(root)
    records: list[dict[str, Any]] = []
    texts: list[str] = []

    for path in sorted(text_dir.glob("*.txt")):
        body = path.read_text(encoding="utf-8", errors="replace").strip()
        if len(body) < 50:
            continue
        meta = manifest.get(path.stem, {})
        for i, chunk in enumerate(chunk_text(body)):
            records.append(
                {
                    "chunk_id": f"{path.stem}__{i:04d}",
                    "doc_id": path.stem,
                    "url": meta.get("url", ""),
                    "source": meta.get("source", ""),
                    "text": chunk,
                }
            )
            texts.append(chunk)

    if not texts:
        raise RuntimeError("No chunks produced — check data/knowledge/scraped/text/")

    index_dir.mkdir(parents=True, exist_ok=True)
    model = _sentence_transformer(DEFAULT_MODEL)
    emb = np.asarray(
        model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    (index_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
        encoding="utf-8",
    )
    np.save(index_dir / "embeddings.npy", emb)
    (index_dir / "meta.json").write_text(
        json.dumps(
            {
                "model_name": DEFAULT_MODEL,
                "dim": int(emb.shape[1]),
                "n_chunks": int(emb.shape[0]),
                "max_words_per_chunk": MAX_WORDS_PER_CHUNK,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return index_dir


@dataclass
class RetrievalHit:
    chunk_id: str
    doc_id: str
    score: float
    text: str
    url: str
    source: str


class ChunkIndex:
    def __init__(self, index_dir: Optional[Path] = None, root: Optional[Path] = None):
        root = root or Path(__file__).resolve().parent.parent
        self.index_dir = Path(index_dir) if index_dir else root / "data" / "knowledge" / "index"
        self.meta = json.loads((self.index_dir / "meta.json").read_text(encoding="utf-8"))
        self.embeddings = np.load(self.index_dir / "embeddings.npy")
        self.chunks: list[dict[str, Any]] = []
        for line in (self.index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                self.chunks.append(json.loads(line))
        self._model: Any = None

    def _model_name(self) -> str:
        return self.meta.get("model_name") or self.meta.get("model", DEFAULT_MODEL)

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = _sentence_transformer(self._model_name())
        return self._model

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        qtext = (query or "").strip() or "social media mental health screen time youth well-being"
        q = self.model.encode(
            [qtext], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        scores = (self.embeddings @ q.T).ravel()
        k = min(top_k, len(scores))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        hits: list[RetrievalHit] = []
        for i in top_idx:
            c = self.chunks[int(i)]
            hits.append(
                RetrievalHit(
                    chunk_id=c["chunk_id"],
                    doc_id=c["doc_id"],
                    score=float(scores[int(i)]),
                    text=c["text"],
                    url=c.get("url", ""),
                    source=c.get("source", ""),
                )
            )
        return hits
