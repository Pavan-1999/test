# src/similarity_engine.py
# One file with: data loading, TF-IDF models, SBERT embeddings (with caching), and search helpers.

from __future__ import annotations
import os, json, time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# --------------------
# Config (edit if you want different defaults)
# --------------------
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --------------------
# Basic cleaning
# --------------------
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    return " ".join(s.lower().strip().split())

def guess_title_column(df: pd.DataFrame) -> str:
    candidates = ["Project Title","Title","Capstone Title","project_title","capstone_title","topic","name"]
    for c in candidates:
        if c in df.columns:
            return c
    return max(df.columns, key=lambda c: df[c].astype(str).str.len().mean())

# --------------------
# Data loader
# --------------------
def load_titles(csv_path: str, title_col: Optional[str] = None) -> Tuple[pd.Series, str]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    col = title_col or guess_title_column(df)
    titles = df[col].astype(str).map(clean_text)
    titles = titles[titles.str.len() > 0].reset_index(drop=True)
    return titles, col

# --------------------
# TF-IDF (word + char) — fast lexical baselines
# --------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfidfIndex:
    def __init__(self, titles: List[str], kind: str = "word"):
        self.kind = kind
        if kind == "word":
            self.vec = TfidfVectorizer(analyzer="word", ngram_range=(1,3), strip_accents="unicode")
        else:
            self.vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5))
        self.matrix = self.vec.fit_transform(titles)
        self.titles = titles

    def search(self, query: str, k: int = 5, drop_exact: bool = True) -> List[Tuple[str, float]]:
        qv = self.vec.transform([clean_text(query)])
        sims = cosine_similarity(qv, self.matrix)[0]
        if drop_exact:
            cq = clean_text(query)
            for i, t in enumerate(self.titles):
                if t == cq:
                    sims[i] = -1.0
        order = np.argsort(-sims)[:k]
        return [(self.titles[i], round(float(sims[i]) * 100.0, 1)) for i in order]

# --------------------
# SBERT embeddings (semantic) with lightweight caching
# --------------------
def _cache_paths(data_dir: str, model_name: str) -> Tuple[Path, Path]:
    emb_path = Path(data_dir) / f"embeddings-{Path(model_name).name}.npy"
    meta_path = Path(data_dir) / f"embeddings-{Path(model_name).name}.meta.json"
    return emb_path, meta_path

def _need_rebuild(csv_path: str, titles: pd.Series, model_name: str, emb_p: Path, meta_p: Path) -> bool:
    if not (emb_p.exists() and meta_p.exists()):
        return True
    try:
        meta = json.loads(meta_p.read_text())
    except Exception:
        return True
    if meta.get("model") != model_name:
        return True
    if int(meta.get("num_rows", -1)) != int(len(titles)):
        return True
    return os.path.getmtime(csv_path) > emb_p.stat().st_mtime

def build_or_load_sbert(
    csv_path: str,
    titles: pd.Series,
    data_dir: str = "data",
    model_name: str = DEFAULT_MODEL
) -> np.ndarray:
    """Return L2-normalized embeddings for titles (NxD). Uses on-disk cache."""
    from sentence_transformers import SentenceTransformer  # lazy import
    emb_path, meta_path = _cache_paths(data_dir, model_name)
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    if _need_rebuild(csv_path, titles, model_name, emb_path, meta_path):
        model = SentenceTransformer(model_name)
        emb = model.encode(
            titles.tolist(),
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        np.save(emb_path, emb)
        meta = {
            "model": model_name,
            "num_rows": int(len(titles)),
            "csv": csv_path,
            "saved_at": time.time(),
            "title_preview": titles.head(3).tolist(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
    else:
        emb = np.load(emb_path)
    return emb

def sbert_search(query: str, titles: pd.Series, emb: np.ndarray, model_name: str = DEFAULT_MODEL, k: int = 5) -> List[Tuple[str, float]]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    q_emb = model.encode([clean_text(query)], normalize_embeddings=True, convert_to_numpy=True)
    scores = (q_emb @ emb.T)[0]  # cosine because both normalized
    idx = np.argsort(-scores)[:k]
    return [(titles.iloc[i], round(float(scores[i]) * 100.0, 1)) for i in idx]

# --------------------
# Hybrid (SBERT + TF-IDF) — like a tiny search engine
# --------------------
def hybrid_search(
    query: str,
    titles: pd.Series,
    emb: np.ndarray,
    model_name: str = DEFAULT_MODEL,
    alpha: float = 0.7,  # SBERT weight (0..1)
    k: int = 5,
) -> List[Tuple[str, float]]:
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1,3), strip_accents="unicode")
    X = vec.fit_transform(titles.tolist())

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    q = clean_text(query)

    q_emb = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)
    sbert_scores = (q_emb @ emb.T)[0]  # 0..1

    qv = vec.transform([q])
    tfidf_scores = cosine_similarity(qv, X)[0]  # 0..1

    blended = alpha * sbert_scores + (1 - alpha) * tfidf_scores
    idx = np.argsort(-blended)[:k]
    return [(titles.iloc[i], round(float(blended[i]) * 100.0, 1)) for i in idx]
