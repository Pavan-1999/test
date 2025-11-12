import hashlib
import io
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- SBERT lazy load ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_sbert(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

# ---- Helpers ----------------------------------------------------------------
def _norm_scores(v: np.ndarray) -> np.ndarray:
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    if v.size == 0:
        return v
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)

def _as_list(col: pd.Series) -> List[str]:
    return [str(x).strip() for x in col.fillna("").astype(str).tolist()]

def _hash_titles(titles: List[str]) -> str:
    return hashlib.sha256("\n".join(titles).encode("utf-8")).hexdigest()

# ---- Cached embeddings/vectorizers ------------------------------------------
@st.cache_data(show_spinner=False)
def build_sbert_embeddings(titles: List[str], model_name: str) -> np.ndarray:
    model = load_sbert(model_name)
    emb = model.encode(titles, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    return emb.astype("float32")

@st.cache_data(show_spinner=False)
def build_tfidf_word(titles: List[str]):
    v = TfidfVectorizer(ngram_range=(1, 3), min_df=1, lowercase=True)
    X = v.fit_transform(titles)
    return v, X

@st.cache_data(show_spinner=False)
def build_tfidf_char(titles: List[str]):
    v = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1, lowercase=True)
    X = v.fit_transform(titles)
    return v, X

# ---- Search methods ----------------------------------------------------------
def sbert_search(query: str, titles: List[str], emb_titles: np.ndarray, model_name: str, k: int) -> List[Tuple[str, float]]:
    model = load_sbert(model_name)
    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    sims = (qv @ emb_titles.T).ravel()   # cosine (because normalized)
    order = np.argsort(-sims)[:k]
    return [(titles[i], float(sims[i])) for i in order]

def tfidf_search(query: str, titles: List[str], vec, mat, k: int) -> List[Tuple[str, float]]:
    q = vec.transform([query])
    sims = cosine_similarity(q, mat).ravel()
    order = np.argsort(-sims)[:k]
    return [(titles[i], float(sims[i])) for i in order]

def hybrid_search(query: str, titles: List[str],
                  emb_titles: np.ndarray, model_name: str,
                  vec_word, X_word, vec_char, X_char,
                  alpha: float, k: int) -> List[Tuple[str, float]]:
    # SBERT
    sbert = dict(sbert_search(query, titles, emb_titles, model_name, k=len(titles)))
    sbert_vec = np.array([sbert.get(t, 0.0) for t in titles])

    # TF-IDF word + char
    t_word = dict(tfidf_search(query, titles, vec_word, X_word, k=len(titles)))
    t_char = dict(tfidf_search(query, titles, vec_char, X_char, k=len(titles)))
    tfidf_vec = 0.5 * np.array([t_word.get(t, 0.0) for t in titles]) + 0.5 * np.array([t_char.get(t, 0.0) for t in titles])

    s_norm = _norm_scores(sbert_vec)
    t_norm = _norm_scores(tfidf_vec)
    score = alpha * s_norm + (1 - alpha) * t_norm

    order = np.argsort(-score)[:k]
    return [(titles[i], float(score[i])) for i in order]

# ---- UI ----------------------------------------------------------------------
st.set_page_config(page_title="Capstone Title Similarity", page_icon="🎓", layout="wide")

st.title("🎓 Capstone Title Similarity Search")
st.caption("Semantic search using **SBERT** (sentence-transformers) with optional **Hybrid** TF-IDF mixing.")

left, right = st.columns([1, 2])

with left:
    st.subheader("Data source")
    source = st.radio("Choose data source", ["Local CSV", "Google Sheet (CSV URL)"], index=0)
    if source == "Local CSV":
        csv_path = st.text_input("Local CSV path", value="data/Capstone Project Database.csv")
        df = pd.read_csv(csv_path)
    else:
        url = st.text_input("Google Sheet CSV URL")
        if url.strip():
            df = pd.read_csv(url)
        else:
            st.stop()

    colname = st.text_input("Title column name (leave blank to auto-detect)", value="")
    if not colname:
        # try common names
        for c in ["Project Title", "Title", "project_title", "name"]:
            if c in df.columns:
                colname = c
                break
        if not colname:
            colname = st.selectbox("Pick a column", df.columns.tolist())

    titles = _as_list(df[colname])
    st.success(f"Loaded **{len(titles)}** titles • Using column: **{colname}**")

    top_k = st.slider("Top K results", 3, 15, 5)
    method = st.radio("Similarity method", ["SBERT (semantic)", "Hybrid (SBERT 70% + TF-IDF 30%)"], index=1)
    alpha = st.slider("Hybrid α (SBERT weight)", 0.0, 1.0, 0.7)

with right:
    st.subheader("Search")
    query = st.text_input("Type a new capstone title to check for similarity:")
    btn = st.button("Search", type="primary")

# Build models/embeddings (cached)
model_name = "sentence-transformers/all-MiniLM-L6-v2"
if titles:
    title_hash = _hash_titles(titles)
    emb_titles = build_sbert_embeddings(titles, model_name)
    vec_w, X_w = build_tfidf_word(titles)
    vec_c, X_c = build_tfidf_char(titles)
else:
    st.stop()

if btn and query.strip():
    if method.startswith("SBERT"):
        results = sbert_search(query, titles, emb_titles, model_name, k=top_k)
        label = "SBERT (cosine)"
        to_pct = lambda s: 100 * _norm_scores(np.array([r[1] for r in results]))
        pct = to_pct(0)  # not used, just placeholder for shape
        rows = [(t, 100 * s) for (t, s) in results]
    else:
        results = hybrid_search(query, titles, emb_titles, model_name, vec_w, X_w, vec_c, X_c, alpha, k=top_k)
        rows = [(t, 100 * s) for (t, s) in results]

    out = pd.DataFrame(rows, columns=["Matched Title", "Similarity %"])
    out["Similarity %"] = out["Similarity %"].map(lambda x: f"{x:0.1f}")
    st.subheader("Top Matches")
    st.dataframe(out, use_container_width=True)

st.markdown("---")
st.caption("Built with 🧠 SBERT embeddings + cosine similarity. Hybrid combines SBERT meaning with TF-IDF n-grams for robustness.")
