# app.py
# Streamlit UI for Capstone Title Similarity (SBERT / Hybrid / TF-IDF)

import os, sys, time
import numpy as np
import pandas as pd
import streamlit as st

# make src importable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from similarity_engine import (
    load_titles, TfidfIndex,
    build_or_load_sbert, sbert_search, hybrid_search
)

# ----------------------- CONFIG -----------------------
DEFAULT_CSV = "data/Capstone Project Database.csv"
DEFAULT_TITLE_COL = None   # auto-detect
DEFAULT_TOPK = 5
DEFAULT_ALPHA = 0.7        # SBERT weight in hybrid
MODEL_NOTE = "Model: sentence-transformers/all-MiniLM-L6-v2 (CPU)"
# ------------------------------------------------------

st.set_page_config(page_title="Capstone Title Similarity", layout="wide")
st.title("🎓 Capstone Title Similarity Search")
st.caption(MODEL_NOTE)

with st.sidebar:
    st.header("Data source")
    csv_mode = st.radio("Choose data source", ["Local CSV", "Google Sheet (CSV URL)"], index=0)

    if csv_mode == "Local CSV":
        csv_path = st.text_input("Local CSV path", value=DEFAULT_CSV)
    else:
        csv_path = st.text_input(
            "Public Google Sheet CSV URL",
            value="",
            placeholder="Paste the published CSV link here (File → Share → Publish to web → CSV)"
        )

    title_col = st.text_input("Title column (optional)", value="" if DEFAULT_TITLE_COL is None else DEFAULT_TITLE_COL)
    top_k = st.slider("Top K results", 3, 15, value=DEFAULT_TOPK, step=1)

    method = st.radio(
        "Similarity method",
        ["SBERT (semantic)", "Hybrid (SBERT 70% + TF-IDF 30%)", "TF-IDF word 1–3", "TF-IDF char 3–5"],
        index=1
    )

    if method.startswith("Hybrid"):
        alpha = st.slider("Hybrid α (SBERT weight)", 0.0, 1.0, value=DEFAULT_ALPHA, step=0.05)
    else:
        alpha = DEFAULT_ALPHA

@st.cache_data(show_spinner=True)
def _load_titles(csv_path: str, col_name: str | None):
    # supports both local file and HTTP(S) CSV
    if csv_path.lower().startswith(("http://","https://")):
        df = pd.read_csv(csv_path)
        # save a local cache so embedding cache refresh works
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/_remote_cache.csv", index=False)
        csv_for_cache = "data/_remote_cache.csv"
    else:
        df = pd.read_csv(csv_path)
        csv_for_cache = csv_path

    # use engine’s helper to guess title column if not provided
    from similarity_engine import guess_title_column, clean_text
    col = (col_name or "").strip() or guess_title_column(df)
    titles = df[col].astype(str).map(clean_text)
    titles = titles[titles.str.len() > 0].reset_index(drop=True)
    return titles, col, csv_for_cache

@st.cache_resource(show_spinner=True)
def _build_tfidf_indexes(titles: pd.Series):
    return (
        TfidfIndex(titles.tolist(), kind="word"),
        TfidfIndex(titles.tolist(), kind="char"),
    )

@st.cache_resource(show_spinner=True)
def _build_sbert(csv_for_cache: str, titles: pd.Series):
    # builds or loads cached embeddings in /data
    emb = build_or_load_sbert(csv_path=csv_for_cache, titles=titles)
    return emb

# ----------------------- LOAD DATA -----------------------
try:
    titles, used_col, csv_for_cache = _load_titles(csv_path, title_col if title_col else None)
    st.success(f"Loaded **{len(titles)}** titles • Using column: **{used_col}**")
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

# prepare models
with st.spinner("Building indexes… (first run may download the model)"):
    if method in ("SBERT (semantic)", "Hybrid (SBERT 70% + TF-IDF 30%)"):
        emb = _build_sbert(csv_for_cache, titles)
    word_idx, char_idx = _build_tfidf_indexes(titles)

# ----------------------- SEARCH UI -----------------------
query = st.text_input("Type a new capstone title to check for similarity:", "")
go = st.button("Search", type="primary")

def _run_search(q: str):
    if not q.strip():
        return []

    if method == "SBERT (semantic)":
        results = sbert_search(q, titles, emb, k=top_k)
    elif method.startswith("Hybrid"):
        results = hybrid_search(q, titles, emb, alpha=alpha, k=top_k)
    elif method == "TF-IDF word 1–3":
        results = word_idx.search(q, k=top_k, drop_exact=False)
    else:
        results = char_idx.search(q, k=top_k, drop_exact=False)

    out = pd.DataFrame(results, columns=["Matched Title", "Similarity %"])
    return out

col1, col2 = st.columns([2, 1])
with col1:
    if go and query:
        with st.spinner("Searching…"):
            df = _run_search(query)
        if len(df):
            st.subheader("Top Matches")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No matches found.")

with col2:
    st.markdown("### Settings")
    st.write(f"**Method:** {method}")
    if method.startswith("Hybrid"):
        st.write(f"**Hybrid α:** {alpha:.2f}")
    st.write(f"**Top K:** {top_k}")
    st.caption("Tip: Hybrid combines SBERT meaning with TF-IDF keywords for robust results.")

st.markdown("---")
st.caption("Built with 🧠 SBERT embeddings + cosine similarity. First run creates an embeddings cache in `/data` for fast reuse.")
