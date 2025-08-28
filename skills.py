# skills.py
"""
Starter skills inventory + a lightweight matcher.
- SKILLS: 100+ DS/ML/Analytics/Cloud/Eng tools & topics
- match_skills(text): returns normalized unique matched skills (list)
Matching strategy:
  1) Exact word/phrase boundary match (preferred)
  2) RapidFuzz fallback for near-misses (e.g., "Pytorch" vs "PyTorch")
You can tune THRESHOLD or synonyms below.
"""

from __future__ import annotations
import re
from typing import List, Dict, Iterable, Tuple, Set
from rapidfuzz import fuzz, process

# --- Canonical skills (keep lowercase for matching) ---
SKILLS: List[str] = [s.lower() for s in [
    # Languages / Core
    "python", "r", "sql", "scala", "java", "c++", "bash", "linux", "powershell", "javascript", "typescript",
    # Python stack
    "pandas", "numpy", "scipy", "statsmodels", "matplotlib", "seaborn", "plotly", "altair", "numexpr", "numba",
    "jupyter", "anaconda", "google colab",
    # ML frameworks
    "scikit-learn", "tensorflow", "keras", "pytorch", "xgboost", "lightgbm", "catboost", "mlflow", "wandb", "optuna",
    "hyperopt", "ray", "dask", "onnx", "triton inference server",
    # NLP / LLM
    "nltk", "spacy", "gensim", "transformers", "hugging face", "langchain", "faiss", "pinecone", "milvus", "chroma",
    "openai api", "bert", "gpt", "llama", "mistral", "reranking", "rag", "tokenization", "prompt engineering",
    # CV / Time series
    "opencv", "yolo", "prophet", "arima", "lstm", "timeseries",
    # Data / Warehouses / ETL
    "postgresql", "mysql", "sqlite", "bigquery", "snowflake", "redshift", "databricks", "spark", "pyspark", "hadoop",
    "hive", "kafka", "rabbitmq", "airflow", "prefect", "dbt", "power query", "superset", "metabase",
    # NoSQL / Graph / Cache
    "mongodb", "cassandra", "neo4j", "redis", "elasticsearch", "opensearch",
    # BI / Viz
    "tableau", "power bi", "looker", "looker studio", "ggplot2", "shiny", "excel", "google sheets",
    # Cloud (AWS/GCP/Azure)
    "aws", "s3", "ec2", "lambda", "glue", "athena", "emr", "sagemaker",
    "gcp", "gcs", "vertex ai", "dataflow", "dataproc", "pub/sub", "bigquery ml",
    "azure", "azure ml", "synapse", "data lake", "blob storage",
    # DevOps / MLOps
    "docker", "kubernetes", "terraform", "git", "github actions", "ci/cd",
    # Web / APIs / Apps
    "fastapi", "flask", "django", "streamlit", "gradio", "rest", "graphql", "json", "xml", "regex",
    # Methods / Stats
    "a/b testing", "ab testing", "experiment design", "hypothesis testing", "bayesian", "monte carlo",
    "linear regression", "logistic regression", "svm", "decision trees", "random forest", "pca", "t-sne", "umap",
    "clustering", "kmeans", "dbscan", "hdbscan",
    # Domains / Ops
    "nlp", "ocr", "asr", "mlops", "model serving", "feature engineering", "feature store",
    # Extras
    "power automate", "lookml", "cuda", "cudf"
]]

# Optional synonyms/aliases → canonical skill (all lowercase)
ALIASES: Dict[str, str] = {
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "tsne": "t-sne",
    "k-means": "kmeans",
    "google cloud": "gcp",
    "google cloud storage": "gcs",
    "aws s3": "s3",
    "amazon s3": "s3",
    "amazon ec2": "ec2",
    "google bigquery": "bigquery",
    "llms": "gpt",
    "bert-base": "bert",
    "openai": "openai api",
    "pgsql": "postgresql",
    "yolo8": "yolo",
    "ms excel": "excel",
    "powerbi": "power bi",
    "looker studio (datastudio)": "looker studio",
    "rag pipeline": "rag",
    "statistical testing": "hypothesis testing",
    "rest api": "rest",
    "graphql api": "graphql",
    "ci cd": "ci/cd",
    "ab-testing": "a/b testing",
}

# Pre-compile word-boundary regex for exact phrase hits
_BOUNDARY_PATTERNS: Dict[str, re.Pattern] = {
    skill: re.compile(rf"(?<!\w){re.escape(skill)}(?!\w)", re.IGNORECASE) for skill in SKILLS
}

# Build lookup set that includes aliases for quick normalization
_CANONICAL: Set[str] = set(SKILLS)
_CANONICAL.update(ALIASES.values())


def _normalize_token(t: str) -> str:
    t = t.strip().lower()
    return ALIASES.get(t, t)


def match_skills(text: str, threshold: int = 90) -> List[str]:
    """
    Return a sorted list of canonical skills present in `text`.
    Strategy:
      1) Exact boundary phrase search for each canonical term.
      2) Fuzzy alias & near-miss rescue via RapidFuzz (threshold ~90).
    """
    if not text:
        return []
    text_low = text.lower()

    hits: Set[str] = set()

    # 1) Exact boundary matches
    for skill, rx in _BOUNDARY_PATTERNS.items():
        if rx.search(text_low):
            hits.add(skill)

    # 2) Fuzzy rescue (aliases + common misspellings)
    # We'll sample candidates from (ALIASES keys + SKILLS) to catch near matches
    candidates = list(set(list(ALIASES.keys()) + SKILLS))
    # RapidFuzz process returns (match, score, idx)
    best = process.extract(text_low, candidates, scorer=fuzz.partial_ratio, score_cutoff=threshold, limit=50)
    for cand, score, _ in best:
        hits.add(_normalize_token(cand))

    # Keep only canonical skills we know
    hits = {ALIASES.get(h, h) for h in hits if h in _CANONICAL}

    return sorted(hits)
