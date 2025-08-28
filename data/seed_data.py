# data/seed_data.py
"""
Create a small seed dataset to test the pipeline WITHOUT internet.
- Generates: data/samples/jd.txt + 5 resume PDFs (text-based) using PyMuPDF
Run:
    python data/seed_data.py
"""

from __future__ import annotations
import os
from pathlib import Path

try:
    import fitz  # PyMuPDF
except Exception as e:
    raise SystemExit("PyMuPDF is required to generate sample PDFs. Install with: pip install pymupdf") from e


BASE = Path(__file__).resolve().parent
SAMPLES = BASE / "samples"
SAMPLES.mkdir(parents=True, exist_ok=True)


def _write_pdf(path: Path, text: str):
    doc = fitz.open()
    page = doc.new_page()
    rect = page.rect
    margin = 36  # 0.5 inch
    bbox = fitz.Rect(margin, margin, rect.width - margin, rect.height - margin)
    page.insert_textbox(bbox, text, fontsize=11, fontname="helv", align=0)
    doc.save(path.as_posix())
    doc.close()


def make_samples():
    # --- JD ---
    jd = """Data Scientist (NLP/Analytics) — West Palm Beach, FL

We are seeking a Data Scientist with 3+ years of experience to build NLP and analytics products.
Must-have: Python, Pandas, NumPy, scikit-learn, SQL, NLP, Transformers/Hugging Face, LangChain, RAG.
Strong plus: PyTorch, TensorFlow, Docker, Git, Airflow, dbt, Spark/Databricks, AWS (S3, EC2, Lambda), Tableau/Power BI.
Nice to have: FastAPI/Flask, MLflow, A/B testing, Experiment Design, Kubernetes, Vertex AI.

Responsibilities:
- Build and deploy NLP pipelines (tokenization, BERT/GPT, embeddings, retrieval, RAG).
- Design ETL with Airflow/dbt, build dashboards (Tableau/Power BI).
- Partner with product to run A/B tests with statistical rigor.

"""
    (SAMPLES / "jd.txt").write_text(jd, encoding="utf-8")

    # --- Resumes (5) ---
    r1 = """JANE DOE
Data Scientist • jane.doe@example.com • West Palm Beach, FL

SUMMARY
Data Scientist with 4 years experience in NLP and analytics. Built RAG pipelines using LangChain, FAISS, and Hugging Face Transformers.
Stack: Python, Pandas, NumPy, scikit-learn, PyTorch, TensorFlow, Docker, Git, Airflow, dbt, Spark, Databricks, AWS (S3, EC2, Lambda).
Delivered A/B tests with hypothesis testing; dashboards in Tableau and Power BI. Deployed FastAPI services; tracked with MLflow.
"""
    r2 = """JOHN SMITH
Machine Learning Engineer • john.smith@mlmail.com • Miami, FL

EXPERIENCE
3+ years building ML systems. Strong in Python, scikit-learn, XGBoost, LightGBM, SQL, Postgres.
Built NLP classifiers with spaCy and Transformers; deployed with Docker, Kubernetes on AWS. Orchestrated jobs with Airflow.
Created BI dashboards with Tableau and Power BI; implemented experiment design and A/B testing culture.
"""
    r3 = """PRIYA KUMAR
Data Analyst → Data Scientist • priya.kumar@data.com

PROFILE
2 years of analytics with Pandas, SQL (BigQuery, Snowflake). Python, Seaborn, Plotly, Statsmodels.
NLP interest: NLTK, Gensim. Some LangChain prototypes. Deployed Streamlit apps. Git/GitHub Actions, Docker basics.
"""
    r4 = """ALAN T
Senior ML Engineer • alan.t@engineer.ai

OVERVIEW
6 years experience building end-to-end ML with TensorFlow and PyTorch. RAG using Pinecone, Transformers, and LangChain.
Strong on GCP (GCS, Vertex AI, Dataflow). Spark/Databricks, Airflow, dbt, MLflow. REST/GraphQL APIs with FastAPI and Flask.
"""
    r5 = """MARIA LOPEZ
NLP Researcher • maria.lopez@lab.org

SUMMARY
3 years in NLP focusing on BERT/GPT, reranking, tokenization, and RAG. Python, Hugging Face, FAISS, Chroma.
MLOps: Docker, Kubernetes; Monitoring with Weights & Biases; experiments with Optuna. A/B testing with Bayesian approaches.
"""

    resumes = [("resume_jane_doe.pdf", r1), ("resume_john_smith.pdf", r2), ("resume_priya_kumar.pdf", r3),
               ("resume_alan_t.pdf", r4), ("resume_maria_lopez.pdf", r5)]

    for fname, text in resumes:
        _write_pdf(SAMPLES / fname, text)

    print(f"✅ Wrote seed files to: {SAMPLES}")


if __name__ == "__main__":
    make_samples()
