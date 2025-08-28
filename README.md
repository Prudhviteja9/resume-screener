📄 Automated Resume Screener (NLP)

An AI-powered resume screener built with Python + Streamlit + NLP embeddings.
It reads multiple resumes (PDFs), compares them against a Job Description, ranks candidates, and explains why — highlighting skills & matching sentences.

👉 🎯 Live Demo
 ← (https://resume-screener-w3wdvsslj5wxmyaun2qwkf.streamlit.app/)

✨ Features

📑 PDF Resume Parsing → extracts text from resumes using PyMuPDF (with pdfminer fallback).

🧠 Semantic Matching → sentence embeddings (all-MiniLM-L6-v2) + cosine similarity.

🛠 Skill Overlap Detection → matches 100+ DS/ML/Analytics/Cloud skills.

📊 Scoring Formula

final_score = 0.6*similarity + 0.3*skill_overlap + 0.1*years_match


🔎 Explanation for Recruiters

Top 10 skills matched

2 most JD-similar sentences from each resume

🧹 Deduplication → detects duplicate resumes.

📥 Download CSV → ranked results exportable for HR teams.

🚀 Live Demo

👉 Try it here:https://resume-screener-w3wdvsslj5wxmyaun2qwkf.streamlit.app/

🛠️ Run Locally

Clone the repo and install requirements:

git clone https://github.com/Prudhviteja9/resume-screener.git
cd resume-screener

# create virtual env (optional)
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Mac/Linux

pip install -r requirements.txt


Run the app:

streamlit run app.py

📂 Project Structure
resume-screener/
│── app.py               # Streamlit app
│── parser.py            # PDF text extraction
│── ranker.py            # Embedding + scoring
│── skills.py            # Skills list + matcher
│── requirements.txt     # Dependencies
│── README.md            # Documentation
│── data/
│   ├── seed_data.py     # Generate sample resumes/JD
│   └── samples/         # Example resumes + JD

⚖️ Limitations & Ethics

⚠️ This is a demo project — not for production HR use.

May miss skills not in the dictionary.

Doesn’t handle image-only resumes unless OCR is added.

Should not be used as the sole hiring filter; bias and fairness must be considered.

🙌 Credits

Built with ❤️ by Prudhvi Teja
Tools: Python · Streamlit · pandas · sentence-transformers · PyMuPDF · rapidfuzz

