# Automated Resume Screener (NLP)

Reads many **resume PDFs**, compares them to a **Job Description**, ranks them, and explains **why**.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- PDF parsing: **PyMuPDF** (primary) with **pdfminer.six** fallback
- Skills: curated list + fuzzy matching via **rapidfuzz**
- App: **Streamlit**
- No paid APIs required.

## 🗂️ Project Structure
