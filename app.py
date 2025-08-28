# app.py
from __future__ import annotations
import streamlit as st

st.set_page_config(page_title="Automated Resume Screener (NLP)", layout="wide")

# --- Show import errors on the page so we never get a blank screen ---
try:
    from parser import extract_text_from_pdf_bytes
    from ranker import rank_resumes, ResumeRecord
    from skills import match_skills  # for JD skills preview
except Exception as e:
    st.error("❌ Import failed. Install missing packages below, then rerun.")
    st.exception(e)
    st.stop()

# ----------------- Header -----------------
st.title("📄 Automated Resume Screener — NLP")
st.caption("Embeddings: sentence-transformers/all-MiniLM-L6-v2 • No paid APIs • Local compute")

with st.expander("How it works"):
    st.markdown("""
- **Similarity (0–1)**: semantic cosine similarity between the JD and each resume.
- **Skill Overlap (0–1)**: fraction of JD skills found in a resume.
- **Years Match (0–1)**: compares stated years in resume vs JD (neutral=1 if JD has no years).
- **Final score** = `0.6*similarity + 0.3*skill_overlap + 0.1*years_match`.
- **Explanation**: top 10 matched skills + 2 most JD-similar sentences from each resume.
""")

left, right = st.columns([1, 2], gap="large")

# ----------------- Inputs with keys (so we can reset them) -----------------
with left:
    st.subheader("1) Job Description")
    jd_file = st.file_uploader(
        "Upload JD (.txt preferred)",
        type=["txt"],
        accept_multiple_files=False,
        key="jd_upl",
    )
    jd_text_area = st.text_area(
        "…or paste JD text here",
        height=200,
        placeholder="Paste JD text if no file…",
        key="jd_text",
    )
    dedupe = st.checkbox("Deduplicate resumes (by content hash)", value=True, key="dedupe")
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        run_btn = st.button("Run Screening", type="primary")
    with col_btn2:
        clear_btn = st.button("Clear All")

with right:
    st.subheader("2) Resume PDFs")
    resume_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="res_upl",
    )

# ----------------- Clear All behavior -----------------
if clear_btn:
    st.session_state["jd_upl"] = None
    st.session_state["jd_text"] = ""
    st.session_state["res_upl"] = None
    st.session_state["dedupe"] = True
    st.rerun()

# ----------------- JD Skills Preview (before running) -----------------
jd_preview_text = ""
if jd_file is not None:
    try:
        jd_preview_text = jd_file.getvalue().decode("utf-8", errors="ignore")
    except Exception:
        jd_preview_text = ""
if not jd_preview_text and st.session_state.get("jd_text"):
    jd_preview_text = st.session_state["jd_text"].strip()

if jd_preview_text:
    jd_preview_skills = match_skills(jd_preview_text)
    with st.expander(f"🔎 JD skills detected ({len(jd_preview_skills)})", expanded=False):
        if jd_preview_skills:
            st.write(", ".join(jd_preview_skills))
        else:
            st.caption("No known skills detected yet — try pasting more of the JD.")

# ----------------- Run pipeline -----------------
if run_btn:
    # 1) JD text (file wins; else textarea)
    jd_text = ""
    if st.session_state.get("jd_upl") is not None:
        try:
            jd_text = st.session_state["jd_upl"].getvalue().decode("utf-8", errors="ignore")
        except Exception:
            jd_text = ""
    if not jd_text and st.session_state.get("jd_text"):
        jd_text = st.session_state["jd_text"].strip()

    if not jd_text:
        st.error("Please provide a Job Description (upload .txt or paste text).")
        st.stop()

    # 2) Resumes
    if not resume_files:
        st.error("Please upload at least one resume PDF.")
        st.stop()

    resumes = []
    for up in resume_files:
        try:
            raw = up.read()
            text = extract_text_from_pdf_bytes(raw)
            if not text:
                st.warning(f"No text extracted from {up.name} (may be a scanned image-only PDF).")
            resumes.append(ResumeRecord(file_name=up.name, text=text))
        except Exception as e:
            st.warning(f"Failed reading {up.name}: {e}")

    if not resumes:
        st.error("No readable resumes found.")
        st.stop()

    with st.spinner("Embedding and scoring…"):
        df = rank_resumes(jd_text, resumes, drop_duplicates=st.session_state.get("dedupe", True))

    if df.empty:
        st.info("No results to display.")
        st.stop()

    # 3) Results
    st.subheader("3) Ranked Results")
    show_cols = [
        "file",
        "name",
        "email",
        "final_score",
        "similarity",
        "skill_overlap",
        "years_match",
        "num_skills_matched",
        "top_skills",
        "is_duplicate",
    ]
    st.dataframe(df[show_cols], use_container_width=True, height=420)

    with st.expander("View matching sentences per resume"):
        for i, row in df.iterrows():
            st.markdown(f"**{i+1}. {row['file']}** — {row.get('name') or 'Unknown'} • {row.get('email') or 'N/A'}")
            sents = row.get("match_sentences", [])
            if sents:
                st.write("Most JD-similar sentences:")
                for s in sents:
                    st.write(f"- {s}")
            else:
                st.write("_No sentences extracted (document may be too short or image-only PDF)._")
            st.markdown("---")

    # CSV download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, file_name="resume_ranking.csv", mime="text/csv")
