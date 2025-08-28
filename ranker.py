# ranker.py
"""
Ranking pipeline:
- Embed JD & resumes using sentence-transformers (all-MiniLM-L6-v2).
- Compute cosine similarity (0–1).
- Compute skill_overlap = |resume∩JD| / max(1, |JD skills|)
- Compute years_match in [0,1] by comparing 'X years' in resume vs JD (if JD has years; else 1.0).
- final = 0.6*similarity + 0.3*skill_overlap + 0.1*years_match
Also:
- Top-10 matched skills for each resume.
- Two most JD-similar sentences from the resume.
- Simple deduping via text hash (flag or drop).
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from skills import match_skills
from parser import extract_email_and_name, hash_text

# Cache model globally to avoid re-loading
_MODEL = None

def _load_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        # Small, fast, good quality for semantic similarity
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL


def _embed(texts: List[str]) -> np.ndarray:
    """Return L2-normalized embeddings."""
    model = _load_model()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """With normalized vectors, cosine similarity is dot product in [−1,1]. Clip to [0,1]."""
    sim = float(np.dot(a, b))
    return max(0.0, min(1.0, (sim + 1) / 2.0)) if sim < 0 else min(1.0, sim)  # model outputs are already >=0 often


_YEARS_RE = re.compile(r"\b(\d{1,2})\s*(\+)?\s*(?:years|year|yrs|yr)\b", re.IGNORECASE)

def _extract_years(text: str) -> Optional[int]:
    """
    Return the maximum stated years-of-experience integer found in text, else None.
    Heuristic: looks for 'X years/yrs'.
    """
    years = []
    for m in _YEARS_RE.finditer(text or ""):
        try:
            years.append(int(m.group(1)))
        except Exception:
            pass
    return max(years) if years else None


_SENT_SPLIT = re.compile(r"(?<=[\.!?])\s+(?=[A-Z0-9])")

def _top_similar_sentences(resume_text: str, jd_emb: np.ndarray, k: int = 2) -> List[str]:
    """Pick k sentences from resume that are most semantically similar to JD."""
    if not resume_text:
        return []
    # Keep sentences of reasonable length (avoid super short/long)
    sents = [s.strip() for s in _SENT_SPLIT.split(resume_text) if 40 <= len(s.strip()) <= 300]
    if not sents:
        # fallback: chunk by lines
        sents = [line.strip() for line in resume_text.splitlines() if 30 <= len(line.strip()) <= 300]
    if not sents:
        return []
    emb_sents = _embed(sents)
    sims = emb_sents @ jd_emb  # cosine because normalized
    top_idx = list(np.argsort(sims)[::-1][:k])
    return [sents[i] for i in top_idx]


@dataclass
class ResumeRecord:
    file_name: str
    text: str


def rank_resumes(
    jd_text: str,
    resumes: List[ResumeRecord],
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Rank resumes vs a Job Description text.
    Returns a DataFrame with scores & explanations.
    """
    jd_text = jd_text or ""
    jd_skills = set(match_skills(jd_text))
    jd_years = _extract_years(jd_text)
    jd_emb = _embed([jd_text])[0]

    rows = []
    seen_hashes: Dict[str, int] = {}

    for rec in resumes:
        txt = rec.text or ""
        txt_hash = hash_text(txt)
        is_dup = txt_hash in seen_hashes

        # Semantics
        resume_emb = _embed([txt])[0]
        similarity = _cosine_sim(resume_emb, jd_emb)  # 0–1

        # Skills
        rskills = set(match_skills(txt))
        overlap = len(rskills & jd_skills) / max(1, len(jd_skills))  # 0–1

        # Years
        r_years = _extract_years(txt)
        if jd_years is None:
            years_match = 1.0  # neutral if JD does not specify
        else:
            if r_years is None:
                years_match = 0.0
            else:
                years_match = min(1.0, r_years / max(1, jd_years))

        # Explanation: top skills + 2 most similar sentences
        top_skills = list((rskills & jd_skills))  # prioritize overlap for explanation
        # backfill with other resume skills if fewer than 10
        if len(top_skills) < 10:
            extras = [s for s in sorted(rskills) if s not in top_skills]
            top_skills.extend(extras[: 10 - len(top_skills)])
        top_skills = top_skills[:10]

        match_sents = _top_similar_sentences(txt, jd_emb, k=2)

        # Final score
        final = 0.6 * similarity + 0.3 * overlap + 0.1 * years_match
        final = float(max(0.0, min(1.0, final)))

        email, name = extract_email_and_name(txt)

        rows.append({
            "file": rec.file_name,
            "name": name,
            "email": email,
            "similarity": round(float(similarity), 4),
            "skill_overlap": round(float(overlap), 4),
            "years_match": round(float(years_match), 4),
            "final_score": round(final, 4),
            "num_skills_matched": len(rskills),
            "top_skills": ", ".join(top_skills),
            "match_sentences": match_sents,
            "text_hash": txt_hash,
            "is_duplicate": is_dup,
        })

        if txt_hash not in seen_hashes:
            seen_hashes[txt_hash] = 1

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Deduping: if requested, drop duplicates by text hash keeping highest final_score
    if drop_duplicates:
        df = df.sort_values("final_score", ascending=False).drop_duplicates(subset=["text_hash"]).reset_index(drop=True)

    # Sort final ranking
    df = df.sort_values(["final_score", "similarity", "skill_overlap"], ascending=False).reset_index(drop=True)
    return df
