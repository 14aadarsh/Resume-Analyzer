import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# NLTK data download (first time only)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ----------------------------
# Constants
# ----------------------------

SECTION_KEYWORDS = {
    "experience": ["experience", "work experience", "employment", "work history", "professional experience", "internship"],
    "education": ["education", "academic", "qualification", "degree", "university", "college", "school"],
    "skills": ["skills", "technical skills", "core competencies", "technologies", "tools", "expertise"],
    "projects": ["projects", "personal projects", "academic projects", "portfolio"],
    "certifications": ["certifications", "certificates", "courses", "training", "licenses"],
    "contact": ["contact", "email", "phone", "linkedin", "github", "address", "mobile"],
    "summary": ["summary", "objective", "profile", "about me", "career objective"],
    "achievements": ["achievements", "awards", "honors", "accomplishments", "recognition"],
}

ACTION_VERBS = [
    "developed", "designed", "implemented", "built", "created", "managed", "led", "improved",
    "optimized", "analyzed", "deployed", "automated", "architected", "engineered", "delivered",
    "launched", "collaborated", "coordinated", "mentored", "trained", "increased", "reduced",
    "achieved", "established", "executed", "spearheaded", "streamlined", "transformed",
    "researched", "presented", "published", "contributed", "maintained", "monitored",
    "resolved", "integrated", "migrated", "scaled", "tested", "documented", "supervised"
]

CONTACT_PATTERNS = {
    "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    "phone": r'(\+?\d[\d\s\-().]{8,15}\d)',
    "linkedin": r'linkedin\.com/in/[a-zA-Z0-9\-]+',
    "github": r'github\.com/[a-zA-Z0-9\-]+',
}

QUANTIFICATION_PATTERNS = [
    r'\d+%',           # percentages
    r'\$[\d,]+',       # dollar amounts
    r'\d+\+',          # numbers with plus
    r'\d+x',           # multipliers
    r'\d+ (users|clients|customers|projects|teams|members|employees)',
    r'(increased|decreased|reduced|improved|grew|boosted).*\d+',
]


# ----------------------------
# Helper Functions
# ----------------------------

def clean_text(text: str) -> str:
    """Text clean karo — lowercase, special chars remove."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_keywords(text: str) -> list:
    """Important keywords nikalo text se."""
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(clean_text(text))
    keywords = [t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 2]
    return keywords


# ----------------------------
# Scoring Functions
# ----------------------------

def score_keyword_match(resume_text: str, jd_text: str) -> dict:
    """
    TF-IDF + Cosine Similarity se keyword match score nikalo.
    Weight: 35%
    """
    if not jd_text.strip():
        return {"score": 50, "matched_keywords": [], "missing_keywords": [], "detail": "No JD provided"}

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform([clean_text(resume_text), clean_text(jd_text)])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        cosine_score = round(similarity * 100, 2)
    except:
        cosine_score = 0

    # Exact keyword matching
    jd_keywords = set(get_keywords(jd_text))
    resume_keywords = set(get_keywords(resume_text))
    matched = jd_keywords.intersection(resume_keywords)
    missing = jd_keywords - resume_keywords

    # Top missing keywords (by importance)
    missing_list = list(missing)[:15]
    matched_list = list(matched)[:15]

    # Final score = cosine similarity weighted
    final_score = min(100, round(cosine_score * 1.5))  # scale up slightly

    return {
        "score": final_score,
        "cosine_similarity": cosine_score,
        "matched_keywords": matched_list,
        "missing_keywords": missing_list,
        "total_jd_keywords": len(jd_keywords),
        "total_matched": len(matched),
    }


def score_sections(resume_text: str) -> dict:
    """
    Important sections detect karo resume mein.
    Weight: 20%
    """
    resume_lower = resume_text.lower()
    found_sections = {}
    missing_sections = []

    for section, keywords in SECTION_KEYWORDS.items():
        found = any(kw in resume_lower for kw in keywords)
        found_sections[section] = found
        if not found:
            missing_sections.append(section)

    # Critical sections
    critical = ["experience", "education", "skills", "contact"]
    critical_found = sum(1 for s in critical if found_sections.get(s, False))
    critical_score = (critical_found / len(critical)) * 100

    # All sections
    all_found = sum(found_sections.values())
    all_score = (all_found / len(SECTION_KEYWORDS)) * 100

    final_score = round((critical_score * 0.7) + (all_score * 0.3))

    return {
        "score": final_score,
        "found_sections": [k for k, v in found_sections.items() if v],
        "missing_sections": missing_sections,
        "critical_sections_found": critical_found,
        "total_sections_found": all_found,
    }


def score_contact_info(resume_text: str) -> dict:
    """
    Contact information check karo.
    Weight: 10%
    """
    found_contacts = {}
    for contact_type, pattern in CONTACT_PATTERNS.items():
        match = re.search(pattern, resume_text, re.IGNORECASE)
        found_contacts[contact_type] = bool(match)

    found_count = sum(found_contacts.values())
    score = round((found_count / len(CONTACT_PATTERNS)) * 100)

    return {
        "score": score,
        "found": [k for k, v in found_contacts.items() if v],
        "missing": [k for k, v in found_contacts.items() if not v],
    }


def score_action_verbs(resume_text: str) -> dict:
    """
    Action verbs aur quantified achievements check karo.
    Weight: 15%
    """
    resume_lower = resume_text.lower()

    # Action verbs found
    found_verbs = [v for v in ACTION_VERBS if v in resume_lower]
    verb_score = min(100, len(found_verbs) * 10)  # 10 verbs = 100

    # Quantification check
    quant_found = 0
    for pattern in QUANTIFICATION_PATTERNS:
        matches = re.findall(pattern, resume_lower)
        quant_found += len(matches)
    quant_score = min(100, quant_found * 15)

    final_score = round((verb_score * 0.5) + (quant_score * 0.5))

    return {
        "score": final_score,
        "action_verbs_found": found_verbs[:10],
        "action_verbs_count": len(found_verbs),
        "quantified_achievements": quant_found,
    }


def score_formatting(resume_text: str) -> dict:
    """
    Basic formatting aur length check.
    Weight: 20%
    """
    issues = []
    score = 100

    # Word count check
    word_count = len(resume_text.split())
    if word_count < 200:
        issues.append("Resume bahut chhota hai (200 words se kam)")
        score -= 30
    elif word_count > 1200:
        issues.append("Resume bahut lamba hai (1200 words se zyada — 1-2 pages ideal)")
        score -= 15

    # Email check
    if not re.search(CONTACT_PATTERNS["email"], resume_text):
        issues.append("Email address nahi mila")
        score -= 20

    # Phone check
    if not re.search(CONTACT_PATTERNS["phone"], resume_text):
        issues.append("Phone number nahi mila")
        score -= 10

    # Special characters (ATS unfriendly)
    special_chars = len(re.findall(r'[★●■◆▶✓✔➤]', resume_text))
    if special_chars > 5:
        issues.append(f"Bahut zyada special characters ({special_chars}) — ATS parse nahi kar sakta")
        score -= 15

    # Tables/columns detection (approximate)
    tab_count = resume_text.count('\t')
    if tab_count > 20:
        issues.append("Table/column format detect hua — ATS mein parsing issues ho sakte hain")
        score -= 10

    return {
        "score": max(0, score),
        "word_count": word_count,
        "issues": issues,
    }


# ----------------------------
# Master Scoring Function
# ----------------------------

def calculate_ats_score(resume_text: str, jd_text: str = "") -> dict:
    """
    Poora ATS score calculate karo — sab modules combine karke.
    """
    # Individual scores
    keyword_result = score_keyword_match(resume_text, jd_text)
    section_result = score_sections(resume_text)
    contact_result = score_contact_info(resume_text)
    action_result = score_action_verbs(resume_text)
    format_result = score_formatting(resume_text)

    # Weighted final score
    weights = {
        "keyword": 0.35,
        "sections": 0.20,
        "formatting": 0.20,
        "action_verbs": 0.15,
        "contact": 0.10,
    }

    final_score = round(
        keyword_result["score"] * weights["keyword"] +
        section_result["score"] * weights["sections"] +
        format_result["score"] * weights["formatting"] +
        action_result["score"] * weights["action_verbs"] +
        contact_result["score"] * weights["contact"]
    )

    # Score category
    if final_score >= 80:
        category = "Excellent"
        emoji = "🟢"
    elif final_score >= 60:
        category = "Good"
        emoji = "🟡"
    elif final_score >= 40:
        category = "Average"
        emoji = "🟠"
    else:
        category = "Poor"
        emoji = "🔴"

    return {
        "final_score": final_score,
        "category": category,
        "emoji": emoji,
        "breakdown": {
            "keyword_match": keyword_result,
            "sections": section_result,
            "contact_info": contact_result,
            "action_verbs": action_result,
            "formatting": format_result,
        },
        "weights": weights,
    }
