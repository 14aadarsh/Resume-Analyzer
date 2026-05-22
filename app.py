import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
from dotenv import load_dotenv

from utils.file_reader import extract_text
from utils.ats_scorer import calculate_ats_score
from utils.grok_analyzer import analyze_resume_with_grok

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ATS Intelligence · Resume Analyzer",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Stylesheet ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0e17;
    color: #c9d1e0;
}
section[data-testid="stSidebar"] {
    background: #0d1120;
    border-right: 1px solid #1e2d45;
}
section[data-testid="stSidebar"] * { color: #8899b4 !important; }
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #e0e8f5 !important;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.stApp { background-color: #0a0e17; }
.block-container { padding-top: 2rem; max-width: 1280px; }

.ats-header { display: flex; align-items: baseline; gap: 14px; margin-bottom: 0.25rem; }
.ats-logo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #00d4aa;
    background: #00d4aa18;
    border: 1px solid #00d4aa33;
    padding: 3px 10px;
    border-radius: 4px;
    letter-spacing: 0.08em;
}
.ats-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #e8f0fe;
    letter-spacing: -0.02em;
    line-height: 1;
}
.ats-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #4a6080;
    margin-bottom: 2rem;
    letter-spacing: 0.04em;
}
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #00d4aa;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.score-block {
    background: #0d1829;
    border: 1px solid #1a2e4a;
    border-radius: 12px;
    padding: 1.8rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.score-block::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #00d4aa, #0090ff);
}
.score-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3.8rem;
    font-weight: 600;
    line-height: 1;
    color: #e8f0fe;
}
.score-denom { font-family: 'JetBrains Mono', monospace; font-size: 1rem; color: #3a5070; }
.score-label { font-size: 0.7rem; letter-spacing: 0.18em; text-transform: uppercase; color: #4a6a90; margin-top: 0.4rem; }
.score-badge {
    display: inline-block;
    margin-top: 0.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    padding: 3px 12px;
    border-radius: 20px;
    letter-spacing: 0.1em;
}
.badge-excellent { background: #00d4aa18; color: #00d4aa; border: 1px solid #00d4aa33; }
.badge-good      { background: #0090ff18; color: #4db8ff; border: 1px solid #0090ff33; }
.badge-average   { background: #f5a62318; color: #f5a623; border: 1px solid #f5a62333; }
.badge-poor      { background: #ff445518; color: #ff6677; border: 1px solid #ff445533; }

.metric-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.55rem 0;
    border-bottom: 1px solid #12203a;
}
.metric-row:last-child { border-bottom: none; }
.metric-name { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #607590; }
.metric-num  { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #c0cfe8; font-weight: 600; }

.tag {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    padding: 2px 9px;
    border-radius: 4px;
    margin: 2px 3px 2px 0;
    letter-spacing: 0.04em;
}
.tag-green { background: #00d4aa12; color: #00d4aa; border: 1px solid #00d4aa22; }
.tag-red   { background: #ff445512; color: #ff7788; border: 1px solid #ff445522; }
.tag-blue  { background: #0090ff12; color: #60b8ff; border: 1px solid #0090ff22; }
.tag-amber { background: #f5a62312; color: #f5c060; border: 1px solid #f5a62322; }

.info-card {
    background: #0d1829;
    border: 1px solid #1a2e4a;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    line-height: 1.6;
    color: #8899b4;
}
.info-card strong { color: #c9d8f0; }

.diff-before {
    background: #1a0d10;
    border-left: 3px solid #ff4455;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #cc7788;
    margin-bottom: 0.3rem;
}
.diff-after {
    background: #0d1a14;
    border-left: 3px solid #00d4aa;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #00d4aa;
}

.stTabs [data-baseweb="tab-list"] {
    background: #0d1120;
    border-bottom: 1px solid #1a2e4a;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    color: #3a5070 !important;
    background: transparent;
    border-bottom: 2px solid transparent;
    padding: 0.6rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    color: #00d4aa !important;
    border-bottom: 2px solid #00d4aa;
    background: transparent;
}
.stButton > button {
    width: 100%;
    background: #00d4aa;
    color: #0a0e17;
    border: none;
    padding: 0.7rem 1.5rem;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    cursor: pointer;
}
.stTextArea textarea, .stTextInput input {
    background: #0d1829 !important;
    border: 1px solid #1e3a5f !important;
    color: #c9d1e0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    border-radius: 8px !important;
}
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid #1e3a5f !important;
    color: #607590 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em !important;
}
header[data-testid="stHeader"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


# ── Chart helpers ──────────────────────────────────────────────────────────────

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono", color="#607590", size=11),
    margin=dict(l=10, r=10, t=36, b=10),
)


def gauge_chart(score: int) -> go.Figure:
    color = "#00d4aa" if score >= 80 else "#4db8ff" if score >= 60 else "#f5a623" if score >= 40 else "#ff4455"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"family": "JetBrains Mono", "color": "#e8f0fe", "size": 42}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#1e3050", "tickfont": {"size": 9}},
            "bar": {"color": color, "thickness": 0.22},
            "bgcolor": "#0d1829",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "#12182a"},
                {"range": [40, 60], "color": "#12202a"},
                {"range": [60, 80], "color": "#0d1e2a"},
                {"range": [80,100], "color": "#0a1e22"},
            ],
            "threshold": {"line": {"color": "#1e3050", "width": 2}, "thickness": 0.8, "value": 60},
        },
    ))
    fig.update_layout(**CHART_THEME, height=240)
    return fig


def breakdown_bar(breakdown: dict) -> go.Figure:
    labels = ["Keyword\nMatch", "Sections", "Formatting", "Action\nVerbs", "Contact\nInfo"]
    values = [
        breakdown["keyword_match"]["score"],
        breakdown["sections"]["score"],
        breakdown["formatting"]["score"],
        breakdown["action_verbs"]["score"],
        breakdown["contact_info"]["score"],
    ]
    weights = ["35%", "20%", "20%", "15%", "10%"]
    colors = ["#00d4aa" if v >= 80 else "#4db8ff" if v >= 60 else "#f5a623" if v >= 40 else "#ff4455"
              for v in values]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors, marker_line_width=0,
        text=[str(v) for v in values],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=11, color="#c0cfe8"),
        customdata=weights,
        hovertemplate="<b>%{x}</b><br>Score: %{y}/100<br>Weight: %{customdata}<extra></extra>",
    ))
    fig.add_hline(y=60, line_dash="dot", line_color="#ff4455", opacity=0.3,
                  annotation_text="threshold · 60",
                  annotation_font_size=9, annotation_font_color="#ff4455")
    fig.update_layout(
        **CHART_THEME, height=320,
        yaxis=dict(range=[0, 115], gridcolor="#0d1829", zeroline=False),
        xaxis=dict(tickfont=dict(size=10)),
        title=dict(text="Score Breakdown", font=dict(size=12, color="#4a6a90")),
    )
    return fig


def keyword_donut(matched: list, missing: list) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=["Matched", "Missing"],
        values=[max(len(matched), 1), max(len(missing), 1)],
        hole=0.65,
        marker_colors=["#00d4aa", "#ff4455"],
        textfont=dict(family="JetBrains Mono", size=10),
        hovertemplate="%{label}: %{value} keywords<extra></extra>",
    ))
    fig.update_layout(
        **CHART_THEME, height=260,
        title=dict(text="Keyword Coverage", font=dict(size=12, color="#4a6a90")),
        showlegend=True,
        legend=dict(font=dict(family="JetBrains Mono", size=9, color="#607590")),
        annotations=[dict(
            text=f"{len(matched)}<br><span style='font-size:9px'>matched</span>",
            x=0.5, y=0.5, font_size=18, showarrow=False,
            font=dict(family="JetBrains Mono", color="#00d4aa"),
        )],
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### WEIGHTS")
    st.markdown("""
| module | weight |
|---|---|
| keyword match | 35% |
| sections | 20% |
| formatting | 20% |
| action verbs | 15% |
| contact info | 10% |
""")

    st.markdown("---")
    st.markdown("### THRESHOLDS")
    st.markdown("""
| band | range |
|---|---|
| excellent | 80 – 100 |
| good | 60 – 79 |
| average | 40 – 59 |
| poor | 0 – 39 |
""")

    st.markdown("---")
    st.caption("TF-IDF · Cosine Similarity · Llama-3.3-70b")

# API key from env only
api_key = os.getenv("GROQ_API_KEY", "")


# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="ats-header">
    <span class="ats-logo">⬡ ATS</span>
    <span class="ats-title">Resume Intelligence</span>
</div>
<p class="ats-sub">// keyword_match · section_detection · nlp_scoring · llm_analysis</p>
""", unsafe_allow_html=True)


# ── Input ──────────────────────────────────────────────────────────────────────

col_upload, col_jd = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown('<p class="section-label">01 · Resume Upload</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop PDF or DOCX", type=["pdf", "docx"], label_visibility="collapsed")
    if uploaded_file:
        st.markdown(f"""
        <div class="info-card" style="margin-top:0.5rem">
            <strong>{uploaded_file.name}</strong><br>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#3a6050">
            ✓ file loaded · ready to parse</span>
        </div>""", unsafe_allow_html=True)

with col_jd:
    st.markdown('<p class="section-label">02 · Job Description (Recommended)</p>', unsafe_allow_html=True)
    jd_text = st.text_area(
        "Paste JD here", height=160, label_visibility="collapsed",
        placeholder="Paste the job description here for accurate keyword matching and role-specific scoring...",
    )
    if jd_text:
        st.caption(f"`{len(jd_text.split())} words` · JD loaded")

st.markdown("<br>", unsafe_allow_html=True)
run_col, _ = st.columns([1, 2])
with run_col:
    analyze_btn = st.button("▶  RUN ANALYSIS", use_container_width=True)


# ── Pipeline ───────────────────────────────────────────────────────────────────

if analyze_btn:
    if not uploaded_file:
        st.error("No resume file detected. Upload a PDF or DOCX to continue.")
        st.stop()

    with st.spinner("Parsing document..."):
        try:
            resume_text = extract_text(uploaded_file)
            if len(resume_text) < 50:
                st.error("Text extraction returned insufficient content. Is the PDF image-scanned?")
                st.stop()
        except Exception as e:
            st.error(f"Parse error: {e}")
            st.stop()

    with st.spinner("Running NLP scoring pipeline..."):
        scores = calculate_ats_score(resume_text, jd_text)

    grok_result = None
    if api_key:
        with st.spinner("Sending to Llama-3.3-70b for deep analysis..."):
            grok_result = analyze_resume_with_grok(resume_text, jd_text, scores, api_key)
    else:
        st.info("GROQ_API_KEY not found in .env file.")

    # ── Overview ──────────────────────────────────────────────────────────────

    st.markdown("---")
    st.markdown('<p class="section-label">03 · Scoring Overview</p>', unsafe_allow_html=True)

    bd    = scores["breakdown"]
    score = scores["final_score"]
    cat   = scores["category"].lower()

    col_gauge, col_score, col_stats = st.columns([1.2, 0.8, 1], gap="large")

    with col_gauge:
        st.plotly_chart(gauge_chart(score), use_container_width=True)

    with col_score:
        st.markdown(f"""
        <div class="score-block">
            <div class="score-label">Overall ATS Score</div>
            <div class="score-val">{score}<span class="score-denom">/100</span></div>
            <div><span class="score-badge badge-{cat}">{scores['category'].upper()}</span></div>
        </div>""", unsafe_allow_html=True)

    with col_stats:
        st.markdown('<p class="section-label" style="margin-top:0.5rem">Key Metrics</p>', unsafe_allow_html=True)
        metrics = [
            ("keyword_match", "Keyword Match"),
            ("sections",      "Section Score"),
            ("formatting",    "Format Score"),
            ("action_verbs",  "Action Verbs"),
            ("contact_info",  "Contact Info"),
        ]
        rows = ""
        for key, label in metrics:
            val = bd[key]["score"]
            c = "#00d4aa" if val >= 80 else "#4db8ff" if val >= 60 else "#f5a623" if val >= 40 else "#ff4455"
            rows += f'<div class="metric-row"><span class="metric-name">{label}</span><span class="metric-num" style="color:{c}">{val}</span></div>'
        st.markdown(f'<div class="info-card" style="padding:0.8rem 1rem">{rows}</div>', unsafe_allow_html=True)

    st.plotly_chart(breakdown_bar(scores["breakdown"]), use_container_width=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────

    st.markdown('<p class="section-label">04 · Detailed Breakdown</p>', unsafe_allow_html=True)
    tab_kw, tab_sec, tab_verbs, tab_fmt, tab_ai = st.tabs([
        "KEYWORDS", "SECTIONS", "ACTION VERBS", "FORMATTING", "AI ANALYSIS"
    ])

    # Keywords
    with tab_kw:
        kw = bd["keyword_match"]
        col1, col2 = st.columns([1, 1.4], gap="large")
        with col1:
            if jd_text:
                st.plotly_chart(keyword_donut(kw["matched_keywords"], kw["missing_keywords"]), use_container_width=True)
                st.markdown(f"""
                <div class="info-card">
                    <div class="metric-row"><span class="metric-name">cosine similarity</span>
                        <span class="metric-num">{kw.get('cosine_similarity',0):.1f}%</span></div>
                    <div class="metric-row"><span class="metric-name">jd keywords total</span>
                        <span class="metric-num">{kw.get('total_jd_keywords',0)}</span></div>
                    <div class="metric-row"><span class="metric-name">matched</span>
                        <span class="metric-num" style="color:#00d4aa">{kw.get('total_matched',0)}</span></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.info("Paste a job description to enable keyword match analysis.")
        with col2:
            if kw["matched_keywords"]:
                st.markdown('<p class="section-label">Matched Keywords</p>', unsafe_allow_html=True)
                st.markdown("".join([f'<span class="tag tag-green">{k}</span>' for k in kw["matched_keywords"]]), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if kw["missing_keywords"]:
                st.markdown('<p class="section-label">Missing — Add These</p>', unsafe_allow_html=True)
                st.markdown("".join([f'<span class="tag tag-red">{k}</span>' for k in kw["missing_keywords"]]), unsafe_allow_html=True)

    # Sections
    with tab_sec:
        sec = bd["sections"]
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown('<p class="section-label">Detected Sections</p>', unsafe_allow_html=True)
            html = "".join([f'<span class="tag tag-green">✓ {s}</span>' for s in sec["found_sections"]])
            st.markdown(html or "<span style='color:#3a5070;font-size:0.8rem'>None detected</span>", unsafe_allow_html=True)
        with col2:
            st.markdown('<p class="section-label">Missing Sections</p>', unsafe_allow_html=True)
            if sec["missing_sections"]:
                st.markdown("".join([f'<span class="tag tag-red">✗ {s}</span>' for s in sec["missing_sections"]]), unsafe_allow_html=True)
            else:
                st.markdown('<span class="tag tag-green">All critical sections present</span>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-card">
            <div class="metric-row">
                <span class="metric-name">critical sections (exp · edu · skills · contact)</span>
                <span class="metric-num">{sec['critical_sections_found']} / 4</span>
            </div>
            <div class="metric-row">
                <span class="metric-name">total sections detected</span>
                <span class="metric-num">{sec['total_sections_found']} / 8</span>
            </div>
        </div>""", unsafe_allow_html=True)

    # Action Verbs
    with tab_verbs:
        av = bd["action_verbs"]
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown('<p class="section-label">Verbs Found in Resume</p>', unsafe_allow_html=True)
            if av["action_verbs_found"]:
                st.markdown("".join([f'<span class="tag tag-green">{v}</span>' for v in av["action_verbs_found"]]), unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#ff7788;font-size:0.8rem">No strong action verbs detected.</span>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            q = av["quantified_achievements"]
            qc = "#00d4aa" if q >= 3 else "#f5a623" if q >= 1 else "#ff4455"
            st.markdown(f"""
            <div class="info-card">
                <div class="metric-row"><span class="metric-name">action verbs count</span>
                    <span class="metric-num">{av['action_verbs_count']}</span></div>
                <div class="metric-row"><span class="metric-name">quantified achievements</span>
                    <span class="metric-num" style="color:{qc}">{q}</span></div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown('<p class="section-label">High-Impact Verbs to Use</p>', unsafe_allow_html=True)
            suggested = ["Engineered","Architected","Optimized","Automated","Deployed",
                         "Spearheaded","Delivered","Reduced","Increased","Migrated"]
            st.markdown("".join([f'<span class="tag tag-blue">{v}</span>' for v in suggested]), unsafe_allow_html=True)
            st.markdown("""
            <div class="info-card" style="margin-top:1rem">
                <strong>Quantification examples</strong><br>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem">
                · "Reduced API latency by 42%"<br>
                · "Trained model on 2M+ records"<br>
                · "Automated pipeline saving 8 hrs/week"<br>
                · "Improved F1 score from 0.71 → 0.89"
                </span>
            </div>""", unsafe_allow_html=True)

    # Formatting
    with tab_fmt:
        fmt = bd["formatting"]
        ci  = bd["contact_info"]
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown('<p class="section-label">Document Stats</p>', unsafe_allow_html=True)
            wc = fmt["word_count"]
            wc_status = "optimal" if 300 <= wc <= 900 else "too short" if wc < 300 else "too long"
            wc_color  = "#00d4aa" if wc_status == "optimal" else "#f5a623"
            st.markdown(f"""
            <div class="info-card">
                <div class="metric-row"><span class="metric-name">word count</span>
                    <span class="metric-num" style="color:{wc_color}">{wc} · {wc_status}</span></div>
                <div class="metric-row"><span class="metric-name">recommended range</span>
                    <span class="metric-num">300 – 900 words</span></div>
            </div>""", unsafe_allow_html=True)
            if fmt["issues"]:
                st.markdown('<p class="section-label" style="margin-top:1rem">Formatting Issues</p>', unsafe_allow_html=True)
                for issue in fmt["issues"]:
                    st.markdown(f'<div class="info-card">⚠ {issue}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="tag tag-green" style="margin-top:1rem;display:inline-block">No critical formatting issues</span>', unsafe_allow_html=True)
        with col2:
            st.markdown('<p class="section-label">Contact Information</p>', unsafe_allow_html=True)
            for item in ci["found"]:
                st.markdown(f'<span class="tag tag-green">✓ {item}</span>', unsafe_allow_html=True)
            for item in ci["missing"]:
                st.markdown(f'<span class="tag tag-red">✗ {item} missing</span>', unsafe_allow_html=True)

    # AI Analysis
    with tab_ai:
        if not api_key:
            st.markdown("""
            <div class="info-card">
                <strong>Groq API key not found</strong><br><br>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem">
                Add GROQ_API_KEY to your .env file<br><br>
                Get free key at groq.com → API Keys
                </span>
            </div>""", unsafe_allow_html=True)

        elif grok_result and grok_result.get("success"):
            analysis = grok_result["analysis"]
            st.markdown('<p class="section-label">Model Assessment</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-card">{analysis.get("overall_assessment","")}</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="info-card" style="border-left:3px solid #00d4aa;border-radius:0 10px 10px 0">
                <strong style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#00d4aa">VERDICT</strong><br>
                {analysis.get("final_verdict","")}
            </div>""", unsafe_allow_html=True)

            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.markdown('<p class="section-label">Strengths</p>', unsafe_allow_html=True)
                for s in analysis.get("top_strengths", []):
                    st.markdown(f'<span class="tag tag-green">✓ {s}</span>', unsafe_allow_html=True)
                st.markdown('<p class="section-label" style="margin-top:1rem">Keywords to Add</p>', unsafe_allow_html=True)
                for k in analysis.get("keyword_suggestions", []):
                    st.markdown(f'<span class="tag tag-amber">+ {k}</span>', unsafe_allow_html=True)
                ats_tips = analysis.get("ats_tips", [])
                if ats_tips:
                    st.markdown('<p class="section-label" style="margin-top:1rem">ATS Tips</p>', unsafe_allow_html=True)
                    for tip in ats_tips:
                        st.markdown(f'<div class="info-card" style="font-size:0.78rem">→ {tip}</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<p class="section-label">Critical Improvements</p>', unsafe_allow_html=True)
                for imp in analysis.get("critical_improvements", []):
                    st.markdown(f"""
                    <div class="info-card" style="border-left:3px solid #f5a623;border-radius:0 10px 10px 0;margin-bottom:0.5rem">
                        <strong style="font-size:0.8rem">{imp.get('issue','')}</strong><br>
                        <span style="font-size:0.75rem;color:#607590">{imp.get('explanation','')}</span><br>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#f5c060">
                        fix → {imp.get('fix','')}
                        </span>
                    </div>""", unsafe_allow_html=True)
                sec_sug = analysis.get("section_suggestions", "")
                if sec_sug:
                    st.markdown('<p class="section-label" style="margin-top:1rem">Section Notes</p>', unsafe_allow_html=True)
                    st.markdown(f'<div class="info-card">{sec_sug}</div>', unsafe_allow_html=True)

            rewrites = analysis.get("bullet_point_rewrites", [])
            if rewrites:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<p class="section-label">Bullet Point Rewrites</p>', unsafe_allow_html=True)
                for rw in rewrites:
                    ca, cb = st.columns(2, gap="medium")
                    with ca:
                        st.markdown(f'<div class="diff-before">− {rw.get("original","")}</div>', unsafe_allow_html=True)
                    with cb:
                        st.markdown(f'<div class="diff-after">+ {rw.get("improved","")}</div>', unsafe_allow_html=True)

        elif grok_result and not grok_result.get("success"):
            st.error(f"API error: {grok_result.get('error','Unknown')}")

    # ── Download ──────────────────────────────────────────────────────────────

    st.markdown("---")
    report = f"""ATS INTELLIGENCE · RESUME ANALYSIS REPORT
{'='*50}
Overall Score  : {scores['final_score']}/100  [{scores['category'].upper()}]

SCORE BREAKDOWN
{'─'*30}
Keyword Match  : {bd['keyword_match']['score']}/100   (weight: 35%)
Sections       : {bd['sections']['score']}/100   (weight: 20%)
Formatting     : {bd['formatting']['score']}/100   (weight: 20%)
Action Verbs   : {bd['action_verbs']['score']}/100   (weight: 15%)
Contact Info   : {bd['contact_info']['score']}/100   (weight: 10%)

MISSING KEYWORDS
{'─'*30}
{', '.join(bd['keyword_match'].get('missing_keywords',[])) or 'None'}

MISSING SECTIONS
{'─'*30}
{', '.join(bd['sections'].get('missing_sections',[])) or 'None'}

FORMATTING ISSUES
{'─'*30}
{chr(10).join(bd['formatting'].get('issues',[])) or 'None'}
"""
    st.download_button("↓  DOWNLOAD REPORT  ·  .txt", data=report,
                       file_name="ats_report.txt", mime="text/plain")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:3rem;padding:1rem 0;
    border-top:1px solid #0d1829;
    font-family:'JetBrains Mono',monospace;
    font-size:0.6rem;color:#1e3050;letter-spacing:0.1em">
ATS INTELLIGENCE · TF-IDF · COSINE SIMILARITY · NLP PIPELINE · LLAMA-3.3-70B
</div>
""", unsafe_allow_html=True)