"""
Social Media Mental Health Risk Assessment — Streamlit demo
Run: streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
METHOD3 = ROOT / "method3"
for p in (ROOT, METHOD3):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Trained XGBoost artifacts live under method2/ (produced by method2/XGBoost.py).
MODEL_DIR = (ROOT / "method2") if (ROOT / "method2").is_dir() else ROOT
XGB_MODEL = MODEL_DIR / "xgb_filter.pkl"
XGB_THRESHOLD = MODEL_DIR / "xgb_threshold.pkl"
XGB_COLS = MODEL_DIR / "xgb_feature_cols.pkl"
RAW_CSV = ROOT / "data" / "Time_Wasters_on_Social_Media.csv"

# Columns to z-score (must match preprocessing notebook exactly)
NUMERIC_COLS = [
    "Total_Time_Spent", "Number_of_Sessions", "Number_of_Videos_Watched",
    "Scroll_Rate", "Engagement", "Self_Control", "Satisfaction"
]

from retriever import ChunkIndex, RetrievalHit, generate_query


def _xgb_artifacts_exist() -> bool:
    return all(p.is_file() for p in (XGB_MODEL, XGB_THRESHOLD, XGB_COLS))

SAMPLE_IDS = [572, 710, 715, 659, 689]


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def load_xgb_bundle():
    """Z-score stats always come from RAW_CSV; classifier weights only if .pkl files exist."""
    if not RAW_CSV.is_file():
        raise FileNotFoundError(f"Missing z-score reference CSV: {RAW_CSV}")
    raw = pd.read_csv(RAW_CSV)
    raw.columns = raw.columns.str.strip().str.replace(" ", "_", regex=False)
    stats = {
        col: {"mean": float(raw[col].mean()), "std": float(raw[col].std())}
        for col in NUMERIC_COLS
    }
    if not _xgb_artifacts_exist():
        return None, None, None, stats
    return (
        joblib.load(XGB_MODEL),
        float(joblib.load(XGB_THRESHOLD)),
        joblib.load(XGB_COLS),
        stats,
    )

@st.cache_resource
def load_index():
    return ChunkIndex(root=ROOT)

@st.cache_resource
def load_profiles() -> pd.DataFrame:
    p = ROOT / "data" / "behavior_profile_dataset.csv"
    if not p.is_file():
        raise FileNotFoundError(f"Missing profile dataset: {p}")
    df = pd.read_csv(p)
    df.columns = df.columns.str.strip()
    return df


def _pick_five_high_risk(df: pd.DataFrame) -> pd.DataFrame:
    model = joblib.load(XGB_MODEL)
    threshold = joblib.load(XGB_THRESHOLD)
    cols = joblib.load(XGB_COLS)
    probs = model.predict_proba(df[cols])[:, 1]
    mask = probs >= threshold
    result = df[mask].copy()
    result["xgb_risk_prob"] = probs[mask]
    return result.sort_values("xgb_risk_prob", ascending=False).head(5)


def _pick_subset(df: pd.DataFrame) -> pd.DataFrame:
    if _xgb_artifacts_exist():
        return _pick_five_high_risk(df)
    return df.loc[SAMPLE_IDS]


# ── Feature engineering ───────────────────────────────────────────────────────

def z_score(raw: float, col: str, stats: dict) -> float:
    """Standardize using training data mean/std (same as StandardScaler in notebook)."""
    return (raw - stats[col]['mean']) / stats[col]['std']


def build_feature_row(
    total_time: float, sessions: float, videos: float,
    scroll_rate: float, engagement: float,
    self_control: float, satisfaction: float,
    watch_reasons: list[str],
    video_cats: list[str],
    stats: dict,
) -> dict:
    """
    Replicates the preprocessing notebook exactly:
      z-score each raw input → compute composites
    """
    z_time       = z_score(total_time,   'Total_Time_Spent',        stats)
    z_sessions   = z_score(sessions,     'Number_of_Sessions',      stats)
    z_videos     = z_score(videos,       'Number_of_Videos_Watched', stats)
    z_scroll     = z_score(scroll_rate,  'Scroll_Rate',             stats)
    z_engagement = z_score(engagement,   'Engagement',              stats)
    z_sc         = z_score(self_control, 'Self_Control',            stats)
    z_sat        = z_score(satisfaction, 'Satisfaction',            stats)

    usage_score       = (z_time + z_sessions + z_videos) / 3
    interaction_score = (z_scroll + z_engagement) / 2
    scroll_rate_z     = z_scroll
    self_control_z    = z_sc
    self_reg_risk     = -((z_sc + z_sat) / 2)

    row: dict = {
        "usage_score":       usage_score,
        "interaction_score": interaction_score,
        "Scroll_Rate":       scroll_rate_z,
        "Self_Control":      self_control_z,
        "self_reg_risk":     self_reg_risk,
    }

    for r in ["Boredom", "Entertainment", "Habit", "Procrastination"]:
        row[f"Watch_Reason_{r}"] = 1 if r in watch_reasons else 0

    for c in ["ASMR", "Comedy", "Entertainment", "Gaming",
              "Jokes/Memes", "Life Hacks", "Pranks", "Trends", "Vlogs"]:
        row[f"Video_Category_{c}"] = 1 if c in video_cats else 0

    return row


# ── Profile summary ───────────────────────────────────────────────────────────

def build_profile_summary(row: dict, prob: float | None) -> str:
    reasons = ", ".join(
        r for r in ["Boredom", "Entertainment", "Habit", "Procrastination"]
        if row.get(f"Watch_Reason_{r}", 0) == 1
    ) or "none selected"
    cats = ", ".join(
        c for c in ["ASMR", "Comedy", "Entertainment", "Gaming",
                    "Jokes/Memes", "Life Hacks", "Pranks", "Trends", "Vlogs"]
        if row.get(f"Video_Category_{c}", 0) == 1
    ) or "none selected"

    parts = []
    u = row["usage_score"]
    if u > 0.5:
        parts.append(f"Above-average social media usage (usage score: {u:.2f}).")
    elif u < -0.5:
        parts.append(f"Below-average social media usage (usage score: {u:.2f}).")
    else:
        parts.append(f"Average social media usage (usage score: {u:.2f}).")

    sr = row["self_reg_risk"]
    if sr > 0.5:
        parts.append("Low self-regulation: struggles to control usage and feels unsatisfied.")
    elif sr < -0.5:
        parts.append("Good self-regulation: able to manage usage habits effectively.")

    if row.get("Watch_Reason_Procrastination"):
        parts.append("Uses social media to procrastinate — a key risk indicator.")
    if row.get("Watch_Reason_Habit"):
        parts.append("Usage is habit-driven, suggesting compulsive patterns.")

    parts.append(f"Watch reasons: {reasons}.")
    parts.append(f"Preferred content: {cats}.")
    out = "  ".join(parts)
    if prob is not None:
        out += f"  Model-estimated risk probability: {prob:.2f}."
    return out


# ── LLM advice ────────────────────────────────────────────────────────────────

def generate_advice(profile_summary: str, hits: list[RetrievalHit]) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "_ANTHROPIC_API_KEY not set — set it to enable AI advice._"

    import anthropic

    context = "\n\n---\n\n".join(
        f"[{i}] Source: {h.source} ({h.doc_id})\n{h.text}"
        for i, h in enumerate(hits, 1)
    )
    prompt = f"""You are a mental-health support assistant for youth social media use.

User profile:
{profile_summary}

Retrieved evidence ({len(hits)} excerpts from WHO, SAMHSA, CDC, NIMH, etc.):
{context}

Using ONLY the retrieved excerpts above, write a SHORT response (max 150 words total):
- 1 sentence summarizing the main risk pattern.
- 3 bullet-point recommendations, each ending with a citation (e.g. [1]).

Be concise. Do not speculate beyond the sources."""

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


# ── Page ──────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Social Media Risk Assessment",
    layout="wide",
)

st.title("Social Media Mental Health Risk Assessment")
st.caption("Answer the questions below, then click **Analyze** to get your personalized risk report.")

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Questionnaire")

    mode = st.radio(
        "Input mode",
        ["Fill in the questionnaire", "Use a sample high-risk profile"],
        horizontal=True,
    )

    def synced(label, min_val, max_val, default, step, key):
        """Slider + number_input that stay in sync via session_state callbacks."""
        skey = f"{key}_s"
        nkey = f"{key}_n"
        if skey not in st.session_state:
            st.session_state[skey] = default
        if nkey not in st.session_state:
            st.session_state[nkey] = float(default)

        def _slider_changed():
            st.session_state[nkey] = float(st.session_state[skey])

        def _num_changed():
            v = st.session_state[nkey]
            st.session_state[skey] = int(v) if step < 1 or isinstance(step, int) else v

        col_s, col_n = st.columns([3, 1])
        with col_s:
            st.slider(label, min_value=min_val, max_value=max_val, step=step,
                      key=skey, on_change=_slider_changed)
        with col_n:
            st.number_input(
                label,
                min_value=float(min_val),
                max_value=float(max_val),
                step=float(step),
                key=nkey,
                on_change=_num_changed,
                label_visibility="collapsed",
            )
        return st.session_state[skey]

    sample_row: dict | None = None
    if mode == "Use a sample high-risk profile":
        try:
            df = load_profiles()
            subset = _pick_subset(df)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

        options = []
        for idx, r in subset.iterrows():
            p = r.get("xgb_risk_prob", None)
            label = f"Row {idx}" + (f" (risk={float(p):.2f})" if p is not None else "")
            options.append((label, int(idx)))
        if not options:
            st.error("No sample profiles available.")
            st.stop()

        sel_label = st.selectbox("Sample profiles", [o[0] for o in options])
        sel_idx = dict(options)[sel_label]
        sample_row = df.loc[sel_idx].to_dict()
        st.caption("Using engineered features from `data/behavior_profile_dataset.csv` for this sample.")
        analyze = st.button("Analyze sample", use_container_width=True, type="primary")
    else:
        st.markdown("##### Usage Patterns")
        total_time   = synced("Daily time on social media (minutes)",          10,  300,   150, 5,   "time")
        sessions     = synced("Times you open social media per day",            1,   20,    10, 1,   "sess")
        videos       = synced("Videos watched per day",                         1,   50,    25, 1,   "vids")

        st.markdown("##### Interaction Style")
        scroll_rate  = synced("Scroll rate (scrolls per minute)",               1,  100,    50, 1,   "scrl")
        engagement   = synced("Engagement (likes / comments / shares per day)", 0, 10000, 5000, 100, "engm")

        st.markdown("##### Self-Regulation _(strongest risk signal)_")
        self_control = synced("Self-control (3 = low, 10 = high)",              3,   10,     7, 1,   "sc")
        satisfaction = synced("Satisfaction with usage habits (1 = low, 9 = high)", 1, 9,   5, 1,   "sat")

        st.markdown("##### Why do you use social media? _(select all that apply)_")
        col_a, col_b = st.columns(2)
        with col_a:
            r_habit   = st.checkbox("Habit", key="reason_Habit")
            r_procras = st.checkbox("Procrastination", key="reason_Procrastination")
        with col_b:
            r_boredom   = st.checkbox("Boredom", key="reason_Boredom")
            r_entertain = st.checkbox("Entertainment", key="reason_Entertainment")

        watch_reasons = (
            (["Habit"]           if r_habit    else []) +
            (["Procrastination"] if r_procras  else []) +
            (["Boredom"]         if r_boredom  else []) +
            (["Entertainment"]   if r_entertain else [])
        )

        st.markdown("##### Content type _(select all that apply)_")
        col_c, col_d, col_e = st.columns(3)
        all_cats = ["ASMR", "Comedy", "Entertainment", "Gaming",
                    "Jokes/Memes", "Life Hacks", "Pranks", "Trends", "Vlogs"]
        cat_checks = {}
        for i, cat in enumerate(all_cats):
            with [col_c, col_d, col_e][i % 3]:
                cat_checks[cat] = st.checkbox(cat, key=f"cat_{cat}")
        video_cats = [c for c, v in cat_checks.items() if v]

        analyze = st.button("Analyze", use_container_width=True, type="primary")


with right:
    st.subheader("Results")

    if not analyze:
        st.info("Fill in the questionnaire and click **Analyze** to see your results.")
    else:
        if sample_row is not None:
            model = threshold = cols = stats = None
            if _xgb_artifacts_exist():
                model = joblib.load(XGB_MODEL)
                threshold = float(joblib.load(XGB_THRESHOLD))
                cols = joblib.load(XGB_COLS)
            row = sample_row
        else:
            try:
                model, threshold, cols, stats = load_xgb_bundle()
            except FileNotFoundError as e:
                st.error(str(e))
                st.stop()

            row = build_feature_row(
                total_time, sessions, videos,
                scroll_rate, engagement,
                self_control, satisfaction,
                watch_reasons, video_cats,
                stats=stats,
            )

        prob: float | None
        if model is None:
            st.warning(
                "Trained classifier files not found. Add **xgb_filter.pkl**, **xgb_threshold.pkl**, "
                "and **xgb_feature_cols.pkl** under `method2/` to enable ML risk scoring. "
                "Showing **profile summary, retrieval, and advice** using your inputs only."
            )
            prob = None
        else:
            features_df = pd.DataFrame([row])[cols]
            prob = float(model.predict_proba(features_df)[0, 1])
            is_high_risk = prob >= threshold

            if prob >= 0.5:
                st.error("**High Risk** — Your usage patterns show significant risk indicators.")
            elif is_high_risk:
                st.warning("**Possible Risk** — Some risk indicators detected. See recommendations below.")
            else:
                st.success("**Low Risk** — Your usage patterns appear within a healthy range.")
                st.stop()

        st.markdown("---")
        st.markdown("#### Profile Summary")
        profile_summary = build_profile_summary(row, prob)
        st.write(profile_summary)

        st.markdown("---")
        st.markdown("#### Retrieved Evidence")
        try:
            index = load_index()
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

        q = generate_query(pd.Series(row))
        st.caption(f"Query: _{q if q else '(fallback)'}_")

        hits = index.search(q, top_k=5)
        for i, h in enumerate(hits, 1):
            with st.expander(f"[{i}] {h.source} — {h.doc_id}  (score: {h.score:.4f})"):
                if h.url:
                    st.markdown(f"[{h.url}]({h.url})")
                st.write(h.text[:600] + ("…" if len(h.text) > 600 else ""))

        st.markdown("---")
        st.markdown("#### Personalized Recommendations")
        with st.spinner("Generating advice based on retrieved evidence..."):
            advice = generate_advice(profile_summary, hits)
        st.write(advice)
