#!/usr/bin/env python3
"""
Run using the command:
python3 method3/terminalDemo.py
"""
import sys
from pathlib import Path
from textwrap import fill
import os

import pandas as pd
import anthropic
import joblib

ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from retriever import ChunkIndex, generate_query

WIDTH = 88
MAX_SESSIONS_PER_DAY = 60

_VIDEO_SLUGS = (
    ("asmr", "ASMR"),
    ("comedy", "Comedy"),
    ("entertainment", "Entertainment"),
    ("gaming", "Gaming"),
    ("jokes/memes", "Jokes/Memes"),
    ("life hacks", "Life Hacks"),
    ("pranks", "Pranks"),
    ("trends", "Trends"),
    ("vlogs", "Vlogs"),
)

SAMPLE_IDS = [572, 710, 715, 659, 689]


def _pick_five_high_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Select 5 high risk."""
    model = joblib.load(ROOT / "xgb_filter.pkl")
    threshold = joblib.load(ROOT / "xgb_threshold.pkl")
    cols = joblib.load(ROOT / "xgb_feature_cols.pkl")

    probs = model.predict_proba(df[cols])[:, 1]
    mask = probs >= threshold

    result = df[mask].copy()
    result["xgb_risk_prob"] = probs[mask]

    return result.sort_values("xgb_risk_prob", ascending=False).head(5)


def _pick_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Use trained XGBoost filters if present; otherwise fixed CSV rows (SAMPLE_IDS)."""
    pkls = (
        ROOT / "xgb_filter.pkl",
        ROOT / "xgb_threshold.pkl",
        ROOT / "xgb_feature_cols.pkl",
    )
    if all(p.is_file() for p in pkls):
        return _pick_five_high_risk(df)
    print(
        "Note: xgb_filter.pkl / xgb_threshold.pkl / xgb_feature_cols.pkl not found — "
        "using SAMPLE_IDS rows instead. Train/export the model to enable XGBoost filtering.",
        file=sys.stderr,
    )
    return df.loc[SAMPLE_IDS]


def _watch_reasons(row: pd.Series) -> str:
    """Watch reason data handling."""
    parts = []
    for name in (
        "Watch_Reason_Boredom",
        "Watch_Reason_Entertainment",
        "Watch_Reason_Habit",
        "Watch_Reason_Procrastination",
    ):
        if row.get(name, 0) == 1:
            parts.append(name.replace("Watch_Reason_", "").lower())
    return ", ".join(parts) if parts else "—"


def _video_cats(row: pd.Series) -> str:
    """Video category data handling."""
    parts = []
    for c in row.index:
        if isinstance(c, str) and c.startswith("Video_Category_") and row[c] == 1:
            parts.append(c.replace("Video_Category_", ""))
    return ", ".join(parts) if parts else "—"


def _float_input(prompt, min_v=0, max_v=1):
    """Float user input error handling."""
    while True:
        try:
            v = float(input(prompt))
            if v < min_v or v > max_v:
                print(f"Enter value between {min_v} and {max_v}")
                continue
            return v
        except ValueError:
            print("Invalid number.")


def _yes_no(prompt):
    """Yes/no user input error handling."""
    while True:
        v = input(prompt).strip().lower()
        if v in ("y", "yes"):
            return 1
        if v in ("n", "no"):
            return 0
        print("Enter y or n.")


def _create_user_row() -> pd.Series:
    """Create new user questionaire."""
    print("\nCreate New User Profile\n")

    hours = _float_input(
        "How many hours a day do you spend on social media? (0–12) ", 0, 12
    )
    sessions = _float_input(
        f"How many times a day do you open a social media app? (1–{MAX_SESSIONS_PER_DAY}) ",
        1,
        float(MAX_SESSIONS_PER_DAY),
    )
    scroll_quick = _float_input(
        "How often do you scroll past videos without watching? (0=rarely, 10=constantly) ",
        0,
        10,
    )
    engage_quick = _float_input(
        "How often do you like, comment, or share content? (0=never, 10=always) ",
        0,
        10,
    )
    longer = _float_input(
        "How often do you use social media longer than you intended? (0=never, 10=always) ",
        0,
        10,
    )
    satisfied = _float_input(
        "How satisfied are you with your current social media use? "
        "(0=very unsatisfied, 10=very satisfied) ",
        0,
        10,
    )

    print("What type of content do you mostly watch? (enter a number, or n/a)")
    for i, (_slug, label) in enumerate(_VIDEO_SLUGS, start=1):
        print(f"  [{i}] {label}")
    chosen_slug: str | None = None
    while True:
        ans = input(f"Choice (1–{len(_VIDEO_SLUGS)} or n/a): ").strip().lower()
        if ans in ("n/a", "na", "none", "-", ""):
            chosen_slug = None
            break
        if ans.isdigit():
            n = int(ans)
            if 1 <= n <= len(_VIDEO_SLUGS):
                chosen_slug = _VIDEO_SLUGS[n - 1][0]
                break
        print(f"Invalid. Enter 1–{len(_VIDEO_SLUGS)} or n/a.")

    print("Why do you usually open social media? (y/n each)")
    boredom = _yes_no("  Boredom? ")
    entertainment = _yes_no("  Entertainment? ")
    habit = _yes_no("  Habit? ")
    procrastination = _yes_no("  Procrastination? ")

    usage_score = (hours / 12 + sessions / MAX_SESSIONS_PER_DAY) / 2
    interaction_score = (scroll_quick + engage_quick) / 20
    self_reg_risk = (longer - satisfied) / 10

    row = {
        "usage_score": usage_score,
        "interaction_score": interaction_score,
        "self_reg_risk": self_reg_risk,
        "Watch_Reason_Boredom": boredom,
        "Watch_Reason_Entertainment": entertainment,
        "Watch_Reason_Habit": habit,
        "Watch_Reason_Procrastination": procrastination,
    }

    for slug, suffix in _VIDEO_SLUGS:
        row[f"Video_Category_{suffix}"] = int(
            chosen_slug is not None and slug == chosen_slug
        )

    return pd.Series(row)


def _append_to_csv(row: pd.Series, path: Path):
    """Append new user to csv."""
    df = pd.read_csv(path)
    new = pd.DataFrame([row])

    for col in df.columns:
        if col not in new.columns:
            new[col] = 0

    new = new[df.columns]
    df = pd.concat([df, new], ignore_index=True)
    df.to_csv(path, index=False)


def _generate_advice(summary, hits):
    """Generate advice from LLM."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return "No API key set."

    context = "\n\n".join([h.text for h in hits])

    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system="You are a assistant that helps users with their social media use. Provided are some documents that give advice to the user based on their needs. You will provide actionable advice that is no longer than 150 words.",
        messages=[
            {"role": "user", "content": (
                f"{summary}\n\n{context}\n\n"
                "Based on the behavioral profile and retrieved guidance above, provide 3–5 practical suggestions. "
                "Begin with a brief disclaimer that this is not medical advice and is for informational purposes only. "
                "Keep recommendations general and actionable, not clinical."
            )}
        ]
    )
    return msg.content[0].text


def _show_profile_menu():
    """Menu display."""
    print("\n==============================")
    print(" Social Media Risk Demo")
    print("==============================")
    print("[1] Sample high-risk profiles")
    print("[2] Create new user")
    print("[q] Quit\n")


def _show_results(row, index):
    """Generate result from query."""
    q = generate_query(row)
    hits = index.search(q, top_k=5)

    print("\nQUERY:", q)
    print("\nTOP RESULTS:\n")

    for i, h in enumerate(hits, 1):
        print(f"{i}. {h.score:.3f} | {h.source}")
        print(fill(h.text[:400]))
        print()

    summary = "User behavior profile analysis"
    advice = _generate_advice(summary, hits)

    print("\nADVICE:\n")
    print(fill(advice))


def main():
    """
    Main function for terminal input.
    """
    csv_path = ROOT / "data" / "behavior_profile_dataset.csv"

    df = pd.read_csv(csv_path)

    index = ChunkIndex(root=ROOT)

    subset = _pick_subset(df)

    while True:
        _show_profile_menu()
        choice = input("Select: ").strip()

        if choice in ("q", "quit"):
            return

        # Sample users
        if choice == "1":
            rows = list(subset.iterrows())

            print("\nSelect user:\n")
            for i, (idx, row) in enumerate(rows, 1):
                print(f"[{i}] User {idx}")

            sel = input(f"\nPick 1–{len(rows)} or b: ").strip().lower()
            if sel == "b":
                continue

            idx, row = rows[int(sel) - 1]

            print(f"\n--- USER {idx} ---")
            _show_results(row, index)
            input("\nPress Enter...")

        # New users
        elif choice == "2":
            row = _create_user_row()

            print("\n--- NEW USER ---")
            _show_results(row, index)

            _append_to_csv(row, csv_path)

            input("\nSaved. Press Enter...")

        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()