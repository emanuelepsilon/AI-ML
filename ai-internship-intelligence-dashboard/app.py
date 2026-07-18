from __future__ import annotations

import csv
import html
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "opportunities.csv"
REPORTS_DIR = ROOT / "reports"
ASSETS_DIR = ROOT / "assets"
DB_PATH = REPORTS_DIR / "internship_intelligence.db"

PROFILE_SKILLS = {
    "python",
    "tensorflow",
    "tflite",
    "machine learning",
    "llm api",
    "rag",
    "agents",
    "mcp",
    "sql",
    "data analysis",
    "optimization",
    "c",
    "git",
    "edge ai",
    "computer vision",
}

FIELD_WEIGHTS = {
    "LLM Systems": 18,
    "Edge AI": 17,
    "Machine Learning": 15,
    "Computer Vision": 13,
    "Data": 10,
}

STATUS_WEIGHTS = {
    "interview": 14,
    "dream": 12,
    "applied": 9,
    "saved": 5,
    "rejected": -12,
}


def ensure_dirs() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    ASSETS_DIR.mkdir(exist_ok=True)


def load_rows() -> list[dict[str, str]]:
    with DATA_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def build_database(rows: list[dict[str, str]]) -> None:
    with connect() as connection:
        connection.execute("DROP TABLE IF EXISTS opportunities")
        connection.execute(
            """
            CREATE TABLE opportunities (
                company TEXT NOT NULL,
                role TEXT NOT NULL,
                field TEXT NOT NULL,
                location TEXT NOT NULL,
                status TEXT NOT NULL,
                salary_sek INTEGER NOT NULL,
                skills TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO opportunities
            (company, role, field, location, status, salary_sek, skills)
            VALUES (:company, :role, :field, :location, :status, :salary_sek, :skills)
            """,
            rows,
        )


def score_row(row: sqlite3.Row) -> dict[str, object]:
    role_skills = {item.strip().lower() for item in row["skills"].split(",")}
    matched = sorted(PROFILE_SKILLS & role_skills)
    missing = sorted(role_skills - PROFILE_SKILLS)
    match_ratio = len(matched) / max(len(role_skills), 1)
    salary_bonus = min(int(row["salary_sek"]) / 32000, 1.0) * 8 if int(row["salary_sek"]) else 3
    score = (
        match_ratio * 55
        + FIELD_WEIGHTS.get(row["field"], 8)
        + STATUS_WEIGHTS.get(row["status"], 0)
        + salary_bonus
    )
    return {
        "company": row["company"],
        "role": row["role"],
        "field": row["field"],
        "location": row["location"],
        "status": row["status"],
        "salary_sek": int(row["salary_sek"]),
        "matched_skills": ", ".join(matched),
        "skill_match": round(match_ratio, 2),
        "score": round(score, 1),
        "missing_signals": ", ".join(missing[:4]),
    }


def rank_opportunities() -> list[dict[str, object]]:
    with connect() as connection:
        rows = connection.execute("SELECT * FROM opportunities").fetchall()
    return sorted((score_row(row) for row in rows), key=lambda item: item["score"], reverse=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_tables(ranked: list[dict[str, object]]) -> None:
    fields: dict[str, list[float]] = defaultdict(list)
    for row in ranked:
        fields[str(row["field"])].append(float(row["score"]))

    summary = [
        {
            "field": field,
            "opportunities": len(scores),
            "average_score": round(sum(scores) / len(scores), 1),
        }
        for field, scores in sorted(fields.items())
    ]
    write_csv(REPORTS_DIR / "field_summary.csv", summary)


def make_charts(ranked: list[dict[str, object]]) -> None:
    plt.style.use("dark_background")

    field_scores: dict[str, list[float]] = defaultdict(list)
    status_counts = Counter(str(row["status"]) for row in ranked)
    for row in ranked:
        field_scores[str(row["field"])].append(float(row["score"]))

    field_names = sorted(field_scores, key=lambda name: sum(field_scores[name]) / len(field_scores[name]))
    averages = [sum(field_scores[name]) / len(field_scores[name]) for name in field_names]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(field_names, averages, color=["#60a5fa", "#34d399", "#facc15", "#a78bfa", "#fb7185"][: len(field_names)])
    ax.set_xlabel("Average portfolio fit score")
    ax.set_title("AI/ML Opportunity Fit By Field")
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "field_score.png", dpi=180)
    plt.close(fig)

    labels = list(status_counts.keys())
    values = [status_counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(labels, values, color="#38bdf8")
    ax.set_ylabel("Count")
    ax.set_title("Application Pipeline")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "status_pipeline.png", dpi=180)
    plt.close(fig)


def card(row: dict[str, object]) -> str:
    return f"""
    <article class="card">
      <div class="score">{row["score"]}</div>
      <h3>{html.escape(str(row["company"]))}</h3>
      <p class="role">{html.escape(str(row["role"]))}</p>
      <p>{html.escape(str(row["field"]))} / {html.escape(str(row["location"]))} / {html.escape(str(row["status"]))}</p>
      <p><strong>Matched:</strong> {html.escape(str(row["matched_skills"]))}</p>
      <p><strong>Next signals:</strong> {html.escape(str(row["missing_signals"]))}</p>
    </article>
    """


def write_dashboard(ranked: list[dict[str, object]]) -> None:
    top_cards = "\n".join(card(row) for row in ranked[:6])
    best = ranked[0]
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Internship Intelligence Dashboard</title>
  <style>
    body {{ margin: 0; font-family: Inter, Segoe UI, Arial, sans-serif; background: #080b10; color: #e8edf4; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 36px 22px 48px; }}
    h1 {{ font-size: clamp(32px, 5vw, 58px); margin: 0 0 10px; letter-spacing: 0; }}
    h2 {{ margin-top: 34px; }}
    .sub {{ color: #9ca3af; max-width: 790px; line-height: 1.55; }}
    .hero {{ border-bottom: 1px solid #263241; padding-bottom: 26px; }}
    .metric-row {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 24px 0; }}
    .metric {{ background: #111827; border: 1px solid #263241; border-radius: 8px; padding: 18px; }}
    .metric b {{ display: block; font-size: 28px; margin-bottom: 4px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    .card {{ position: relative; background: #101721; border: 1px solid #263241; border-radius: 8px; padding: 18px; min-height: 190px; }}
    .score {{ position: absolute; top: 14px; right: 14px; background: #22c55e; color: #06100b; border-radius: 999px; padding: 6px 10px; font-weight: 800; }}
    .role {{ color: #93c5fd; font-weight: 700; }}
    .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    img {{ width: 100%; border-radius: 8px; border: 1px solid #263241; background: #0f1720; }}
    @media (max-width: 800px) {{ .metric-row, .grid, .charts {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>AI Internship Intelligence Dashboard</h1>
      <p class="sub">A SQLite-backed ranking system that connects an AI/ML technical profile to internship opportunities using skill overlap, role relevance, application stage, and explainable scoring.</p>
      <div class="metric-row">
        <div class="metric"><b>{len(ranked)}</b>tracked roles</div>
        <div class="metric"><b>{best["company"]}</b>top match</div>
        <div class="metric"><b>{best["score"]}</b>best fit score</div>
      </div>
    </section>
    <h2>Ranked Opportunities</h2>
    <section class="grid">{top_cards}</section>
    <h2>Analytics</h2>
    <section class="charts">
      <img src="../assets/field_score.png" alt="Average score by field">
      <img src="../assets/status_pipeline.png" alt="Application status pipeline">
    </section>
  </main>
</body>
</html>
"""
    (REPORTS_DIR / "dashboard.html").write_text(html_text, encoding="utf-8")


def print_report(ranked: list[dict[str, object]]) -> None:
    print("AI Internship Intelligence Dashboard")
    print("------------------------------------")
    print(f"Loaded opportunities: {len(ranked)}")
    print("Top ranked roles:")
    for index, row in enumerate(ranked[:5], start=1):
        print(f"{index}. {row['company']:<13} {row['score']:>5}  {row['role']}  [{row['field']}]")
    print()
    print("Generated:")
    print(f"- {REPORTS_DIR / 'dashboard.html'}")
    print(f"- {REPORTS_DIR / 'top_ranked_opportunities.csv'}")
    print(f"- {DB_PATH}")


def main() -> None:
    ensure_dirs()
    rows = load_rows()
    build_database(rows)
    ranked = rank_opportunities()
    write_csv(REPORTS_DIR / "top_ranked_opportunities.csv", ranked)
    write_summary_tables(ranked)
    make_charts(ranked)
    write_dashboard(ranked)
    print_report(ranked)


if __name__ == "__main__":
    main()
