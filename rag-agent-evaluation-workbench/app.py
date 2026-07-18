from __future__ import annotations

import csv
import html
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "documents"
REPORTS_DIR = ROOT / "reports"

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-]+")
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "into",
    "when",
    "what",
    "why",
    "how",
    "does",
    "can",
    "must",
    "should",
    "while",
    "than",
    "then",
    "they",
    "their",
    "over",
    "under",
}

SENSITIVE_DECISION_TERMS = {
    "approve",
    "approval",
    "loan",
    "credit",
    "customer",
    "medical",
    "diagnosis",
    "legal",
}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class QueryCase:
    query: str
    expected_sources: tuple[str, ...]
    should_answer: bool


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS]


def chunk_document(path: Path, max_words: int = 95) -> list[Chunk]:
    words = path.read_text(encoding="utf-8").split()
    chunks: list[Chunk] = []
    for start in range(0, len(words), max_words):
        text = " ".join(words[start : start + max_words])
        if text.strip():
            chunks.append(
                Chunk(
                    chunk_id=f"{path.stem}-{len(chunks) + 1:02d}",
                    source=path.name,
                    text=text,
                    tokens=tuple(tokenize(text)),
                )
            )
    return chunks


def load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        chunks.extend(chunk_document(path))
    return chunks


def idf_weights(chunks: list[Chunk]) -> dict[str, float]:
    document_frequency: Counter[str] = Counter()
    for chunk in chunks:
        document_frequency.update(set(chunk.tokens))
    n_docs = len(chunks)
    return {
        token: math.log((1 + n_docs) / (1 + frequency)) + 1.0
        for token, frequency in document_frequency.items()
    }


def vectorize(tokens: list[str] | tuple[str, ...], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokens)
    length = max(sum(counts.values()), 1)
    return {token: (count / length) * idf.get(token, 0.0) for token, count in counts.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(value * b.get(token, 0.0) for token, value in a.items())
    a_norm = math.sqrt(sum(value * value for value in a.values()))
    b_norm = math.sqrt(sum(value * value for value in b.values()))
    if not a_norm or not b_norm:
        return 0.0
    return dot / (a_norm * b_norm)


def retrieve(query: str, chunks: list[Chunk], idf: dict[str, float], top_k: int = 4) -> list[tuple[Chunk, float]]:
    query_vec = vectorize(tokenize(query), idf)
    ranked = [
        (chunk, cosine(query_vec, vectorize(chunk.tokens, idf)))
        for chunk in chunks
    ]
    return sorted(ranked, key=lambda item: item[1], reverse=True)[:top_k]


def answer(query: str, retrieved: list[tuple[Chunk, float]], threshold: float = 0.10) -> str:
    query_tokens = set(tokenize(query))
    has_sensitive_decision = bool(query_tokens & SENSITIVE_DECISION_TERMS)
    if has_sensitive_decision:
        return "The local knowledge base does not contain enough evidence to answer this reliably."

    useful = [(chunk, score) for chunk, score in retrieved if score >= threshold]
    if not useful:
        return "The local knowledge base does not contain enough evidence to answer this reliably."

    evidence = " ".join(chunk.text for chunk, _ in useful[:2])
    first_sentence = re.split(r"(?<=[.!?])\s+", evidence.strip())[0]
    sources = ", ".join(f"[{chunk.source}]" for chunk, _ in useful[:2])
    return f"{first_sentence} Sources: {sources}"


def query_cases() -> list[QueryCase]:
    return [
        QueryCase(
            "How should a RAG system decide whether to answer or refuse?",
            ("rag_reliability.md",),
            True,
        ),
        QueryCase(
            "What metrics are useful for evaluating retrieval quality?",
            ("evaluation_metrics.md",),
            True,
        ),
        QueryCase(
            "How can an agent expose tool calls without hiding important decisions?",
            ("agent_observability.md",),
            True,
        ),
        QueryCase(
            "What matters when deploying neural networks to constrained edge hardware?",
            ("edge_ai_notes.md",),
            True,
        ),
        QueryCase(
            "Which customer should receive a loan approval today?",
            (),
            False,
        ),
    ]


def evaluate(chunks: list[Chunk]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    idf = idf_weights(chunks)
    metric_rows: list[dict[str, str]] = []
    trace_rows: list[dict[str, str]] = []

    for case in query_cases():
        retrieved = retrieve(case.query, chunks, idf)
        result = answer(case.query, retrieved)
        retrieved_sources = tuple(chunk.source for chunk, _ in retrieved)
        source_hits = sum(1 for source in case.expected_sources if source in retrieved_sources)
        recall = source_hits / max(len(case.expected_sources), 1) if case.expected_sources else 1.0
        refused = "does not contain enough evidence" in result
        correct_refusal = refused == (not case.should_answer)
        grounded = all(source in result for source in case.expected_sources) if case.should_answer else refused

        metric_rows.append(
            {
                "query": case.query,
                "expected_sources": "; ".join(case.expected_sources),
                "top_source": retrieved_sources[0] if retrieved_sources else "",
                "retrieval_recall": f"{recall:.2f}",
                "grounded": str(grounded),
                "correct_refusal": str(correct_refusal),
                "answer": result,
            }
        )

        for rank, (chunk, score) in enumerate(retrieved, start=1):
            trace_rows.append(
                {
                    "query": case.query,
                    "rank": str(rank),
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "score": f"{score:.3f}",
                    "preview": chunk.text[:160],
                }
            )

    return metric_rows, trace_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_html(metrics: list[dict[str, str]], traces: list[dict[str, str]]) -> None:
    passed = sum(1 for row in metrics if row["grounded"] == "True" and row["correct_refusal"] == "True")
    avg_recall = sum(float(row["retrieval_recall"]) for row in metrics) / len(metrics)

    rows_html = "\n".join(
        f"""
        <tr>
          <td>{html.escape(row["query"])}</td>
          <td>{html.escape(row["top_source"])}</td>
          <td>{row["retrieval_recall"]}</td>
          <td>{row["grounded"]}</td>
          <td>{row["correct_refusal"]}</td>
        </tr>
        """
        for row in metrics
    )

    trace_html = "\n".join(
        f"""
        <tr>
          <td>{html.escape(row["query"])}</td>
          <td>{row["rank"]}</td>
          <td>{html.escape(row["source"])}</td>
          <td>{row["score"]}</td>
        </tr>
        """
        for row in traces[:16]
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG Agent Evaluation Workbench</title>
  <style>
    body {{ margin: 0; font-family: Inter, Segoe UI, Arial, sans-serif; background: #080b10; color: #e8edf4; }}
    main {{ max-width: 1080px; margin: 0 auto; padding: 38px 22px 54px; }}
    h1 {{ margin: 0 0 8px; font-size: 42px; letter-spacing: 0; }}
    h2 {{ margin-top: 34px; }}
    p {{ color: #aab4c2; line-height: 1.55; }}
    .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 24px 0; }}
    .metric {{ background: #111827; border: 1px solid #2c394a; border-radius: 8px; padding: 18px; }}
    .metric b {{ display: block; color: #34d399; font-size: 30px; }}
    table {{ width: 100%; border-collapse: collapse; background: #101722; border: 1px solid #2c394a; }}
    th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid #263241; vertical-align: top; }}
    th {{ color: #93c5fd; }}
    @media (max-width: 800px) {{ .metrics {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <h1>RAG Agent Evaluation Workbench</h1>
    <p>Local evaluation pipeline for retrieval quality, citation grounding, refusal behavior, and trace inspection.</p>
    <section class="metrics">
      <div class="metric"><b>{passed}/{len(metrics)}</b>cases passed</div>
      <div class="metric"><b>{avg_recall:.2f}</b>average recall</div>
      <div class="metric"><b>{len(traces)}</b>retrieval traces</div>
    </section>
    <h2>Evaluation Summary</h2>
    <table>
      <thead><tr><th>Query</th><th>Top Source</th><th>Recall</th><th>Grounded</th><th>Refusal</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    <h2>Retrieval Trace Sample</h2>
    <table>
      <thead><tr><th>Query</th><th>Rank</th><th>Source</th><th>Score</th></tr></thead>
      <tbody>{trace_html}</tbody>
    </table>
  </main>
</body>
</html>
"""
    (REPORTS_DIR / "evaluation_report.html").write_text(page, encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    chunks = load_chunks()
    metrics, traces = evaluate(chunks)
    write_csv(REPORTS_DIR / "evaluation_metrics.csv", metrics)
    write_csv(REPORTS_DIR / "retrieval_trace.csv", traces)
    write_html(metrics, traces)

    passed = sum(1 for row in metrics if row["grounded"] == "True" and row["correct_refusal"] == "True")
    print("RAG Agent Evaluation Workbench")
    print("-----------------------------")
    print(f"documents: {len(list(DOCS_DIR.glob('*.md')))}")
    print(f"chunks: {len(chunks)}")
    print(f"evaluation cases: {len(metrics)}")
    print(f"passed: {passed}/{len(metrics)}")
    print("output:")
    print(f"- {REPORTS_DIR / 'evaluation_report.html'}")
    print(f"- {REPORTS_DIR / 'evaluation_metrics.csv'}")
    print(f"- {REPORTS_DIR / 'retrieval_trace.csv'}")


if __name__ == "__main__":
    main()
