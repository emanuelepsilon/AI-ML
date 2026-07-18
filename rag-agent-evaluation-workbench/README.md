# RAG Agent Evaluation Workbench

Flagship applied AI systems project focused on retrieval-augmented generation, agent reliability, citation quality, and evaluation.

The project is designed as a compact but serious workbench for testing whether an AI assistant can answer technical questions from a local knowledge base while staying grounded in retrieved evidence.

## Core Idea

Modern AI applications are not just model calls. They need retrieval, ranking, source attribution, refusal behavior, evaluation sets, and reproducible diagnostics. This project implements those pieces as a local evaluation pipeline.

## System Capabilities

- Local document ingestion and chunking
- TF-IDF retrieval with cosine similarity
- Citation-aware answer assembly
- Query set evaluation
- Retrieval recall checks
- Grounding checks
- Refusal checks for unsupported questions
- HTML report output
- CSV metrics output

## Run

```powershell
python app.py
```

Output:

- `reports/evaluation_report.html`
- `reports/evaluation_metrics.csv`
- `reports/retrieval_trace.csv`

## Architecture

```text
documents/
  -> chunking
  -> retrieval index
  -> query runner
  -> citation-aware answer builder
  -> evaluation harness
  -> HTML and CSV reports
```

## Why It Matters

This project demonstrates practical AI engineering beyond a notebook:

- grounding model behavior in external context
- inspecting retrieval traces
- measuring whether answers cite the right sources
- separating supported and unsupported questions
- producing reproducible evaluation artifacts

## Project Status

Initial local workbench implemented. Next steps include adding LLM provider adapters, reranking, vector database backends, and agent tool execution traces.
