# AI-ML

Applied artificial intelligence and machine learning work covering retrieval-augmented generation, agent systems, model evaluation, deep learning, computer vision and edge AI.

## Flagship Project

### [RAG Agent Evaluation Workbench](rag-agent-evaluation-workbench)

A local evaluation workbench for retrieval-augmented generation systems. It tests whether an assistant can retrieve the right evidence, cite the correct sources, refuse unsupported questions, and expose retrieval traces for debugging.

```text
documents -> chunking -> retrieval -> citation-aware answer builder -> evaluation -> reports
```

The project includes local document ingestion, TF-IDF retrieval, cosine similarity ranking, citation-aware answers, refusal checks, retrieval traces, CSV metrics and an HTML evaluation report.

Current evaluation result:

```text
5/5 evaluation cases passed
```

## Project Archive

- `rag-agent-evaluation-workbench` - flagship RAG, citation grounding and agent evaluation workbench.
- `ai-internship-intelligence-dashboard` - AI internship ranking dashboard with SQLite analytics, explainable skill matching, charts, and an HTML report.
- `EmanuelsLLM` - small educational language model implementation.
- `ai-engineering-projects` - structured AI engineering practice, including an LLM API chat project.
- `gemini-terminal-agent` - terminal-based Gemini API chat agent with search-grounding work.
- `huggingface-llamaindex-langgraph-learning` - LlamaIndex, LangGraph, RAG, MCP, and workflow practice scripts.
- `mcp-practice-server` - MCP practice server with tools, resources, prompts, and a terminal client.
- `tensorflow-practice` - TensorFlow and scikit-learn practice scripts.
- `edge-ml-tflite-practice` - TFLite and TabPFN model testing practice.
- `edge-optimization-project` - edge AI optimization project with training, conversion, and benchmark scripts.

## Notes

Secrets, local environments, databases, caches, and personal documents are ignored. Project folders are learning and portfolio material gathered from local work and GitHub repositories.
