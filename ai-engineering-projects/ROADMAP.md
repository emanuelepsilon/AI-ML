# AI Engineering Internship Roadmap

You already have the harder-to-fake foundation: math, programming, ML basics and scientific thinking. The missing layer is applied LLM engineering: turning models into useful, testable software systems.

The goal is not to pretend you are a senior engineer. The goal is to become the student who can say: "I built this, I know where it fails, and I can improve it."

## Skill Map

| Skill | What it means | Why companies ask for it | What you should build | Internship connection |
| --- | --- | --- | --- | --- |
| Python for AI systems | Writing clean Python programs that call models, process data, expose APIs and run tests. | Most AI prototypes and internal tools are Python first. | CLI tools, FastAPI services, data loaders, tests and notebooks. | Shows you can move from coursework scripts to maintainable software. |
| LLM API usage | Calling hosted models through a structured API, passing instructions and reading responses. | Companies use APIs before training their own models. | Chat apps, summarizers, classifiers and structured extraction tools. | Shows you can integrate models into products. |
| RAG | Retrieving external context before asking an LLM to answer. | It reduces hallucinations and lets models use private company knowledge. | Document assistants over PDFs, policies, tickets or reports. | One of the most common applied AI internship tasks. |
| Embeddings | Converting text into vectors that preserve semantic meaning. | Retrieval, clustering, recommendations and duplicate detection depend on embeddings. | Semantic search, similarity search and document clustering. | Shows you understand the bridge between ML representation and product search. |
| Chunking | Splitting long documents into useful pieces before embedding or retrieval. | Bad chunking creates bad RAG even with a good model. | Chunkers with overlap, metadata and page references. | Lets you discuss real tradeoffs in RAG interviews. |
| Vector databases | Storing embeddings and searching by semantic similarity. | Companies need fast retrieval over many documents. | Search services using Chroma, FAISS, Qdrant or Pinecone. | Shows practical data infrastructure knowledge. |
| Prompt engineering | Designing model instructions, examples and output constraints. | Model behavior is strongly affected by inputs. | Role prompts, structured outputs, fallback behavior and prompt tests. | Shows you can control model behavior systematically. |
| AI agents | LLM systems that decide steps and use tools to complete tasks. | Teams are experimenting with agents for research, operations and workflow automation. | Agents that call calculators, APIs, file search and business tools. | Shows you understand tool use beyond simple chat. |
| Tool calling | Letting the model request structured tool executions. | Products need models to interact with real systems, not just produce text. | Function calling for weather, finance, database lookup or document search. | Shows you can safely connect models to external actions. |
| Workflow orchestration | Organizing multi-step AI processes with clear stages and state. | Production AI is a pipeline, not one prompt. | Load, clean, embed, retrieve, answer, evaluate and log. | Shows you can reason about repeatable systems. |
| Evaluations | Testing model outputs for quality, correctness and safety. | AI systems are probabilistic and need measurement. | Golden datasets, regression tests, rubric scoring and failure reports. | Makes you stand out because many students only demo happy paths. |
| Data pipelines | Moving raw data through cleaning, transformation, storage and serving. | AI quality depends heavily on data preparation. | Ingestion scripts, parsers, validators and scheduled updates. | Connects your engineering physics discipline to applied AI work. |
| MCP | A protocol for connecting LLM clients to tools, files, databases and services. | It is becoming a common way to expose context and actions to AI systems. | A small MCP server exposing local project notes or search tools. | Shows awareness of modern AI tooling architecture. |
| Deployment basics | Running an app outside your laptop with environment variables and logs. | Interns often ship prototypes for teammates to test. | FastAPI apps, Docker files and simple cloud deployment. | Shows product-readiness, not just local demos. |
| Logging and debugging | Recording inputs, outputs, errors, latency and model settings. | LLM bugs are often data, prompt, retrieval or config bugs. | Request logs, trace IDs, latency metrics and error handling. | Lets you debug like an engineer, not guess. |
| Security and privacy basics | Protecting API keys, sensitive data and user inputs. | AI apps often touch private company data. | `.env` config, redaction, no secrets in Git and safe tool permissions. | Critical for internships in fintech, claims, healthcare or internal search. |
| Documentation and GitHub presentation | Explaining the problem, architecture, setup, screenshots, limitations and next steps. | Hiring teams skim projects quickly. | Strong READMEs, diagrams, examples and interview notes. | Turns projects into evidence recruiters can understand. |

## Project Roadmap

Build these in order. Each project should have a README, tests, example data and an "Interview Notes" section.

## Project 1: LLM API Chat CLI

Problem: You need a small professional Python chat app that calls an LLM API and handles configuration, errors and conversation history.

Why it matters: This is the smallest useful unit of applied LLM engineering. Nearly every bigger system starts here.

Technical stack: Python, OpenAI Python SDK, Responses API, dotenv, pytest.

Features to build:

- CLI chat loop.
- Environment-based API key and model configuration.
- Developer/system instruction.
- Conversation memory in-process.
- Clear error messages for missing keys, bad models and failed requests.
- Tests that mock the LLM API instead of spending money.

Folder structure:

```text
01_llm_api_chat/
  README.md
  pyproject.toml
  .env.example
  src/llm_chat/
    __init__.py
    cli.py
    config.py
    openai_chat.py
  tests/
    test_openai_chat.py
```

Implementation plan:

1. Create project metadata and dependencies.
2. Load config from `.env` and environment variables.
3. Wrap the LLM API call in a small client class.
4. Add a CLI loop for user input.
5. Add tests using a fake client.
6. Write a README with setup, run commands, tradeoffs and interview notes.

README should include:

- What the app does.
- How to install and run it.
- Required environment variables.
- Example conversation.
- How the API call works.
- Common errors.
- What you learned.

Interview explanation:

- "I built a minimal LLM app around a hosted model API. I separated configuration, API interaction and CLI logic, then tested the chat layer with a fake client so tests do not require network access or API spend."

Extensions:

- Add streaming.
- Add conversation saving to JSON.
- Add token and cost tracking.
- Add a FastAPI endpoint.
- Add structured JSON output mode.

Portfolio bullets:

- Built a Python LLM chat application using environment-based configuration and a hosted model API.
- Added conversation history, error handling and test coverage with mocked API calls.
- Documented setup, limitations and common production concerns such as latency, cost and secrets.

## Project 2: Document RAG Assistant

Problem: A user has PDFs or text files and wants grounded answers with citations.

Why it matters: RAG is one of the most requested internship skills because companies need AI over private documents.

Technical stack: Python, pypdf, sentence-transformers or OpenAI embeddings, Chroma or FAISS, OpenAI Responses API, pytest.

Features to build:

- Load PDF and text documents.
- Clean text and preserve source metadata.
- Chunk documents with overlap.
- Embed chunks.
- Store vectors locally.
- Retrieve top-k chunks for a query.
- Generate an answer using only retrieved context.
- Return citations with file and page or chunk IDs.

Folder structure:

```text
02_document_rag_assistant/
  README.md
  data/raw/
  data/index/
  src/rag_assistant/
    ingest.py
    chunking.py
    embeddings.py
    vector_store.py
    answer.py
    cli.py
  tests/
```

Implementation plan:

1. Build a document loader.
2. Build a chunker with chunk size and overlap settings.
3. Generate embeddings for each chunk.
4. Store vectors and metadata.
5. Retrieve relevant chunks by semantic similarity.
6. Generate grounded answers with citations.
7. Add tests for chunking and retrieval.

README should include:

- A diagram of ingest to retrieve to answer.
- Example documents and questions.
- Chunking choices and tradeoffs.
- Known failure modes.

Interview explanation:

- "RAG improves answers by retrieving relevant context first. I handled ingestion, chunking, embedding, vector search and answer generation as separate pipeline stages so each stage can be debugged."

Extensions:

- Hybrid keyword plus vector search.
- Reranking.
- Web UI.
- Query rewriting.
- Evaluation dataset with expected citations.

Portfolio bullets:

- Built a RAG assistant over local documents with chunking, embeddings, semantic retrieval and citation-aware answers.
- Designed a repeatable ingestion pipeline and separated retrieval from answer generation for easier debugging.
- Evaluated answer quality with curated test questions and retrieval checks.

## Project 3: Vector Search Lab

Problem: A team wants to search support tickets, research notes or product docs by meaning instead of exact keywords.

Why it matters: Vector search is the infrastructure layer behind RAG, recommendations and semantic clustering.

Technical stack: Python, FastAPI, Chroma or Qdrant, embeddings, pytest, optional Docker.

Features to build:

- Dataset ingestion from CSV or JSONL.
- Embedding generation.
- Vector database upsert and search.
- Metadata filters such as source, date or category.
- REST endpoint for semantic search.
- Benchmark script for latency and retrieval quality.

Folder structure:

```text
03_vector_search_lab/
  README.md
  data/
  src/vector_search/
    ingest.py
    db.py
    api.py
    benchmark.py
  tests/
```

Implementation plan:

1. Prepare a small searchable dataset.
2. Embed records and store vectors with metadata.
3. Implement semantic search.
4. Add metadata filtering.
5. Expose search through FastAPI.
6. Measure latency and inspect bad matches.

README should include:

- Dataset description.
- Vector database choice.
- Example search queries.
- Latency results.
- Limitations.

Interview explanation:

- "A vector database stores embeddings so semantically similar items can be retrieved quickly. I used metadata filters and benchmarks to show search behavior beyond a demo."

Extensions:

- Add reranking.
- Compare FAISS, Chroma and Qdrant.
- Add approximate nearest neighbor settings.
- Add a UI for exploring results.

Portfolio bullets:

- Implemented semantic search over structured records using embeddings and a vector database.
- Added metadata filtering and latency measurement to make the search system closer to production use.
- Compared retrieval behavior across query types and documented failure cases.

## Project 4: Tool-Using AI Agent

Problem: A user wants an assistant that can answer by calling tools such as calculator, file search, market lookup or calendar-like functions.

Why it matters: Agents are LLM systems that move from text generation to action. Companies want interns who understand both the promise and the danger.

Technical stack: Python, OpenAI Responses API tool calling or an agent framework, pydantic, pytest, optional FastAPI.

Features to build:

- Define safe tools with schemas.
- Let the model request a tool call.
- Execute the tool in Python.
- Feed the tool result back to the model.
- Log every step.
- Add permission checks for risky tools.

Folder structure:

```text
04_tool_agent/
  README.md
  src/tool_agent/
    agent.py
    tools.py
    schemas.py
    cli.py
    logging_utils.py
  tests/
```

Implementation plan:

1. Build deterministic Python tools.
2. Describe tools with JSON schemas.
3. Implement the model to tool to model loop.
4. Add a trace log.
5. Add tests for tool behavior and agent routing.
6. Document tool safety constraints.

README should include:

- Tool list.
- Example trace.
- Safety rules.
- Failure modes.

Interview explanation:

- "An agent is not magic. It is a loop where the model chooses actions, the application executes approved tools, and the model uses results to continue. The application owns safety and permissions."

Extensions:

- Add web search with allowlisted domains.
- Add a simple planning step.
- Add human approval for sensitive actions.
- Add MCP in the next project.

Portfolio bullets:

- Built a tool-using AI agent with structured tool schemas, execution logs and permission boundaries.
- Demonstrated how LLM reasoning connects to deterministic Python functions.
- Tested tool behavior independently from model behavior.

## Project 5: Evaluation Harness

Problem: You need to know whether changes to prompts, retrieval or models make answers better or worse.

Why it matters: Evaluation is where AI engineering becomes engineering. It is a major differentiator in interviews.

Technical stack: Python, pytest, pandas, JSONL datasets, optional LLM-as-judge, matplotlib.

Features to build:

- Golden question dataset.
- Expected answer facts.
- Retrieval evaluation such as hit rate and mean reciprocal rank.
- Answer checks using keyword, rubric or model-judge scoring.
- Regression report comparing runs.
- Failure analysis output.

Folder structure:

```text
05_eval_harness/
  README.md
  datasets/
  src/evals/
    run_eval.py
    metrics.py
    graders.py
    report.py
  reports/
  tests/
```

Implementation plan:

1. Create a small dataset of questions and expected facts.
2. Run the RAG system over all questions.
3. Score retrieval separately from answer quality.
4. Save results to JSONL or CSV.
5. Generate a summary report.
6. Add regression thresholds.

README should include:

- Dataset format.
- Metrics.
- Example report.
- How to interpret failures.

Interview explanation:

- "I evaluated retrieval and answer generation separately. That makes debugging clearer: a bad answer may come from bad retrieval, bad prompting, missing source data or model limitations."

Extensions:

- Add CI checks.
- Add model comparison.
- Add latency and cost tracking.
- Add human review labels.

Portfolio bullets:

- Built an evaluation harness for RAG outputs with retrieval metrics, answer scoring and regression reports.
- Separated retrieval failures from generation failures to make system improvement measurable.
- Documented failure examples and concrete improvement actions.

## Project 6: Applied AI Prototype

Problem: Build a realistic business tool, such as fintech research assistant, claims processing assistant, market-data memo generator, document analysis tool or internal knowledge search.

Recommended version: Fintech document analyst for market notes and company filings.

Why it matters: Final portfolio projects should look like work an AI internship team might actually assign.

Technical stack: Python, FastAPI, Streamlit or simple web UI, vector database, RAG, tool calling, evaluation harness, Docker, logging.

Features to build:

- Upload or ingest business documents.
- Extract structured facts.
- Ask questions over the document set.
- Retrieve sources and cite them.
- Generate short analyst-style summaries.
- Run evaluations on known questions.
- Log latency, model and retrieval metadata.
- Include privacy notes and limitations.

Folder structure:

```text
06_fintech_document_analyst/
  README.md
  app/
  data/
  src/fintech_analyst/
    ingest.py
    rag.py
    extraction.py
    tools.py
    api.py
    evals.py
  tests/
  docker/
```

Implementation plan:

1. Pick a narrow business workflow.
2. Ingest a small document corpus.
3. Build RAG search and cited answers.
4. Add one or two deterministic tools.
5. Add structured extraction.
6. Add evaluations and logs.
7. Package with a clear README and demo screenshots.

README should include:

- Problem statement.
- User workflow.
- Architecture diagram.
- Setup and demo commands.
- Evaluation results.
- Security and privacy notes.
- Limitations and future work.

Interview explanation:

- "I built an applied AI prototype around a realistic workflow. The system combines document ingestion, vector retrieval, LLM reasoning, structured extraction, evaluation and logging. I can explain how each component affects answer quality, latency and risk."

Extensions:

- Add role-based access.
- Add document redaction.
- Add batch processing.
- Add deployment to a small cloud service.
- Add MCP server exposing search and extraction tools.

Portfolio bullets:

- Built an applied AI prototype for business document analysis using RAG, tool calling, structured extraction and evaluations.
- Added source citations, logging and privacy notes to make the system credible for real company data.
- Demonstrated ability to reason about model limitations, retrieval quality, cost and latency.

## What To Build First, Second And Third

First: Project 1, the LLM API Chat CLI. It teaches API calls, config, secrets, tests and basic prompt control.

Second: Project 2, the Document RAG Assistant. It teaches chunking, embeddings, retrieval and grounded generation.

Third: Project 5, the Evaluation Harness, even before the final prototype. Evaluations make your later projects much stronger.

Then build Project 3 and Project 4, and finish by combining everything in Project 6.

## Interview Preparation

Question: What is RAG?

Strong student answer: RAG means retrieval-augmented generation. Instead of asking the model from memory alone, the system retrieves relevant external context first and sends that context with the question. This is useful when the answer depends on private, recent or domain-specific documents.

Question: Why use a vector database?

Strong student answer: A vector database stores embeddings and supports similarity search. It lets the system find text with similar meaning even when the query uses different words. In RAG, it is usually the retrieval layer.

Question: What are embeddings?

Strong student answer: Embeddings are numerical vectors that represent semantic meaning. Similar texts should have nearby vectors. They are useful for search, clustering, recommendations and duplicate detection.

Question: What is chunking and why does it matter?

Strong student answer: Chunking splits long documents into smaller pieces before embedding. If chunks are too small, they lose context. If they are too large, retrieval becomes noisy and may exceed the model context budget. Good chunking improves both retrieval and citations.

Question: How would you choose chunk size?

Strong student answer: I would start with a simple token or character size, add overlap, preserve headings and metadata, then evaluate retrieval hit rate on realistic questions. The best size depends on document structure and query type.

Question: What is an AI agent?

Strong student answer: An AI agent is an LLM-driven system that can choose actions, use tools and continue based on results. The important part is not autonomy for its own sake, but a controlled loop where the application defines tools, permissions and logging.

Question: What is tool calling?

Strong student answer: Tool calling is when the model requests a structured function call, such as searching documents or calculating a value. The application executes the tool and returns the result. The model should not directly control unsafe actions.

Question: How do you evaluate an LLM application?

Strong student answer: I separate components. For RAG, I evaluate retrieval quality, answer correctness, citation quality, latency and cost. I use a dataset of realistic questions and compare results across prompt, model and retrieval changes.

Question: What causes hallucinations?

Strong student answer: Hallucinations can come from missing context, ambiguous prompts, overconfident model behavior, bad retrieval or asking for information the model does not have. RAG, citations and evaluations reduce the risk but do not eliminate it.

Question: How do you improve bad RAG answers?

Strong student answer: I first check whether the right chunks were retrieved. If retrieval is bad, I improve chunking, metadata filters, embeddings, reranking or query rewriting. If retrieval is good but the answer is bad, I improve the prompt, output format or model choice.

Question: How do API costs matter?

Strong student answer: Costs depend on input tokens, output tokens, model choice and number of calls. RAG can increase input tokens because retrieved context is included. I would log token usage, cap context size and use cheaper models where quality is enough.

Question: How do you handle latency?

Strong student answer: I measure each stage: loading, embedding, vector search, model call and post-processing. Common improvements include caching, smaller context, faster models, streaming and avoiding unnecessary tool calls.

Question: What data privacy concerns exist?

Strong student answer: API keys must not be committed. Sensitive documents should be handled intentionally, with clear provider policies, access controls and logging rules. I would avoid sending unnecessary personal data and redact when possible.

Question: What should be logged?

Strong student answer: I would log model name, prompt version, retrieval IDs, latency, errors and high-level request metadata. I would avoid logging raw secrets or sensitive user data unless there is a clear secure reason.

Question: What is MCP?

Strong student answer: Model Context Protocol is a standard way to connect AI clients to tools, files, databases and external systems. Instead of every client integrating tools differently, an MCP server can expose capabilities in a consistent way.

Question: When would you avoid an agent?

Strong student answer: I would avoid an agent when the workflow is simple and deterministic. A direct pipeline is easier to test, cheaper and safer. Agents are useful when the system needs flexible tool use or multi-step decisions.

Question: What makes a student AI project credible?

Strong student answer: A credible project has a real problem, runnable code, tests, documentation, known limitations and evaluation results. It should show engineering judgment, not just a flashy prompt demo.

