# Architecture

## Processing Flow

```mermaid
flowchart LR
    A[Company CSV] --> D[Import and validation]
    B[Invoice documents] --> D
    C[Transaction CSV] --> D
    D --> E[(PostgreSQL)]
    E --> F[Invoice categorisation]
    E --> G[Payment reconciliation]
    E --> H[Anomaly detection]
    F --> I[Review dashboard]
    G --> I
    H --> I
    E --> J[Evidence retrieval]
    J --> K[Optional LLM explanation]
    K --> I
    I --> L[Human approval log]
```

The import layer parses three invoice layouts and validates required fields before storage. Invoice categories come from a TF IDF and logistic regression pipeline. Reconciliation scores amount, date and reference agreement. Anomaly detection combines Isolation Forest output with transparent amount and counterparty rules.

The assistant retrieves database records before producing an answer. It works without an external model through deterministic summaries. When an API model is configured, the same evidence is placed in a restricted prompt. The assistant cannot approve a reconciliation.

## Data Model

```mermaid
erDiagram
    COMPANY ||--o{ INVOICE : owns
    COMPANY ||--o{ TRANSACTION : owns
    INVOICE ||--o{ RECONCILIATION : proposes
    TRANSACTION ||--o{ RECONCILIATION : proposes
    TRANSACTION ||--o{ ANOMALY : triggers
    RECONCILIATION ||--o{ APPROVAL_EVENT : records
```

`Company`, `Invoice` and `Transaction` hold imported facts. `Reconciliation` and `Anomaly` hold model output. `ApprovalEvent` stores the reviewer, action and timestamp for each accepted match.

## Decision Boundary

Models suggest categories, matches and anomalies. The application does not execute payments or change accounting records. A reconciliation remains `suggested` until a named reviewer approves it. The approval is written to a separate audit table.

## Provider Boundary

The core calculations do not depend on an LLM. Optional model access uses an OpenAI compatible endpoint configured through environment variables. Keys remain outside source control.

