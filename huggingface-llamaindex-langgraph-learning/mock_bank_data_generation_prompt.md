# Prompt: Generate a Realistic Mock Bank Dataset

You are generating a complete fictional banking data environment for an AI agent/RAG/tool-calling project.

The goal is to create a realistic mock bank folder full of structured and unstructured data that feels like an actual internal banking knowledge base. This data will later be used to test an AI banking assistant that can search documents, query tables, reason over customer and loan data, and answer operational banking questions.

Do not use real people, real account numbers, real addresses, or real sensitive information. Everything must be fictional.

## Output Format

Create a folder named:

```txt
mock_bank/
```

Inside it, create many realistic files and subfolders. Use a mix of:

```txt
.csv
.xlsx
.json
.md
.txt
.pdf-ready markdown files
```

If you cannot create actual `.xlsx` files, create `.csv` files with clear names and mention that they can be converted to Excel.

## Main Goal

Generate enough data that it feels like a small-to-medium bank with:

- customers
- accounts
- loans
- mortgages
- credit cards
- transactions
- risk scoring
- compliance policies
- KYC records
- fraud alerts
- internal procedures
- product documentation
- customer support logs
- branch data
- employee/relationship manager assignments
- audit notes
- regulatory-style documents
- ambiguous cases where several customers or loans partially match a query

The data should be rich enough for a RAG system and tool-calling agent to answer questions such as:

- “Which customers have high loan risk?”
- “Show me all loans connected to Alice Rivera.”
- “What policy applies to mortgage affordability checks?”
- “Which customer has suspicious activity in March?”
- “Compare two similar loan applicants.”
- “Which accounts belong to customer CUST-1042?”
- “Find all customers whose KYC review is overdue.”
- “What documents support this loan decision?”
- “What are the bank’s rules for credit approval?”
- “Which transactions triggered fraud alerts?”

## Required Folder Structure

Create this structure:

```txt
mock_bank/
  README.md
  data_dictionary.md
  customers/
    customers.csv
    customer_profiles.json
    kyc_reviews.csv
    customer_notes.md
  accounts/
    accounts.csv
    balances.csv
    transactions_2025_q1.csv
    transactions_2025_q2.csv
    transaction_categories.csv
  loans/
    loan_applications.csv
    active_loans.csv
    mortgage_applications.csv
    credit_card_applications.csv
    loan_risk_scores.csv
    collateral_records.csv
    affordability_checks.csv
  risk/
    risk_policy.md
    credit_scoring_policy.md
    fraud_alerts.csv
    high_risk_customers.csv
    risk_committee_minutes.md
  compliance/
    aml_policy.md
    kyc_policy.md
    gdpr_data_handling_policy.md
    suspicious_activity_reports.csv
    compliance_audit_findings.md
  products/
    personal_loan_terms.md
    mortgage_terms.md
    savings_account_terms.md
    credit_card_terms.md
    business_account_terms.md
  support/
    customer_support_tickets.csv
    complaint_logs.csv
    call_summaries.md
  branches/
    branches.csv
    relationship_managers.csv
    branch_performance.csv
  documents/
    internal_memo_lending_standards_2025.md
    internal_memo_fraud_patterns.md
    onboarding_manual.md
    loan_officer_playbook.md
    escalation_procedures.md
  examples/
    sample_questions.md
    evaluation_set.json
    golden_workflows.md
  audit/
    audit_log.csv
    audit_policy.md
  api/
    mock_api_contract.md
    endpoint_examples.json
  bank_app/
    README.md
    package.json or requirements.txt
    .env.example
    src/
      app entrypoint
      auth/
      data/
      pages/
      components/
      agent/
```

## Audit Trail Requirements

Create an audit folder:

```txt
mock_bank/audit/
```

Include:

```txt
audit_log.csv
audit_policy.md
```

The audit log should simulate realistic system activity, including:

- employee login events
- customer login events
- customer profile views
- customer data edits
- loan status changes
- KYC status changes
- fraud alert reviews
- suspicious activity report reviews
- AI assistant queries
- AI assistant tool calls
- customer messages sent
- employee replies sent

Use columns like:

```txt
audit_id
timestamp
actor_user_id
actor_role
action_type
target_type
target_id
summary
source_ip
status
```

Example actions:

```txt
VIEW_CUSTOMER
UPDATE_CUSTOMER_NOTE
UPDATE_LOAN_STATUS
VIEW_KYC_RECORD
CREATE_CUSTOMER_MESSAGE
AI_QUERY
AI_TOOL_CALL
EXPORT_REPORT
PERMISSION_DENIED
```

The audit policy should explain:

- what actions must be logged
- why auditability matters in banking
- which actions are employee-only
- which actions customers can perform
- how AI assistant activity should be logged
- that final credit decisions must remain human-owned

## RBAC Matrix

Create a role-based access control matrix in:

```txt
mock_bank/bank_app/src/auth/permissions.md
```

Include at least these roles:

```txt
customer
employee
loan_officer
compliance_officer
branch_manager
admin
```

Even if the app only implements `customer` and `employee` login at first, document the richer real-bank permission model.

The RBAC matrix should define permissions for:

- customer profile read
- customer profile edit
- account read
- transaction read
- loan application create
- loan application review
- loan status update
- KYC read
- KYC update
- fraud alert read
- fraud alert update
- suspicious activity report read
- support message create
- support message reply
- AI assistant use
- AI assistant access to internal policy
- audit log read
- data export

Use a clear table format.

Important rules:

- Customers may only see their own data.
- Customers may create loan applications and messages.
- Customers may ask general product/policy questions.
- Customers must not access internal risk/compliance notes.
- Employees may search customers and view operational records.
- Loan officers may review loan applications but not unilaterally approve final credit decisions.
- Compliance officers may view AML/KYC/SAR records.
- Branch managers may view branch-level reporting.
- Admins may manage mock users and settings.
- All sensitive employee actions must be logged.

## Mock API Layer

Create a mock API contract folder:

```txt
mock_bank/api/
```

Include:

```txt
mock_api_contract.md
endpoint_examples.json
```

The API contract should describe REST-style endpoints that a real app or agent could call later.

Include endpoints such as:

```txt
GET /api/customers
GET /api/customers/{customer_id}
PATCH /api/customers/{customer_id}
GET /api/customers/{customer_id}/accounts
GET /api/customers/{customer_id}/transactions
GET /api/customers/{customer_id}/loans
GET /api/loan-applications
GET /api/loan-applications/{loan_application_id}
PATCH /api/loan-applications/{loan_application_id}
GET /api/risk/customer/{customer_id}
GET /api/fraud-alerts
PATCH /api/fraud-alerts/{alert_id}
GET /api/kyc/{customer_id}
PATCH /api/kyc/{customer_id}
GET /api/messages
POST /api/messages
GET /api/policies/search?q=
POST /api/agent/query
GET /api/audit-log
```

For each endpoint, define:

- purpose
- allowed roles
- request parameters
- response shape
- example response
- audit logging behavior

The generated app does not need a full real backend, but its data-access functions should be organized so these endpoints can be implemented later.

## Evaluation Set

Create an evaluation set for testing the future AI agent:

```txt
mock_bank/examples/evaluation_set.json
```

Include at least 50 evaluation examples.

Each example should include:

```json
{
  "eval_id": "EVAL-001",
  "category": "loan_risk",
  "user_role": "employee",
  "question": "Summarize the risk factors for LOAN-30012.",
  "expected_sources": [
    "loans/loan_applications.csv",
    "loans/loan_risk_scores.csv",
    "risk/credit_scoring_policy.md"
  ],
  "expected_answer_notes": "Should mention DTI, credit score, KYC status, and that final approval must be human-owned.",
  "must_not_include": [
    "final approval",
    "guaranteed rejection"
  ]
}
```

Evaluation categories should include:

- customer lookup
- ambiguous customer lookup
- loan risk summary
- policy question
- KYC/compliance question
- fraud triage
- transaction lookup
- customer-facing question
- permission denial
- auditability
- management reporting

Include examples where the correct behavior is to ask for clarification.

Include examples where the correct behavior is to refuse because the role lacks permission.

Include examples where the correct behavior is to cite policy and record IDs.

## Golden Workflows

Create:

```txt
mock_bank/examples/golden_workflows.md
```

This file should document realistic workflows that the future AI system should support.

Include at least these workflows:

### Workflow 1: Loan Officer Review

- employee searches customer
- employee opens loan application
- system retrieves customer profile, accounts, risk score, affordability check, KYC status, and lending policy
- AI drafts a risk summary
- AI cites source records
- employee adds a note
- audit log records the action

### Workflow 2: Customer Loan Application

- customer logs in
- customer views products
- customer applies for loan
- system stores application
- employee sees application
- AI helps employee summarize the application

### Workflow 3: KYC Escalation

- employee views overdue KYC customer
- AI retrieves KYC policy
- compliance officer reviews missing documents
- status is updated
- audit log records the change

### Workflow 4: Fraud Alert Triage

- employee opens fraud alerts
- AI summarizes suspicious transaction pattern
- employee escalates or dismisses alert
- audit log records decision

### Workflow 5: Customer Support Message

- customer sends message
- employee receives message
- AI drafts a safe reply
- employee edits and sends
- audit log records reply

### Workflow 6: Permission Denial

- customer asks for internal risk notes
- system refuses
- AI explains limitation politely
- audit log records permission denial

## Optional Deployable Web App Scaffold

Also create a separate deployable web application scaffold named:

```txt
mock_bank/bank_app/
```

The app should be a realistic internal/customer banking portal that uses the generated mock bank data as local files.

Important: this app is only a mock demo scaffold. Do not implement real production authentication, real banking security, real payment movement, or real credit decisions. Use fake login and fake sessions only.

The app should be clean, modular, and easy for another developer to later integrate a real RAG/tool-calling AI agent.

### Preferred App Type

Use one of these options:

- Next.js / React web app
- Streamlit app
- FastAPI backend with simple frontend
- Gradio demo app

Choose the option that is easiest to generate completely and run locally.

### App Roles

Create two user roles:

```txt
employee
customer
```

Create a small fake users file:

```txt
mock_bank/bank_app/src/auth/users.json
```

Example fake users:

```json
[
  {
    "user_id": "EMP-001",
    "email": "loan.officer@northstarbank.test",
    "password": "demo123",
    "role": "employee",
    "display_name": "Emma Loan Officer"
  },
  {
    "user_id": "CUST-1001",
    "email": "alice.rivera@example.test",
    "password": "demo123",
    "role": "customer",
    "customer_id": "CUST-1001",
    "display_name": "Alice Rivera"
  }
]
```

### Employee Portal

Employees should be able to:

- log in as a fake employee
- view a dashboard
- search customers
- view customer profiles
- view accounts
- view transactions
- view loan applications
- view mortgage applications
- view risk scores
- view KYC/compliance status
- create/edit mock notes
- update mock customer fields
- update mock loan application status
- view fraud alerts
- view suspicious activity reports
- access an AI assistant panel

Employee write access should be simulated locally. If implementing real persistence is difficult, save edits to a local JSON file such as:

```txt
mock_bank/bank_app/src/data/local_changes.json
```

The employee AI assistant placeholder should support prompts like:

- Summarize customer CUST-1042.
- Find high-risk loan applications.
- What policy applies to incomplete KYC?
- Draft a loan review summary.
- Which accounts have suspicious transactions?

For now, the assistant can return placeholder responses, but the code should clearly show where a real LLM/RAG/tool-calling backend will be integrated later.

### Customer Portal

Customers should be able to:

- log in as a fake customer
- view only their own profile
- view only their own accounts
- view only their own transactions
- view only their own active loans
- apply for a new personal loan
- apply for a mortgage
- send a message to bank employees
- view responses/messages from employees
- ask general product/policy questions through an AI assistant panel

Customers must not be able to:

- view other customers
- edit bank records directly
- view internal compliance files
- view internal risk committee notes
- approve or reject loans
- access employee-only data

Customer loan applications should be stored as local mock records, for example:

```txt
mock_bank/bank_app/src/data/customer_submitted_applications.json
```

Customer messages should be stored as local mock records, for example:

```txt
mock_bank/bank_app/src/data/customer_messages.json
```

### Role-Based Access Rules

Include a clear role permission file:

```txt
mock_bank/bank_app/src/auth/permissions.md
```

It should explain:

- what employees can read
- what employees can edit
- what customers can read
- what customers can create
- what customers are blocked from accessing
- what the AI assistant is allowed to answer for each role

### AI Assistant Placeholder

Create an `agent/` folder with placeholder files such as:

```txt
mock_bank/bank_app/src/agent/README.md
mock_bank/bank_app/src/agent/agent_placeholder.py or agent_placeholder.ts
mock_bank/bank_app/src/agent/tool_contracts.md
```

The agent placeholder should describe future tools:

- `search_policy_documents(query)`
- `lookup_customer(customer_id)`
- `search_customers(name_or_query)`
- `get_customer_accounts(customer_id)`
- `get_customer_transactions(customer_id, date_range)`
- `get_loan_application(application_id)`
- `calculate_debt_to_income(customer_id)`
- `get_risk_summary(customer_id)`
- `create_customer_message(customer_id, message)`
- `create_loan_application(customer_id, application_data)`
- `update_employee_note(employee_id, customer_id, note)`

The placeholder must make clear that the real LLM integration will be added later.

### App Pages

Create these pages/screens:

- Login
- Employee dashboard
- Customer dashboard
- Customer search
- Customer detail
- Loan review
- Risk overview
- Fraud alerts
- Compliance/KYC overview
- Messages inbox
- Customer loan application form
- AI assistant panel

### Visual Design and UI Quality

The app must look like a polished, modern financial technology product, not a quick AI-generated demo.

The UI should feel appropriate for a serious bank or fintech product in 2026:

- clean
- calm
- premium
- trustworthy
- fast
- smooth
- professional
- highly readable
- enterprise-grade

Avoid anything that looks like:

- early 2000s websites
- generic Bootstrap demo pages
- random bright gradients
- oversized emoji-heavy layouts
- low-effort AI-generated “vibecoded” UI
- cluttered dashboards with no hierarchy
- inconsistent spacing
- inconsistent font sizes
- weak contrast
- fake-looking cards everywhere
- cheap crypto-app styling

Use a modern banking/fintech visual language inspired by high-quality products such as:

- Stripe Dashboard
- Revolut Business
- Wise Business
- Linear
- Vercel Dashboard
- Ramp
- Mercury Banking
- modern enterprise admin dashboards

Do not copy any brand directly. Use them only as quality references.

#### UI Requirements

The app should include:

- a polished login screen
- a clean sidebar or top navigation
- beautiful dashboard cards
- well-designed tables with sorting/filtering/search-ready structure
- smooth hover states
- clear loading/empty/error states
- role badges
- risk status badges
- KYC status badges
- loan status badges
- readable customer profile pages
- modern form layouts
- professional spacing and alignment
- responsive layout for desktop and tablet widths
- clear information hierarchy
- subtle shadows/borders
- accessible color contrast
- consistent typography
- tasteful icons if available

#### Design System

Create a small internal design system with:

- color tokens
- spacing scale
- typography scale
- reusable button styles
- reusable card styles
- reusable table styles
- reusable status badge styles
- reusable form field styles
- reusable page header component

Prefer a restrained color palette:

- dark navy / slate
- white / off-white
- soft gray backgrounds
- muted blue accents
- green for healthy/approved states
- amber for warnings/manual review
- red for high-risk/blocked states

The interface should look credible enough that someone could imagine it being used by bank employees.

#### Animation and Smoothness

Add subtle, professional motion:

- smooth page transitions if easy
- hover transitions
- button press states
- loading skeletons
- drawer/modal transitions if used

Do not over-animate. Motion should make the product feel refined, not gimmicky.

#### AI Assistant UX

The AI assistant panel should look like a serious internal copilot:

- visible source/citation area placeholder
- tool activity placeholder
- confidence/safety notice area
- role-aware warning text
- message history
- input box with suggested prompts

For employee users, suggested prompts should include:

- Summarize this customer.
- Pull relevant lending policy.
- Explain this risk score.
- Find missing KYC documents.
- Draft a loan review note.

For customer users, suggested prompts should include:

- What loan products are available?
- How do I apply for a personal loan?
- What documents do I need for a mortgage?
- Send a message to my advisor.

The assistant should never visually imply that it can make final credit approval decisions.

#### Quality Bar

Before finishing, review the generated UI and improve anything that looks generic, messy, outdated, or obviously AI-generated.

The final app should feel like a credible portfolio project for an AI engineer building enterprise banking agents.

### App README

Create `mock_bank/bank_app/README.md` explaining:

- how to install dependencies
- how to run locally
- fake login credentials
- employee vs customer permissions
- where the mock bank data is loaded from
- where local edits are saved
- where the future AI/RAG agent should be integrated

## Data Size

Generate a lot of data, not just toy examples.

Minimum target:

- 100 fictional customers
- 150 accounts
- 1,000+ transactions across Q1 and Q2
- 80 loan applications
- 40 active loans
- 30 mortgage applications
- 50 credit card applications
- 30 fraud alerts
- 25 suspicious activity reports
- 50 support tickets
- 20 complaint logs
- 10 branches
- 15 relationship managers

The data should be internally consistent. IDs must match across files.

For example:

- `customers.csv` has `customer_id`
- `accounts.csv` references existing `customer_id`
- `transactions_2025_q1.csv` references existing `account_id`
- `loan_applications.csv` references existing `customer_id`
- `loan_risk_scores.csv` references existing `loan_application_id`
- `relationship_managers.csv` references existing branch IDs

## Naming and IDs

Use realistic but fictional IDs:

```txt
CUST-1001
ACC-50001
TXN-900001
LOAN-30001
MORT-70001
CARD-80001
BR-001
RM-001
SAR-001
ALERT-001
```

Use diverse fictional names from different backgrounds.

Example customers:

- Alice Rivera
- Ben Carter
- Fatima Noor
- Lukas Andersson
- Sofia Lindgren
- Milan Petrović
- Naomi Chen
- Omar Haddad
- Ingrid Bergström
- Daniel Mensah

Do not make all customers clean and simple. Include:

- customers with similar names
- customers sharing addresses
- customers with multiple accounts
- customers with old closed accounts
- customers with missing KYC documents
- customers with borderline credit scores
- customers with high income but high debt
- customers with low income but strong repayment history
- customers with suspicious transactions
- customers connected to multiple support complaints

## Realistic Bank Fields

Include columns like these where relevant:

### Customers

```txt
customer_id
first_name
last_name
date_of_birth
country
city
address
email
phone
employment_status
employer_name
annual_income
customer_since
kyc_status
risk_rating
relationship_manager_id
```

### Accounts

```txt
account_id
customer_id
account_type
currency
opened_date
closed_date
status
branch_id
current_balance
overdraft_limit
```

### Transactions

```txt
transaction_id
account_id
transaction_date
amount
currency
merchant
merchant_category
country
channel
description
is_flagged
flag_reason
```

### Loan Applications

```txt
loan_application_id
customer_id
product_type
requested_amount
currency
application_date
declared_income
monthly_debt_payments
credit_score
employment_status
purpose
status
decision_reason
assigned_officer
```

### Risk Scores

```txt
risk_score_id
loan_application_id
customer_id
credit_score
debt_to_income_ratio
payment_history_score
fraud_risk_score
affordability_result
overall_risk_band
risk_notes
```

## Policy Documents

Create realistic internal policy documents in markdown.

They should include:

- approval thresholds
- manual review triggers
- affordability rules
- KYC rules
- AML rules
- fraud escalation rules
- GDPR/data handling notes
- examples of allowed and disallowed decisions

Important: include enough policy detail that an AI agent can answer policy questions from these files.

Example policy content:

- Customers with debt-to-income ratio above 43% require manual review.
- Credit scores below 580 are high risk.
- Mortgage applicants must pass affordability stress testing.
- Missing KYC documents block new lending.
- Suspicious activity reports must be escalated within 2 business days.
- The assistant must never make final approval/rejection decisions.

## Ambiguity and RAG Challenge Cases

Add deliberate ambiguity so the AI assistant must ask clarification questions.

Examples:

- Two customers named Alice Rivera.
- A customer with two loan applications.
- Several “Lindgren” customers.
- Multiple accounts with similar balances.
- Similar transaction descriptions from different merchants.
- A customer with both a personal loan and a mortgage.
- Support tickets referring to customers by partial name only.

Document these challenge cases in:

```txt
examples/sample_questions.md
```

## Sample Questions File

Create `examples/sample_questions.md` with at least 40 questions.

Include categories:

```txt
Customer lookup
Loan risk
Policy lookup
Compliance
Fraud
Transactions
Ambiguous queries
Management reporting
```

Example questions:

- Which customers are high risk and why?
- Find all accounts for Alice Rivera.
- Which Alice Rivera do you mean?
- What policy applies if KYC is incomplete?
- Which transactions triggered fraud alerts in Q2?
- What is the debt-to-income ratio for LOAN-30012?
- Which mortgage applications require manual review?
- Summarize customer CUST-1042.
- Which support tickets mention unauthorized card use?
- Which branch has the highest number of complaints?

## README.md

Create a clear `README.md` explaining:

- what this mock dataset is
- that all data is fictional
- how the folders are organized
- which files are structured data
- which files are policy/reference documents
- how this data can be used for RAG and tool-calling agents
- example workflows

## Data Dictionary

Create `data_dictionary.md` explaining every file and major column.

## Important Quality Rules

- Make the data internally consistent.
- Use realistic banking language.
- Do not use real personal data.
- Do not make the dataset tiny.
- Do not make every answer obvious.
- Include edge cases and ambiguity.
- Include enough policy text for retrieval.
- Include enough tabular data for database/API-style querying.
- Include enough narrative notes for RAG-style search.

## Final Output

Return the complete generated folder/files.

If you are running in an environment where you can create files, actually create the folder and files.

If you cannot create files directly, output a downloadable archive or provide the full file contents in clearly separated sections.
