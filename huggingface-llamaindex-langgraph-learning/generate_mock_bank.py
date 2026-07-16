import csv
import json
import random
import textwrap
from datetime import date, timedelta
from pathlib import Path

random.seed(42)
base = Path("mock_bank")


def mkdirs():
    for folder in [
        "customers",
        "accounts",
        "loans",
        "risk",
        "compliance",
        "products",
        "support",
        "branches",
        "documents",
        "examples",
        "bank_app/src/auth",
        "bank_app/src/data",
        "bank_app/src/pages",
        "bank_app/src/components",
        "bank_app/src/agent",
    ]:
        (base / folder).mkdir(parents=True, exist_ok=True)


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


first_names = [
    "Alice",
    "Alicia",
    "Ben",
    "Fatima",
    "Lukas",
    "Sofia",
    "Milan",
    "Naomi",
    "Omar",
    "Ingrid",
    "Daniel",
    "Maya",
    "Hanna",
    "Jonas",
    "Nora",
    "Elias",
    "Leila",
    "Arjun",
    "Priya",
    "Mateo",
    "Sara",
    "Noah",
    "Eva",
    "Yusuf",
    "Clara",
    "Anton",
    "Mei",
    "Carlos",
    "Amara",
    "Victor",
    "Elena",
    "Samir",
    "Julia",
    "Thomas",
    "Nadia",
    "Peter",
    "Iris",
    "Leo",
    "Mina",
    "Oscar",
]
last_names = [
    "Rivera",
    "Carter",
    "Noor",
    "Andersson",
    "Lindgren",
    "Petrovic",
    "Chen",
    "Haddad",
    "Bergstrom",
    "Mensah",
    "Khan",
    "Patel",
    "Garcia",
    "Muller",
    "Johansson",
    "Smith",
    "Nguyen",
    "Rossi",
    "Ahmed",
    "Kowalski",
    "Ivanov",
    "Wilson",
    "Brown",
    "Kim",
    "Silva",
    "Novak",
    "Ali",
    "Hansen",
    "Eriksson",
    "Larsson",
]
cities = [
    "Stockholm",
    "Gothenburg",
    "Malmo",
    "Uppsala",
    "Vasteras",
    "Orebro",
    "Linkoping",
    "Helsingborg",
    "Norrkoping",
    "Umea",
    "Lund",
    "Jonkoping",
]
countries = [
    "Sweden",
    "Norway",
    "Denmark",
    "Finland",
    "Germany",
    "France",
    "Spain",
    "United Kingdom",
    "Netherlands",
    "Canada",
]
employers = [
    "Nordic Solar AB",
    "Blue Harbor Logistics",
    "PixelForge Studio",
    "Aster Health Group",
    "GreenLine Foods",
    "CivicWorks Consulting",
    "BrightPath Education",
    "Metsa Design",
    "Aurora Robotics",
    "Freelance",
    "City Hospital",
    "North Pier Retail",
]
rm_ids = [f"RM-{i:03d}" for i in range(1, 16)]
branch_ids = [f"BR-{i:03d}" for i in range(1, 11)]


def build_data():
    branches = []
    for i, bid in enumerate(branch_ids, 1):
        branches.append(
            {
                "branch_id": bid,
                "branch_name": f"{cities[(i - 1) % len(cities)]} Community Branch",
                "city": cities[(i - 1) % len(cities)],
                "country": "Sweden",
                "opened_date": f"{2008 + i}-03-15",
                "manager_name": f"{first_names[i + 3]} {last_names[i + 5]}",
                "phone": f"+46-8-55{i:03d}-{100 + i:03d}",
            }
        )

    customers = []
    forced = [
        ("Alice", "Rivera"),
        ("Alice", "Rivera"),
        ("Alicia", "Rivera"),
        ("Sofia", "Lindgren"),
        ("Erik", "Lindgren"),
        ("Hanna", "Lindgren"),
    ]
    for i in range(100):
        cid = f"CUST-{1001 + i}"
        fn, ln = forced[i] if i < len(forced) else (random.choice(first_names), random.choice(last_names))
        city = random.choice(cities)
        if i in [0, 1]:
            city = "Stockholm"
        kyc = random.choices(
            ["verified", "pending_review", "missing_documents", "expired", "overdue"],
            [55, 18, 10, 7, 10],
        )[0]
        customers.append(
            {
                "customer_id": cid,
                "first_name": fn,
                "last_name": ln,
                "date_of_birth": str(date(1958, 1, 1) + timedelta(days=random.randint(7000, 22000))),
                "country": random.choice(countries),
                "city": city,
                "address": f"{random.randint(10, 249)} {random.choice(['Birch', 'Harbor', 'Maple', 'North', 'Cedar', 'Canal', 'Market'])} Street Apt {random.randint(1, 80)}",
                "email": f"{fn.lower()}.{ln.lower()}{i + 1}@example.test",
                "phone": f"+46-70-{random.randint(1000000, 9999999)}",
                "employment_status": random.choice(
                    ["employed", "self_employed", "student", "retired", "unemployed", "contractor"]
                ),
                "employer_name": random.choice(employers),
                "annual_income": random.randrange(28000, 185000, 1000),
                "customer_since": str(date(2012, 1, 1) + timedelta(days=random.randint(0, 4800))),
                "kyc_status": kyc,
                "risk_rating": random.choices(["low", "medium", "high", "watchlist"], [45, 32, 16, 7])[0],
                "relationship_manager_id": random.choice(rm_ids),
            }
        )
    customers[10]["address"] = customers[11]["address"] = "44 Canal Street Apt 12"

    rms = [
        {
            "relationship_manager_id": rm,
            "full_name": f"{first_names[i]} {last_names[i + 2]}",
            "branch_id": branch_ids[i % 10],
            "email": f"rm{i + 1:03d}@northstarbank.test",
            "portfolio_size": random.randint(42, 118),
            "specialty": random.choice(
                ["mortgages", "small business", "premium banking", "credit rehabilitation", "general retail"]
            ),
        }
        for i, rm in enumerate(rm_ids)
    ]

    accounts, balances = [], []
    acct_num = 50001
    for customer in customers:
        for _ in range(random.choices([1, 2, 3], [55, 32, 13])[0]):
            account_id = f"ACC-{acct_num}"
            acct_num += 1
            status = random.choices(["open", "closed", "restricted"], [85, 10, 5])[0]
            opened = date(2014, 1, 1) + timedelta(days=random.randint(0, 4200))
            balance = round(random.uniform(-500, 55000), 2)
            account = {
                "account_id": account_id,
                "customer_id": customer["customer_id"],
                "account_type": random.choice(
                    ["checking", "savings", "salary", "business", "youth", "foreign_currency"]
                ),
                "currency": random.choices(["SEK", "EUR", "USD"], [80, 15, 5])[0],
                "opened_date": str(opened),
                "closed_date": "" if status != "closed" else str(opened + timedelta(days=random.randint(400, 2500))),
                "status": status,
                "branch_id": random.choice(branch_ids),
                "current_balance": balance,
                "overdraft_limit": random.choice([0, 1000, 2500, 5000, 10000]),
            }
            accounts.append(account)
            balances.append(
                {
                    "account_id": account_id,
                    "as_of_date": "2025-06-30",
                    "available_balance": balance - random.choice([0, 0, 100, 250]),
                    "ledger_balance": balance,
                    "pending_amount": round(random.uniform(0, 750), 2),
                    "currency": account["currency"],
                }
            )
    while len(accounts) < 155:
        customer = random.choice(customers)
        account_id = f"ACC-{acct_num}"
        acct_num += 1
        balance = round(random.uniform(50, 30000), 2)
        accounts.append(
            {
                "account_id": account_id,
                "customer_id": customer["customer_id"],
                "account_type": random.choice(["checking", "savings", "credit_card_settlement"]),
                "currency": "SEK",
                "opened_date": "2024-02-12",
                "closed_date": "",
                "status": "open",
                "branch_id": random.choice(branch_ids),
                "current_balance": balance,
                "overdraft_limit": random.choice([0, 2500, 5000]),
            }
        )
        balances.append(
            {
                "account_id": account_id,
                "as_of_date": "2025-06-30",
                "available_balance": balance,
                "ledger_balance": balance,
                "pending_amount": 0,
                "currency": "SEK",
            }
        )

    categories = [
        ("groceries", "Everyday retail food spend"),
        ("salary", "Payroll and income"),
        ("rent", "Housing payments"),
        ("utilities", "Recurring household bills"),
        ("travel", "Air, rail, lodging, and fuel"),
        ("cash", "ATM and cash services"),
        ("transfer", "Internal and external transfers"),
        ("loan_payment", "Loan repayment"),
        ("card", "Card purchase"),
        ("gambling", "Gambling or betting merchant"),
        ("crypto", "Digital asset exchange"),
        ("healthcare", "Medical and pharmacy"),
        ("subscription", "Recurring digital services"),
    ]
    merchants = {
        "groceries": ["FreshMart", "Nordic Grocer", "City Foods"],
        "salary": ["Employer Payroll", "Contractor Payout"],
        "rent": ["RentPay Services"],
        "utilities": ["Stockholm Energy", "Water Utility"],
        "travel": ["SkyNorth Airlines", "Metro Transit", "FuelPoint"],
        "cash": ["ATM Withdrawal"],
        "transfer": ["Internal Transfer", "Wire Transfer"],
        "loan_payment": ["Northstar Loan Payment"],
        "card": ["Market Square", "TechTown", "HomeStyle"],
        "gambling": ["Betting Arcade", "Lucky Odds"],
        "crypto": ["CoinHarbor Exchange"],
        "healthcare": ["Aster Pharmacy", "City Clinic"],
        "subscription": ["StreamCloud", "NewsDesk Plus"],
    }
    transactions, flagged_txns = [], []
    txn_id = 900001
    for start, end, count in [(date(2025, 1, 1), date(2025, 3, 31), 560), (date(2025, 4, 1), date(2025, 6, 30), 570)]:
        for _ in range(count):
            account = random.choice(accounts)
            cat = random.choices(
                [c[0] for c in categories],
                [18, 5, 8, 8, 9, 7, 13, 5, 13, 3, 2, 5, 4],
            )[0]
            amount = round(random.uniform(5, 2500), 2)
            amount = round(random.uniform(1000, 6500), 2) if cat in ["salary", "transfer"] and random.random() < 0.45 else -amount
            flagged = random.random() < 0.045 or (cat in ["gambling", "crypto"] and random.random() < 0.18)
            reason = "" if not flagged else random.choice(
                [
                    "velocity pattern",
                    "high-risk merchant",
                    "unusual geography",
                    "round-dollar transfers",
                    "customer dispute",
                    "possible account takeover",
                ]
            )
            row = {
                "transaction_id": f"TXN-{txn_id}",
                "account_id": account["account_id"],
                "transaction_date": str(start + timedelta(days=random.randint(0, (end - start).days))),
                "amount": amount,
                "currency": account["currency"],
                "merchant": random.choice(merchants[cat]),
                "merchant_category": cat,
                "country": random.choice(countries),
                "channel": random.choice(["card_present", "card_not_present", "mobile", "branch", "atm", "wire"]),
                "description": f"{cat.replace('_', ' ').title()} transaction at {random.choice(merchants[cat])}",
                "is_flagged": str(flagged).lower(),
                "flag_reason": reason,
            }
            transactions.append(row)
            if flagged:
                flagged_txns.append(row)
            txn_id += 1

    loan_apps, risk_scores, collateral, affordability = [], [], [], []
    for i in range(80):
        customer = random.choice(customers)
        app_id = f"LOAN-{30001 + i}"
        amount = random.randrange(5000, 85000, 500)
        debt = random.randrange(150, 3900, 25)
        income = int(customer["annual_income"])
        score = random.randint(510, 810)
        dti = round((debt * 12) / max(income, 1), 3)
        status = random.choices(
            ["submitted", "in_review", "approved", "declined", "manual_review", "withdrawn"],
            [14, 20, 28, 14, 20, 4],
        )[0]
        if customer["kyc_status"] in ["missing_documents", "overdue", "expired"] and status == "approved":
            status = "manual_review"
        reason = random.choice(
            [
                "meets standard policy",
                "DTI above preferred threshold",
                "borderline credit score",
                "missing KYC evidence",
                "strong repayment history",
                "manual review required",
            ]
        )
        loan_apps.append(
            {
                "loan_application_id": app_id,
                "customer_id": customer["customer_id"],
                "product_type": random.choice(
                    ["personal_loan", "auto_loan", "debt_consolidation", "green_home_improvement"]
                ),
                "requested_amount": amount,
                "currency": "SEK",
                "application_date": str(date(2025, 1, 1) + timedelta(days=random.randint(0, 180))),
                "declared_income": income,
                "monthly_debt_payments": debt,
                "credit_score": score,
                "employment_status": customer["employment_status"],
                "purpose": random.choice(
                    ["home repairs", "vehicle purchase", "debt consolidation", "education", "medical expenses", "small business equipment"]
                ),
                "status": status,
                "decision_reason": reason,
                "assigned_officer": random.choice(rm_ids),
            }
        )
        band = "low" if score >= 700 and dti < 0.35 else "medium" if score >= 620 and dti < 0.43 else "high"
        risk_scores.append(
            {
                "risk_score_id": f"RS-{i + 1:04d}",
                "loan_application_id": app_id,
                "customer_id": customer["customer_id"],
                "credit_score": score,
                "debt_to_income_ratio": dti,
                "payment_history_score": random.randint(45, 98),
                "fraud_risk_score": random.randint(1, 94),
                "affordability_result": "pass" if dti <= 0.43 and score >= 580 else "manual_review",
                "overall_risk_band": band,
                "risk_notes": reason,
            }
        )
        affordability.append(
            {
                "check_id": f"AFF-{i + 1:04d}",
                "application_id": app_id,
                "customer_id": customer["customer_id"],
                "monthly_net_income": round(income / 12 * 0.72, 2),
                "declared_monthly_debt": debt,
                "stressed_interest_rate": "8.00%",
                "surplus_after_stress": round(income / 12 * 0.72 - debt - random.randint(900, 2600), 2),
                "result": "pass" if dti <= 0.43 else "manual_review",
                "reviewer_note": "Generated mock affordability check; not a credit decision.",
            }
        )
        if random.random() < 0.35:
            collateral.append(
                {
                    "collateral_id": f"COL-{len(collateral) + 1:04d}",
                    "application_id": app_id,
                    "customer_id": customer["customer_id"],
                    "collateral_type": random.choice(["vehicle", "cash_savings", "equipment", "property_second_lien"]),
                    "estimated_value": random.randrange(3000, 120000, 1000),
                    "valuation_date": "2025-06-15",
                    "lien_status": random.choice(["clear", "existing_lien", "valuation_pending"]),
                }
            )

    active_loans = [
        {
            "loan_id": f"ACT-{40001 + i}",
            "loan_application_id": app["loan_application_id"],
            "customer_id": app["customer_id"],
            "product_type": app["product_type"],
            "principal_balance": round(float(app["requested_amount"]) * random.uniform(0.55, 0.98), 2),
            "interest_rate": round(random.uniform(4.2, 12.9), 2),
            "origination_date": "2025-02-15",
            "maturity_date": "2030-02-15",
            "payment_status": random.choices(
                ["current", "late_15", "late_30", "hardship_plan"], [75, 10, 8, 7]
            )[0],
        }
        for i, app in enumerate(random.sample(loan_apps, 40))
    ]

    mortgages = []
    for i in range(30):
        customer = random.choice(customers)
        amount = random.randrange(800000, 5200000, 50000)
        dti = round(random.uniform(0.21, 0.58), 3)
        mortgages.append(
            {
                "mortgage_application_id": f"MORT-{70001 + i}",
                "customer_id": customer["customer_id"],
                "property_city": random.choice(cities),
                "property_value": round(amount * random.uniform(1.05, 1.45), 2),
                "requested_amount": amount,
                "currency": "SEK",
                "application_date": str(date(2025, 1, 1) + timedelta(days=random.randint(0, 180))),
                "loan_to_value_ratio": round(random.uniform(0.55, 0.88), 3),
                "debt_to_income_ratio": dti,
                "stress_test_result": "pass" if dti <= 0.43 else "manual_review",
                "status": random.choice(["submitted", "in_review", "manual_review", "approved", "declined"]),
                "decision_notes": random.choice(
                    ["pending valuation", "stress test passed", "manual affordability review", "missing property insurance quote"]
                ),
            }
        )

    cards = []
    for i in range(50):
        customer = random.choice(customers)
        cards.append(
            {
                "credit_card_application_id": f"CARD-{80001 + i}",
                "customer_id": customer["customer_id"],
                "requested_limit": random.choice([10000, 20000, 30000, 50000, 75000]),
                "application_date": str(date(2025, 1, 1) + timedelta(days=random.randint(0, 180))),
                "credit_score": random.randint(520, 820),
                "status": random.choice(["submitted", "approved", "declined", "manual_review"]),
                "decision_reason": random.choice(
                    ["score within policy", "income verification pending", "recent delinquencies", "KYC check required", "limit reduced due to utilization"]
                ),
            }
        )

    fraud_alerts = []
    for i, transaction in enumerate(flagged_txns[:30]):
        account = next(a for a in accounts if a["account_id"] == transaction["account_id"])
        fraud_alerts.append(
            {
                "alert_id": f"ALERT-{i + 1:03d}",
                "transaction_id": transaction["transaction_id"],
                "account_id": transaction["account_id"],
                "customer_id": account["customer_id"],
                "alert_date": transaction["transaction_date"],
                "alert_type": transaction["flag_reason"],
                "severity": random.choice(["low", "medium", "high", "critical"]),
                "status": random.choice(["open", "investigating", "closed_false_positive", "escalated"]),
                "analyst_note": random.choice(
                    ["Customer contact pending", "Pattern matches Q2 fraud memo", "Confirmed travel note absent", "Merchant dispute opened"]
                ),
            }
        )

    sars = [
        {
            "sar_id": f"SAR-{i + 1:03d}",
            "customer_id": alert["customer_id"],
            "related_alert_id": alert["alert_id"],
            "filed_date": str(date.fromisoformat(alert["alert_date"]) + timedelta(days=random.randint(1, 3))),
            "status": random.choice(["draft", "filed", "under_review", "closed"]),
            "suspicion_summary": random.choice(
                [
                    "Structured transfers below review threshold",
                    "Unusual merchant category after dormant period",
                    "High-risk country transfer pattern",
                    "Possible mule account behavior",
                ]
            ),
        }
        for i, alert in enumerate(fraud_alerts[:25])
    ]

    kyc = [
        {
            "review_id": f"KYC-{i + 1:04d}",
            "customer_id": customer["customer_id"],
            "last_review_date": str((last := date(2023, 1, 1) + timedelta(days=random.randint(0, 900)))),
            "next_review_due": str(last + timedelta(days=random.choice([365, 730, 1095]))),
            "status": customer["kyc_status"],
            "missing_items": random.choice(
                ["", "proof_of_address", "source_of_funds", "identity_document", "beneficial_owner_declaration"]
            ),
            "reviewer": random.choice(rm_ids),
            "notes": random.choice(
                [
                    "standard review cycle",
                    "enhanced due diligence required",
                    "customer uploaded partial documents",
                    "address mismatch requires follow-up",
                ]
            ),
        }
        for i, customer in enumerate(customers)
    ]

    high_risk = [
        {
            "customer_id": customer["customer_id"],
            "risk_rating": customer["risk_rating"],
            "primary_reason": random.choice(
                ["KYC overdue", "fraud alert history", "high DTI exposure", "unusual transaction pattern", "recent complaints"]
            ),
            "review_owner": customer["relationship_manager_id"],
            "next_action_date": "2025-07-15",
        }
        for customer in customers
        if customer["risk_rating"] in ["high", "watchlist"] or customer["kyc_status"] in ["missing_documents", "overdue"]
    ]

    support, complaints = [], []
    for i in range(50):
        customer = random.choice(customers)
        support.append(
            {
                "ticket_id": f"TCK-{i + 1:04d}",
                "customer_id": customer["customer_id"],
                "opened_date": str(date(2025, 1, 1) + timedelta(days=random.randint(0, 180))),
                "channel": random.choice(["phone", "email", "secure_message", "branch"]),
                "category": random.choice(
                    ["card dispute", "loan question", "login issue", "address update", "unauthorized card use", "mortgage status", "fee question"]
                ),
                "priority": random.choice(["low", "normal", "high", "urgent"]),
                "status": random.choice(["open", "pending_customer", "resolved", "escalated"]),
                "summary": random.choice(
                    [
                        f"Customer referred to as {customer['first_name']} only in call notes.",
                        "Asked for update on loan application.",
                        "Reported unauthorized card use after travel.",
                        "Needs KYC upload link reset.",
                        "Question about mortgage affordability result.",
                    ]
                ),
            }
        )
    for i in range(20):
        customer = random.choice(customers)
        complaints.append(
            {
                "complaint_id": f"CMP-{i + 1:04d}",
                "customer_id": customer["customer_id"],
                "received_date": str(date(2025, 1, 1) + timedelta(days=random.randint(0, 180))),
                "category": random.choice(["service delay", "fee dispute", "credit decision explanation", "fraud handling", "branch experience"]),
                "severity": random.choice(["standard", "material", "executive_review"]),
                "status": random.choice(["new", "investigating", "resolved", "closed"]),
                "resolution_summary": random.choice(
                    [
                        "fee refunded as goodwill",
                        "decision letter reissued",
                        "case escalated to fraud team",
                        "customer accepted explanation",
                        "additional documents requested",
                    ]
                ),
            }
        )

    branch_perf = [
        {
            "branch_id": branch["branch_id"],
            "quarter": "2025-Q2",
            "new_accounts": random.randint(35, 120),
            "loan_applications": random.randint(12, 55),
            "complaints": random.randint(0, 9),
            "kyc_overdue_count": random.randint(2, 28),
            "customer_satisfaction_score": round(random.uniform(3.4, 4.8), 2),
        }
        for branch in branches
    ]

    profiles = [
        {
            "customer_id": customer["customer_id"],
            "preferred_name": customer["first_name"],
            "household_id": f"HH-{random.randint(200, 260)}",
            "profile_summary": f"{customer['first_name']} {customer['last_name']} is a {customer['employment_status']} customer in {customer['city']} with {customer['risk_rating']} risk rating and KYC status {customer['kyc_status']}.",
            "communication_preference": random.choice(["email", "phone", "secure_message"]),
            "rag_notes": [
                random.choice(
                    [
                        "Has asked about refinancing.",
                        "Often uses mobile channel.",
                        "May share address with another customer.",
                        "Borderline affordability in recent application.",
                        "Support history includes partial-name references.",
                    ]
                )
            ],
        }
        for customer in customers
    ]

    return locals()


def write_dataset(d):
    write_csv(base / "branches/branches.csv", d["branches"])
    write_csv(base / "branches/relationship_managers.csv", d["rms"])
    write_csv(base / "branches/branch_performance.csv", d["branch_perf"])
    write_csv(base / "customers/customers.csv", d["customers"])
    (base / "customers/customer_profiles.json").write_text(json.dumps(d["profiles"], indent=2), encoding="utf-8")
    write_csv(base / "customers/kyc_reviews.csv", d["kyc"])
    write_csv(base / "accounts/accounts.csv", d["accounts"])
    write_csv(base / "accounts/balances.csv", d["balances"])
    write_csv(base / "accounts/transaction_categories.csv", [{"category": c, "description": desc} for c, desc in d["categories"]])
    write_csv(base / "accounts/transactions_2025_q1.csv", [t for t in d["transactions"] if t["transaction_date"] < "2025-04-01"])
    write_csv(base / "accounts/transactions_2025_q2.csv", [t for t in d["transactions"] if t["transaction_date"] >= "2025-04-01"])
    write_csv(base / "loans/loan_applications.csv", d["loan_apps"])
    write_csv(base / "loans/active_loans.csv", d["active_loans"])
    write_csv(base / "loans/mortgage_applications.csv", d["mortgages"])
    write_csv(base / "loans/credit_card_applications.csv", d["cards"])
    write_csv(base / "loans/loan_risk_scores.csv", d["risk_scores"])
    write_csv(base / "loans/collateral_records.csv", d["collateral"])
    write_csv(base / "loans/affordability_checks.csv", d["affordability"])
    write_csv(base / "risk/fraud_alerts.csv", d["fraud_alerts"])
    write_csv(base / "risk/high_risk_customers.csv", d["high_risk"])
    write_csv(base / "compliance/suspicious_activity_reports.csv", d["sars"])
    write_csv(base / "support/customer_support_tickets.csv", d["support"])
    write_csv(base / "support/complaint_logs.csv", d["complaints"])


def write_docs():
    write_text(
        base / "README.md",
        """
        # Northstar Community Bank Mock Dataset

        This folder is a complete fictional banking data environment for RAG, tool-calling, and agent evaluation. All people, addresses, emails, account numbers, transactions, policies, and bank records are synthetic.

        ## Folder Guide

        - `customers/`: customer master records, KYC reviews, narrative notes, and JSON profiles.
        - `accounts/`: account records, balances, transaction files for 2025 Q1 and Q2, and transaction category metadata.
        - `loans/`: personal loan, mortgage, credit card, affordability, risk, collateral, and active loan records.
        - `risk/`: credit and fraud policy documents, fraud alerts, high-risk customer extracts, and committee minutes.
        - `compliance/`: AML, KYC, GDPR, SAR, and audit reference material.
        - `products/`: product terms for loans, mortgages, savings, cards, and business accounts.
        - `support/`: support tickets, complaint logs, and call summaries.
        - `branches/`: branches, relationship managers, and performance data.
        - `documents/`: internal memos, manuals, playbooks, and escalation procedures.
        - `examples/`: sample RAG and tool-calling questions.
        - `bank_app/`: a demo Streamlit portal scaffold using this local mock data.

        ## Example Workflows

        Load CSV files into DuckDB, SQLite, Pandas, or a tool runtime; index markdown documents for retrieval; ask an AI assistant to combine structured lookups with policy retrieval; and test ambiguity handling with duplicate names, partial support notes, and overlapping loan records.

        The assistant must never make final credit, compliance, or legal decisions.
        """,
    )
    write_text(
        base / "data_dictionary.md",
        """
        # Data Dictionary

        All identifiers are fictional. Customer IDs use `CUST-####`, accounts use `ACC-#####`, transactions use `TXN-######`, loan applications use `LOAN-#####`, mortgages use `MORT-#####`, card applications use `CARD-#####`, branches use `BR-###`, relationship managers use `RM-###`, alerts use `ALERT-###`, and SARs use `SAR-###`.

        ## customers/customers.csv

        Customer master table with `customer_id`, name, demographics, contact fields, employment, income, KYC status, risk rating, and relationship manager.

        ## accounts/*.csv

        Account, balance, category, and transaction tables. Negative amounts are customer outflows and positive amounts are inflows.

        ## loans/*.csv

        Loan, mortgage, card, risk, affordability, collateral, and active-loan records. Lending files reference existing customer IDs.

        ## risk, compliance, support, branches, documents

        Operational extracts and markdown files for RAG-style retrieval, policy lookup, audit support, and management reporting.
        """,
    )
    common = """
    ## Non-Decision Rule
    The AI assistant may retrieve facts, summarize evidence, compare cases, and draft review notes. It must never issue a final approval, rejection, SAR filing decision, or legal determination.

    ## Manual Review Triggers
    - Debt-to-income ratio above 43% requires manual review.
    - Credit score below 580 is high risk.
    - Missing, expired, or overdue KYC blocks new lending until resolved.
    - Suspicious activity reports must be escalated within 2 business days.
    """
    write_text(base / "risk/risk_policy.md", "# Risk Policy\n" + common)
    write_text(base / "risk/credit_scoring_policy.md", "# Credit Scoring Policy\n" + common + "\nScores 700 and above may proceed through standard review if affordability passes. Scores 620-699 require income and employment verification. Scores 580-619 require manual review and compensating factors.")
    write_text(base / "risk/risk_committee_minutes.md", "# Risk Committee Minutes - 2025 Q2\n\nThe committee reviewed rising fraud alerts involving card-not-present merchants, crypto exchanges, and round-dollar transfers. Mortgage affordability exceptions remain concentrated in high-income customers with high existing debt.")
    write_text(base / "compliance/aml_policy.md", "# AML Policy\n" + common + "\nStructured transfers, mule-account indicators, unusual international wires, and high-risk merchant activity require enhanced review.")
    write_text(base / "compliance/kyc_policy.md", "# KYC Policy\n" + common + "\nCustomers must have current identity evidence, proof of address, source-of-funds information when required, and beneficial ownership data for business relationships.")
    write_text(base / "compliance/gdpr_data_handling_policy.md", "# GDPR and Data Handling Policy\n\nEmployees may access customer data only for a legitimate banking purpose. Customers may view only their own profile, accounts, transactions, active loans, messages, and submitted applications.")
    write_text(base / "compliance/compliance_audit_findings.md", "# Compliance Audit Findings\n\n1. Several KYC reviews were overdue.\n2. Some support tickets refer to customers by first name only.\n3. Fraud alert closure notes were inconsistent.\n4. Loan exceptions above 43% DTI require stronger documentation.")
    products = {
        "personal_loan_terms.md": "Unsecured personal loans may be used for debt consolidation, education, vehicle deposits, or household repairs.",
        "mortgage_terms.md": "Mortgage applicants must pass affordability stress testing. Loan-to-value above 85% requires senior review.",
        "savings_account_terms.md": "Savings accounts support deposits, withdrawals, and interest accrual.",
        "credit_card_terms.md": "Credit card limits depend on score, income, utilization, KYC, and fraud history.",
        "business_account_terms.md": "Business accounts require beneficial owner checks and expected transaction volume review.",
    }
    for filename, body in products.items():
        write_text(base / "products" / filename, f"# {filename.replace('_', ' ').replace('.md', '').title()}\n\n{body}\n\nFictional demo terms only.")
    write_text(base / "customers/customer_notes.md", "# Customer Notes\n\n- CUST-1001 Alice Rivera and CUST-1002 Alice Rivera are separate customers with similar names.\n- CUST-1011 and CUST-1012 share an address at 44 Canal Street Apt 12.\n- Several Lindgren customers exist and are not all related.\n- Some support tickets mention only first names or partial surnames.")
    write_text(base / "support/call_summaries.md", "# Call Summaries\n\nMarch call notes include a customer named Alice asking about an auto loan and a separate Alice asking about a disputed card transaction. A Lindgren customer called about mortgage stress testing without a full customer ID.")
    write_text(base / "documents/internal_memo_lending_standards_2025.md", "# Internal Memo: Lending Standards 2025\n\nDTI above 43% requires manual review. Credit scores below 580 are high risk. Missing KYC blocks new lending. The AI assistant can draft summaries but cannot approve or decline credit.")
    write_text(base / "documents/internal_memo_fraud_patterns.md", "# Internal Memo: Fraud Patterns\n\nQ2 monitoring found increased alerts for crypto exchanges, round-dollar wires, card-not-present electronics purchases, and transactions shortly after contact detail changes.")
    write_text(base / "documents/onboarding_manual.md", "# Onboarding Manual\n\nSearch by customer ID first. If a name search returns multiple customers, ask for clarification. Customer data access must match a legitimate work purpose.")
    write_text(base / "documents/loan_officer_playbook.md", "# Loan Officer Playbook\n\nGather income, monthly debt, credit score, KYC status, purpose, affordability result, and risk notes. Draft recommendations must be reviewed by an authorized employee.")
    write_text(base / "documents/escalation_procedures.md", "# Escalation Procedures\n\nEscalate incomplete KYC to KYC operations. Escalate suspected fraud to fraud operations within 1 business day when severity is high or critical. Escalate potential SAR matters to compliance within 2 business days.")
    sections = {
        "Customer lookup": ["Find all accounts for Alice Rivera.", "Which Alice Rivera do you mean?", "Summarize customer CUST-1042.", "List customers sharing an address.", "Find all Lindgren customers."],
        "Loan risk": ["Which customers are high risk and why?", "What is the debt-to-income ratio for LOAN-30012?", "Which loan applications have DTI above 43%?", "Compare two similar personal loan applicants.", "Which active loans are late or on hardship plans?"],
        "Policy lookup": ["What policy applies if KYC is incomplete?", "What happens when credit score is below 580?", "Can the AI assistant approve a loan?", "When is mortgage stress testing required?", "What documentation is required for a lending exception?"],
        "Compliance": ["Find all customers whose KYC review is overdue.", "Which SARs are still drafts?", "What is the SAR escalation deadline?", "Which customers have missing source of funds evidence?", "Which audit findings mention support ticket ambiguity?"],
        "Fraud": ["Which transactions triggered fraud alerts in Q2?", "Which alerts mention crypto or high-risk merchants?", "Which customer has suspicious activity in March?", "Which alerts are still open?", "Which transactions may indicate account takeover?"],
        "Transactions": ["Show Q1 transactions for CUST-1001.", "Which accounts have suspicious transactions?", "Find large round-dollar transfers.", "Which customers have gambling transactions?", "Compare Q1 and Q2 flagged transaction counts."],
        "Ambiguous queries": ["Show me Alice loans.", "Find the Lindgren mortgage call.", "Which customer complained about card fraud?", "What documents support this loan decision?", "Which account has the 5000 SEK balance?"],
        "Management reporting": ["Which branch has the highest number of complaints?", "Which relationship manager has the largest portfolio?", "How many KYC reviews are overdue by branch?", "Which product has most manual reviews?", "Summarize Q2 risk committee concerns."],
    }
    body = ["# Sample Questions"]
    for section, questions in sections.items():
        body.append(f"\n## {section}")
        body.extend(f"- {q}" for q in questions)
    write_text(base / "examples/sample_questions.md", "\n".join(body))


def write_app():
    users = [
        {"user_id": "EMP-001", "email": "loan.officer@northstarbank.test", "password": "demo123", "role": "employee", "display_name": "Emma Loan Officer"},
        {"user_id": "CUST-1001", "email": "alice.rivera@example.test", "password": "demo123", "role": "customer", "customer_id": "CUST-1001", "display_name": "Alice Rivera"},
        {"user_id": "CUST-1042", "email": "demo.customer@example.test", "password": "demo123", "role": "customer", "customer_id": "CUST-1042", "display_name": "Demo Customer"},
    ]
    (base / "bank_app/src/auth/users.json").write_text(json.dumps(users, indent=2), encoding="utf-8")
    (base / "bank_app/src/data/local_changes.json").write_text(json.dumps({"employee_notes": [], "customer_updates": [], "loan_status_updates": []}, indent=2), encoding="utf-8")
    (base / "bank_app/src/data/customer_submitted_applications.json").write_text("[]\n", encoding="utf-8")
    (base / "bank_app/src/data/customer_messages.json").write_text("[]\n", encoding="utf-8")
    write_text(base / "bank_app/src/auth/permissions.md", "# Role Permissions\n\nEmployees may read customer, account, transaction, lending, risk, fraud, SAR summary, KYC, support, complaint, policy, and branch records. Employees may create mock notes and local status updates.\n\nCustomers may read only their own profile, accounts, transactions, active loans, submitted applications, product terms, general policy summaries, and messages. Customers may create loan applications and messages.\n\nCustomers cannot view other customers, internal compliance files, SAR details, risk committee minutes, employee-only notes, fraud investigation notes, or approve or reject loans.\n\nThe assistant may summarize and retrieve but must never make final credit, legal, compliance, or fraud decisions.")
    write_text(base / "bank_app/src/agent/README.md", "# Agent Placeholder\n\nReserved for a future LLM/RAG/tool-calling banking assistant. The current app returns deterministic placeholder responses and never calls a real model.")
    write_text(base / "bank_app/src/agent/tool_contracts.md", "# Tool Contracts\n\n- `search_policy_documents(query)`\n- `lookup_customer(customer_id)`\n- `search_customers(name_or_query)`\n- `get_customer_accounts(customer_id)`\n- `get_customer_transactions(customer_id, date_range)`\n- `get_loan_application(application_id)`\n- `calculate_debt_to_income(customer_id)`\n- `get_risk_summary(customer_id)`\n- `create_customer_message(customer_id, message)`\n- `create_loan_application(customer_id, application_data)`\n- `update_employee_note(employee_id, customer_id, note)`")
    write_text(base / "bank_app/src/agent/agent_placeholder.py", 'def generate_assistant_response(role: str, prompt: str) -> str:\n    lowered = prompt.lower()\n    if role == "customer" and any(term in lowered for term in ["sar", "risk committee", "other customer", "fraud alert"]):\n        return "I can answer general product questions and summarize your own records, but I cannot show internal or other-customer data."\n    if "kyc" in lowered:\n        return "Placeholder: incomplete, expired, or overdue KYC blocks new lending until resolved."\n    if "high-risk" in lowered or "high risk" in lowered:\n        return "Placeholder: high-risk cases are usually driven by DTI above 43%, credit score below 580, unresolved KYC, or fraud indicators."\n    return "Placeholder assistant response. A future LLM/RAG agent will search policy files and query structured records here."\n')
    write_text(base / "bank_app/requirements.txt", "streamlit>=1.36\npandas>=2.2")
    write_text(base / "bank_app/package.json", '{"scripts":{"note":"This scaffold uses Streamlit; run python -m streamlit run src/app.py"},"private":true}')
    write_text(base / "bank_app/.env.example", "MOCK_BANK_DATA_ROOT=..\nDEMO_SESSION_SECRET=not-for-production")
    write_text(base / "bank_app/README.md", "# Mock Bank App\n\nA Streamlit demo portal for the fictional Northstar Community Bank dataset. It uses fake login, fake sessions, and local JSON files only.\n\n## Install\n\n```bash\npython -m pip install -r requirements.txt\n```\n\n## Run\n\n```bash\npython -m streamlit run src/app.py\n```\n\nRun from `mock_bank/bank_app/`.\n\n## Fake Credentials\n\n- Employee: `loan.officer@northstarbank.test` / `demo123`\n- Customer: `alice.rivera@example.test` / `demo123`\n- Customer: `demo.customer@example.test` / `demo123`\n\nLocal writes are saved under `src/data/`. Replace `src/agent/agent_placeholder.py` with a real tool-calling backend later.")
    write_text(base / "bank_app/src/pages/README.md", "# Pages\n\nThe Streamlit scaffold implements Login, Employee dashboard, Customer dashboard, Customer search, Customer detail, Loan review, Risk overview, Fraud alerts, Compliance/KYC overview, Messages inbox, Customer loan application form, and AI assistant panel in `src/app.py`.")
    write_text(base / "bank_app/src/components/README.md", "# Components\n\nReusable UI components can be extracted here as the demo grows.")
    write_text(base / "bank_app/src/data/README.md", "# Local Data\n\nThis folder stores mock local writes for the demo portal.")

    app = r'''
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent.parent
USERS_PATH = ROOT / "auth" / "users.json"
LOCAL_CHANGES = ROOT / "data" / "local_changes.json"
CUSTOMER_APPS = ROOT / "data" / "customer_submitted_applications.json"
CUSTOMER_MESSAGES = ROOT / "data" / "customer_messages.json"

st.set_page_config(page_title="Northstar Bank Mock Portal", layout="wide")

@st.cache_data
def read_csv(rel):
    return pd.read_csv(DATA_ROOT / rel)

@st.cache_data
def read_users():
    return json.loads(USERS_PATH.read_text())

def read_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default

def write_json(path, data):
    path.write_text(json.dumps(data, indent=2))

def fake_login():
    st.title("Northstar Bank Mock Portal")
    st.caption("Fictional demo data only. Fake authentication for local testing.")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Log in", type="primary"):
        user = next((u for u in read_users() if u["email"] == email and u["password"] == password), None)
        if user:
            st.session_state.user = user
            st.rerun()
        st.error("Invalid fake credentials")

def assistant_panel(role):
    st.subheader("AI Assistant Panel")
    prompt = st.text_area("Prompt", placeholder="Summarize customer CUST-1042")
    if st.button("Ask assistant"):
        from agent.agent_placeholder import generate_assistant_response
        st.info(generate_assistant_response(role, prompt))

def employee_portal(user):
    customers = read_csv("customers/customers.csv")
    accounts = read_csv("accounts/accounts.csv")
    q1 = read_csv("accounts/transactions_2025_q1.csv")
    q2 = read_csv("accounts/transactions_2025_q2.csv")
    loans = read_csv("loans/loan_applications.csv")
    mortgages = read_csv("loans/mortgage_applications.csv")
    risk = read_csv("loans/loan_risk_scores.csv")
    kyc = read_csv("customers/kyc_reviews.csv")
    fraud = read_csv("risk/fraud_alerts.csv")
    sar = read_csv("compliance/suspicious_activity_reports.csv")
    tickets = read_csv("support/customer_support_tickets.csv")
    page = st.sidebar.radio("Employee pages", ["Dashboard", "Customer search", "Customer detail", "Loan review", "Risk overview", "Fraud alerts", "Compliance/KYC", "Messages inbox", "AI assistant"])
    st.sidebar.write(user["display_name"])
    if st.sidebar.button("Log out"):
        st.session_state.clear(); st.rerun()
    if page == "Dashboard":
        st.title("Employee Dashboard")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers", len(customers)); c2.metric("Loan apps", len(loans)); c3.metric("Fraud alerts", len(fraud)); c4.metric("Open tickets", int((tickets.status != "resolved").sum()))
        st.dataframe(customers.head(20), use_container_width=True)
    elif page == "Customer search":
        st.title("Customer Search")
        q = st.text_input("Name, city, customer ID, or email")
        data = customers
        if q:
            data = data[data.apply(lambda row: q.lower() in " ".join(map(str, row.values)).lower(), axis=1)]
        st.dataframe(data, use_container_width=True)
    elif page == "Customer detail":
        st.title("Customer Detail")
        cid = st.selectbox("Customer", customers.customer_id.tolist())
        cust_accounts = accounts[accounts.customer_id == cid]
        st.dataframe(customers[customers.customer_id == cid], use_container_width=True)
        st.subheader("Accounts"); st.dataframe(cust_accounts, use_container_width=True)
        st.subheader("Transactions"); st.dataframe(pd.concat([q1, q2])[lambda df: df.account_id.isin(cust_accounts.account_id)], use_container_width=True)
        note = st.text_area("Employee note")
        if st.button("Save note"):
            data = read_json(LOCAL_CHANGES, {"employee_notes": []})
            data.setdefault("employee_notes", []).append({"employee_id": user["user_id"], "customer_id": cid, "note": note, "created_at": datetime.utcnow().isoformat()})
            write_json(LOCAL_CHANGES, data); st.success("Saved locally")
    elif page == "Loan review":
        st.title("Loan Review"); st.dataframe(loans, use_container_width=True)
        app_id = st.selectbox("Application to update", loans.loan_application_id.tolist())
        new_status = st.selectbox("Mock status", ["submitted", "in_review", "manual_review", "approved", "declined", "withdrawn"])
        if st.button("Save mock status update"):
            data = read_json(LOCAL_CHANGES, {"loan_status_updates": []})
            data.setdefault("loan_status_updates", []).append({"employee_id": user["user_id"], "loan_application_id": app_id, "status": new_status, "updated_at": datetime.utcnow().isoformat()})
            write_json(LOCAL_CHANGES, data); st.success("Saved locally")
        st.subheader("Mortgage Applications"); st.dataframe(mortgages, use_container_width=True)
    elif page == "Risk overview":
        st.title("Risk Overview"); st.dataframe(risk, use_container_width=True)
    elif page == "Fraud alerts":
        st.title("Fraud Alerts"); st.dataframe(fraud, use_container_width=True)
    elif page == "Compliance/KYC":
        st.title("Compliance and KYC"); st.dataframe(kyc, use_container_width=True); st.subheader("Suspicious Activity Reports"); st.dataframe(sar, use_container_width=True)
    elif page == "Messages inbox":
        st.title("Messages Inbox"); st.json(read_json(CUSTOMER_MESSAGES, []))
    else:
        assistant_panel("employee")

def customer_portal(user):
    cid = user["customer_id"]
    customers = read_csv("customers/customers.csv")
    accounts = read_csv("accounts/accounts.csv")
    active = read_csv("loans/active_loans.csv")
    tx = pd.concat([read_csv("accounts/transactions_2025_q1.csv"), read_csv("accounts/transactions_2025_q2.csv")])
    page = st.sidebar.radio("Customer pages", ["Dashboard", "Loan application", "Messages", "AI assistant"])
    st.sidebar.write(user["display_name"])
    if st.sidebar.button("Log out"):
        st.session_state.clear(); st.rerun()
    own_accounts = accounts[accounts.customer_id == cid]
    own_tx = tx[tx.account_id.isin(own_accounts.account_id)]
    if page == "Dashboard":
        st.title("Customer Dashboard")
        st.subheader("My Profile"); st.dataframe(customers[customers.customer_id == cid], use_container_width=True)
        st.subheader("My Accounts"); st.dataframe(own_accounts, use_container_width=True)
        st.subheader("My Transactions"); st.dataframe(own_tx, use_container_width=True)
        st.subheader("My Active Loans"); st.dataframe(active[active.customer_id == cid], use_container_width=True)
    elif page == "Loan application":
        st.title("Customer Loan Application Form")
        product = st.selectbox("Product", ["personal_loan", "mortgage"])
        amount = st.number_input("Requested amount", min_value=1000, value=50000, step=1000)
        purpose = st.text_input("Purpose")
        if st.button("Submit mock application"):
            data = read_json(CUSTOMER_APPS, [])
            data.append({"customer_id": cid, "product": product, "amount": amount, "purpose": purpose, "submitted_at": datetime.utcnow().isoformat(), "status": "submitted"})
            write_json(CUSTOMER_APPS, data); st.success("Submitted locally")
        st.json([a for a in read_json(CUSTOMER_APPS, []) if a.get("customer_id") == cid])
    elif page == "Messages":
        st.title("Messages")
        msg = st.text_area("Message to bank employees")
        if st.button("Send message"):
            data = read_json(CUSTOMER_MESSAGES, [])
            data.append({"customer_id": cid, "message": msg, "created_at": datetime.utcnow().isoformat(), "status": "new"})
            write_json(CUSTOMER_MESSAGES, data); st.success("Message saved locally")
        st.json([m for m in read_json(CUSTOMER_MESSAGES, []) if m.get("customer_id") == cid])
    else:
        assistant_panel("customer")

if "user" not in st.session_state:
    fake_login()
elif st.session_state.user["role"] == "employee":
    employee_portal(st.session_state.user)
else:
    customer_portal(st.session_state.user)
'''
    write_text(base / "bank_app/src/app.py", app)


if __name__ == "__main__":
    mkdirs()
    data = build_data()
    write_dataset(data)
    write_docs()
    write_app()
    print(
        json.dumps(
            {
                "path": str(base.resolve()),
                "customers": len(data["customers"]),
                "accounts": len(data["accounts"]),
                "transactions": len(data["transactions"]),
                "loan_applications": len(data["loan_apps"]),
                "active_loans": len(data["active_loans"]),
                "mortgages": len(data["mortgages"]),
                "cards": len(data["cards"]),
                "fraud_alerts": len(data["fraud_alerts"]),
                "sars": len(data["sars"]),
                "support_tickets": len(data["support"]),
                "complaints": len(data["complaints"]),
                "branches": len(data["branches"]),
                "relationship_managers": len(data["rms"]),
            },
            indent=2,
        )
    )
