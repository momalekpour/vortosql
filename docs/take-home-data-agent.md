# Take-Home Test: Senior Software Developer — Data Agents

## Overview

You are applying for a **Software Developer** role on the **Data Agents** team. This team builds AI-powered agents that translate natural language into data queries, with a strong focus on safety, accuracy, and guardrails.

This take-home test evaluates your ability to build a simple **Natural Language to SQL (NL2SQL)** application. You are expected to use **AI/agentic tooling** (e.g., GitHub Copilot, Claude, ChatGPT, Cursor, etc.) to assist you in building the solution.

**Time limit: 48 hours from receipt.**

---

## Objective

Build a **console application** that:

1. Accepts natural language questions from the user
2. Dynamically generates SQL queries against a provided SQLite database (`employees.db`)
3. Executes the generated SQL and returns the results to the user

A UI is **not required** — a terminal/console-based interactive loop is perfectly acceptable.

---

## Requirements

### Language

Use one of the following:

- **Python**
- **C# (.NET)**
- **Node.js / TypeScript**

### LLM Provider

Your choice. You may use any LLM provider or model (OpenAI, Anthropic, Azure OpenAI, Ollama, etc.). If an API key is required, document how the evaluator should supply their own key.

### Framework

No specific framework is required. Use whatever approach you feel best demonstrates your skills — whether that's a lightweight script, LangChain, Semantic Kernel, a custom agent loop, or anything else.

---

## Application Behavior

### Startup — Department Selection (Mandatory Guardrail)

When the application starts, it must:

1. **Randomly select** one of three departments: `Sales`, `Marketing`, or `Engineering`
2. **Log the selected department** to the console (e.g., `"[INFO] Department selected: Marketing"`)
3. **Enforce this department as a guardrail for all queries** — every SQL query generated must filter results to only the selected department. Results from other departments must **never** be returned.

This guardrail is **mandatory** and will be a key evaluation criterion.

### Query Loop

After startup, the application should:

1. Prompt the user for a natural language question
2. Generate a SQL query based on the question and the database schema
3. Execute the SQL against the provided `employees.db` SQLite database
4. Display the results in a readable format
5. Repeat until the user exits (e.g., by typing `exit` or `quit`)

### Queryable Domains

The user can ask questions about:

- **Employee details** — names, roles, departments, salaries, bonuses, start dates
- **Certifications** — which employees hold which certifications, when they were achieved
- **Benefits** — benefits packages, remaining balances

Example questions a user might ask:

- *"Who are the software engineers?"*
- *"Which employees have an AWS certification?"*
- *"What is the average salary?"*
- *"List employees who started after 2023 and their certifications"*
- *"Who has the highest remaining benefits balance?"*

---

## Provided Database

You will receive an `employees.db` SQLite file containing pre-populated test data. The schema is as follows:

### Employee

| Column | Type | Constraints |
|---|---|---|
| EmployeeId | INTEGER | PRIMARY KEY |
| Name | TEXT | NOT NULL |
| Department | TEXT | NOT NULL — one of: `Sales`, `Marketing`, `Engineering` |
| Role | TEXT | NOT NULL |
| EmploymentStartDate | TEXT | NOT NULL — format: `YYYY-MM-DD` |
| SalaryAmount | REAL | NOT NULL |
| YearlyBonusAmount | REAL | |

### Certification

| Column | Type | Constraints |
|---|---|---|
| CertificationId | INTEGER | PRIMARY KEY |
| EmployeeId | INTEGER | NOT NULL, FOREIGN KEY → Employee(EmployeeId) |
| CertificationName | TEXT | NOT NULL |
| DateAchieved | TEXT | NOT NULL — format: `YYYY-MM-DD` |

### Benefits

| Column | Type | Constraints |
|---|---|---|
| BenefitId | INTEGER | PRIMARY KEY |
| EmployeeId | INTEGER | NOT NULL, FOREIGN KEY → Employee(EmployeeId) |
| BenefitsPackage | TEXT | NOT NULL |
| RemainingBalance | REAL | NOT NULL |

### Relationships

- `Certification.EmployeeId` → `Employee.EmployeeId` (many-to-one; an employee may have zero or more certifications)
- `Benefits.EmployeeId` → `Employee.EmployeeId` (many-to-one; an employee may have zero or more benefits records)

---

## Deliverables

1. **GitHub Repository or ZIP file** — public or private (if private, grant access to the evaluator). Must contain runnable code.
2. **README** — must include:
   - Setup and installation instructions
   - How to run the application
   - Brief description of your architecture/approach
   - Which AI tools you used during development and how they helped
3. **Live Demo** — be prepared to run the application, demonstrate it handling various queries, and explain your design decisions and tradeoffs.

---

## Acceptance Criteria

| # | Criterion |
|---|---|
| 1 | Application starts, randomly selects one of the three departments, and logs the selection |
| 2 | User can enter natural language questions and receive query results |
| 3 | Generated SQL is valid and produces correct results from the provided database |
| 4 | **Department guardrail is enforced** — results are always filtered to the selected department; no cross-department data leakage |
| 5 | Application handles edge cases gracefully (invalid queries, empty results, ambiguous questions) |
| 6 | Code is clean, well-structured, and follows idiomatic practices for the chosen language |
| 7 | README contains clear setup instructions, architecture description, and AI tooling notes |
| 8 | Candidate can demo the solution live and articulate their design rationale |

---

## Notes

- Focus on **correctness and guardrail enforcement** over feature breadth
- You are encouraged — and expected — to use AI coding assistants throughout development
- There is no single "right" architecture; we want to see how you think about the problem
- If you make assumptions, document them in your README

Good luck!
