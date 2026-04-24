# SecureGen AI
## Lab 1 - Project Plan

## 1. Problem Statement & Objectives

### Background
Recent advances in AI-assisted programming tools such as GitHub Copilot, Amazon CodeWhisperer, and Tabnine have improved developer productivity by generating code suggestions in real time. However, these systems also introduce a major cybersecurity risk: they can reproduce insecure coding patterns learned from public code corpora and present them to developers as plausible solutions. Evidence from the SecurityEval benchmark by Siddiq and Santos (MSR4P&S 2022) shows that this risk is not theoretical. Across 130 Python prompts spanning 75 CWE vulnerability types, 84.6% of Copilot-generated outputs were insecure, with the weak cryptography category reaching a 100% insecure generation rate. These findings demonstrate that AI code completion systems can systematically amplify common vulnerability patterns.

### Problem Statement
Despite the growing use of AI code generation tools, there is still a gap in real-time, context-aware vulnerability detection for AI-generated code. Existing static analyzers are often applied after code is written, which delays feedback and reduces their usefulness during code generation workflows. Developers therefore need a system that can identify insecure patterns immediately, classify the vulnerability type, explain why the code is risky, and recommend a secure remediation path. SecureGen AI addresses this gap by combining machine learning, explainability, and developer-facing interfaces into a single platform for secure code assessment.

### Objectives
1. Build a curated four-class vulnerability dataset from SecurityEval by mapping 75 CWE types into `sql_injection`, `hardcoded_secret`, `weak_crypto`, and `other_vuln`.
2. Fine-tune `microsoft/codebert-base` to classify insecure Python code into the four target categories, with a target precision of at least 90% for production-ready detection quality.
3. Develop a RESTful backend API that accepts submitted code snippets, performs inference, and returns the predicted vulnerability class, confidence score, and remediation guidance.
4. Create an interactive web dashboard that allows users to submit code, visualize results, inspect historical analyses, and understand system outputs clearly.
5. Implement an explainability layer using attention-based token highlighting so users can see which parts of the code most influenced the model's decision.

## 2. Main Features

| Feature Name | Description | Priority |
| --- | --- | --- |
| Vulnerability Detection | Classifies submitted code into `sql_injection`, `hardcoded_secret`, `weak_crypto`, or `other_vuln` using a fine-tuned CodeBERT model. | P0 |
| Attention Heatmap | Highlights influential tokens using model attention weights to provide transparent, developer-friendly explanations. | P0 |
| Remediation Suggestions | Maps predicted CWE patterns to secure coding fixes and short guidance notes for rapid remediation. | P0 |
| REST API | Exposes endpoints such as `POST /analyze`, `GET /history`, and `GET /health` for application and tooling integration. | P0 |
| Web Dashboard | Provides a frontend interface for code submission, confidence display, trend tracking, and analysis review. | P1 |
| Multi-Language Support | Prepares the platform architecture to support additional languages beyond Python in future project phases. | P2 |
| Dataset Explorer | Allows users to inspect labeled vulnerability examples, source distributions, and category breakdowns from SecurityEval. | P1 |
| CI/CD Plugin Stub | Defines an initial integration stub for pipeline-based scanning in pull requests and continuous integration workflows. | P2 |

## 3. Task Breakdown (WBS)

### 1. Data Engineering
- 1.1 Load the SecurityEval dataset from Hugging Face and validate schema consistency.
- 1.2 Map each CWE identifier into one of the four SecureGen AI categories using deterministic rules.
- 1.3 Perform data cleaning checks for duplicates, missing values, and label consistency.
- 1.4 Engineer auxiliary features such as token counts and line counts for exploratory analysis and baseline models.
- 1.5 Export cleaned analysis-ready data for notebook experiments and backend testing.

### 2. ML Model
- 2.1 Train baseline classical models using TF-IDF and engineered features for comparative benchmarking.
- 2.2 Fine-tune CodeBERT for four-class vulnerability classification on SecurityEval.
- 2.3 Evaluate models using 5-fold stratified cross-validation because the dataset contains only 130 samples.
- 2.4 Implement attention-weight extraction for token-level explainability.
- 2.5 Save and version the best-performing model artifact for backend deployment.

### 3. Backend API
- 3.1 Build FastAPI endpoints for code analysis, history retrieval, and health checks.
- 3.2 Implement the inference pipeline for tokenization, model scoring, softmax confidence calculation, and response formatting.
- 3.3 Add a remediation rule engine that maps vulnerability predictions to secure code fixes and CWE references.
- 3.4 Integrate Redis caching to reduce repeated inference latency for duplicate snippets.
- 3.5 Persist analysis results and metadata in PostgreSQL for traceability and dashboard reporting.

### 4. Frontend
- 4.1 Design a React-based interface for code submission, prediction display, and usability-focused workflows.
- 4.2 Build result components for risk badges, confidence indicators, attention heatmaps, and remediation panels.
- 4.3 Add a history view with recent analyses and simple trend visualizations.
- 4.4 Ensure responsive behavior across desktop and mobile layouts.
- 4.5 Connect frontend actions to backend API endpoints and validate end-to-end interaction flows.

### 5. Deployment & Docs
- 5.1 Containerize the frontend, backend, database, and cache using Docker for reproducible setup.
- 5.2 Define environment configuration, health checks, and service dependencies for local deployment.
- 5.3 Prepare project documentation covering architecture, setup steps, and usage instructions.
- 5.4 Produce academic deliverables including the EDA report, benchmarking report, and product documentation.
- 5.5 Draft a CI/CD integration stub to support future automated scanning workflows.

## 4. Timeline & Roles

### 6-Week Sprint Timeline

| Week | Goal | Deliverables |
| --- | --- | --- |
| Week 1 | Define scope and prepare dataset workflow | Project plan, architecture draft, dataset loading notebook, CWE mapping rules |
| Week 2 | Complete exploratory data analysis | Cleaned dataset, engineered features, six EDA visualizations, EDA report draft |
| Week 3 | Benchmark baseline models | TF-IDF pipeline, Logistic Regression, Random Forest, SVM, Gradient Boosting results |
| Week 4 | Fine-tune and evaluate CodeBERT | Trained CodeBERT model, cross-validation metrics, confusion matrix, attention analysis |
| Week 5 | Build product prototype | FastAPI backend, React dashboard or demo app, PostgreSQL and Redis integration |
| Week 6 | Finalize integration and documentation | End-to-end demo, product documentation, final benchmark report, presentation-ready outputs |

### Team Roles

| Role | Responsibilities |
| --- | --- |
| ML Engineer | Owns baseline benchmarking, CodeBERT fine-tuning, evaluation metrics, and explainability extraction. |
| Backend Dev | Builds FastAPI services, inference orchestration, remediation engine, and storage/cache integration. |
| Frontend Dev | Develops the dashboard or demo interface, user interactions, result visualization, and responsiveness. |
| Data Analyst/QA | Performs EDA, validates dataset integrity, tests outputs, and checks consistency of findings and reports. |
| Project Manager | Coordinates schedule, tracks risks, aligns deliverables across labs, and ensures academic requirements are met. |

## 5. Risk Analysis

| Risk | Likelihood | Impact | Mitigation Strategy |
| --- | --- | --- | --- |
| Insufficient data | High | High | Use 5-fold stratified cross-validation, class balancing strategies, and careful regularization to reduce overfitting on the 130-sample dataset. |
| GPU constraints | Medium | High | Use efficient batch sizes, sequence truncation, and checkpoint saving; fall back to CPU-compatible experiments for smaller runs if needed. |
| API latency | Medium | Medium | Add Redis caching, optimize tokenization and inference, and keep sequence length limited to practical bounds. |
| High false positive rate | Medium | High | Tune decision thresholds, compare against classical baselines, and incorporate error analysis before deployment claims. |
| Annotation disagreements | Low | Medium | Use deterministic CWE-to-category mapping rules and document category assumptions clearly in the methodology. |
| Scope creep | Medium | Medium | Prioritize P0 features first, treat multi-language support and CI/CD plugins as stretch goals, and review weekly progress against deliverables. |
| Integration failures | Medium | High | Define clear API contracts early, test frontend-backend-model interactions incrementally, and use Dockerized services for consistent environments. |

## Conclusion
SecureGen AI is designed as a practical and research-grounded platform for detecting insecure patterns in AI-generated code. By leveraging SecurityEval, combining baseline ML benchmarking with CodeBERT fine-tuning, and exposing results through a usable product interface, the project delivers both academic value and applied cybersecurity relevance. The plan above provides a structured path for completing the project within six weeks while managing the technical and organizational risks associated with a small but security-critical dataset.
