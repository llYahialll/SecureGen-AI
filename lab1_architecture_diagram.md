# SecureGen AI
## Lab 1 - Step 2 Architecture Diagram

## Product Architecture Overview
SecureGen AI follows a four-tier architecture that separates user interaction, application logic, machine learning inference, and persistent storage. This structure supports modular development, clearer responsibilities across the stack, and easier scaling of the system as the project evolves. The architecture is designed to support real-time code analysis, explainable predictions, caching for repeated requests, and storage of historical results for later inspection.

## ASCII Architecture Diagram

```text
+----------------------------------------------------------------------------------+
| TIER 1 - FRONTEND (React + Vite)                                                 |
|----------------------------------------------------------------------------------|
| Components:                                                                      |
| - Code submission editor (CodeMirror)                                            |
| - Risk gauge                                                                     |
| - Attention heatmap                                                              |
| - Trend charts (Recharts)                                                        |
| - Remediation panel                                                              |
|                                                                                  |
| Interactions:                                                                    |
| - POST /analyze -----------------------------------------------------------------+----+
| - GET /history ------------------------------------------------------------------+--+ |
+----------------------------------------------------------------------------------+  | |
                                                                                      | |
                                                                                      v |
+----------------------------------------------------------------------------------+  | |
| TIER 2 - BACKEND API (FastAPI, Python)                                           |  | |
|----------------------------------------------------------------------------------|  | |
| Endpoints:                                                                       |  | |
| - POST /analyze                                                                  |  | |
| - GET /history                                                                   |  | |
| - GET /health                                                                    |  | |
|                                                                                  |  | |
| Components:                                                                      |  | |
| - Inference pipeline: tokenize -> model -> softmax                               |  | |
| - Remediation rule engine: CWE -> fix map                                        |  | |
| - Redis cache check                                                              |  | |
+----------------------------------------------------------------------------------+  | |
            |                               |                            ^             | |
            | snippet tokens                | cache check                | query history| |
            v                               v                            |   (dashed)   | |
+-------------------------------+   +---------------------------+        |             | |
| TIER 3 - ML MODEL LAYER       |   | Redis Cache              |--------+             | |
|-------------------------------|   |---------------------------|                      | |
| - CodeBERT fine-tuned         |   | snippet hash -> result    |                      | |
| - 3 binary classifiers:       |   | TTL = 1 hour              |                      | |
|   SQLi, Secrets, Weak Crypto  |   +---------------------------+                      | |
| - Attention weight extractor  |                                                        |
+-------------------------------+                                                        |
            |                                                                               |
            | label + confidence                                                            |
            v                                                                               |
+-------------------------------------------------------------------------------------------+
| TIER 4 - STORAGE                                                                          |
|-------------------------------------------------------------------------------------------|
| PostgreSQL: snippets, analyses, users tables                                              |
| Dataset store: annotated training corpus                                                   |
+-------------------------------------------------------------------------------------------+
            ^
            |
            +------------------------------ write result ------------------------------------+
```

## Data Flows

1. `POST /analyze` moves downward from the frontend to the backend when a user submits a code snippet for inspection.
2. `snippet tokens` move downward from the backend API to the ML model layer after tokenization and preprocessing.
3. `label + confidence` move upward from the model layer to the backend after inference is complete.
4. `write result` moves downward from the backend to PostgreSQL so each completed analysis can be stored for history and reporting.
5. `cache check` flows from the backend to Redis and back to the backend to avoid repeated inference for identical snippets.
6. `query history` flows from PostgreSQL to the backend and is returned to the frontend when users request prior analyses.

## Tier Explanations

### Tier 1 - Frontend (React + Vite)
The frontend is the user-facing layer of SecureGen AI. It provides an interface for entering code, viewing vulnerability predictions, inspecting attention-based highlights, and reviewing historical trends. By isolating user interaction in the React application, the system can offer a responsive and developer-friendly experience without embedding model logic directly in the browser.

### Tier 2 - Backend API (FastAPI, Python)
The backend API coordinates all runtime operations after a request leaves the frontend. It receives submitted code through `POST /analyze`, performs preprocessing, checks Redis for cached outputs, sends code to the model for inference, applies remediation mappings, and returns a structured response. It also exposes `GET /history` for historical analysis retrieval and `GET /health` for service monitoring and deployment checks.

### Tier 3 - ML Model Layer
The ML layer contains the vulnerability detection intelligence of the platform. Its core is a fine-tuned CodeBERT model specialized for secure code classification, supported by binary detection logic for SQL injection, hardcoded secrets, and weak cryptography, as well as an attention extraction mechanism for explainability. This tier transforms code tokens into class predictions and confidence values that the backend can translate into actionable results.

### Tier 4 - Storage
The storage tier preserves both operational and training-related data. PostgreSQL stores submitted snippets, analysis outcomes, and user-linked metadata, while Redis acts as a fast temporary cache keyed by snippet hashes with a one-hour TTL. A separate dataset store holds the annotated SecurityEval-derived corpus used for experimentation, benchmarking, and future retraining.

## Summary
This architecture allows SecureGen AI to combine academic model experimentation with a practical product workflow. The separation of responsibilities across the frontend, backend, model, and storage tiers ensures that the project remains maintainable, scalable, and aligned with the lab requirement for a clear four-layer product architecture with traceable data flows.
