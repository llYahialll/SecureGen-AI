# SecureGen AI

SecureGen AI is a university Deep Learning and Cybersecurity project focused on detecting insecure patterns in AI-generated code. The project studies secure code generation risks using the SecurityEval benchmark and organizes the work into four labs: project planning, exploratory data analysis, model benchmarking, and a product demo.

## Project Focus

- Vulnerability classes: `sql_injection`, `hardcoded_secret`, `weak_crypto`, `other_vuln`
- Core model target: fine-tuned `microsoft/codebert-base`
- Dataset reference: SecurityEval by Siddiq and Santos
- Product goal: detect insecure code, explain risky tokens, and suggest remediation

## Repository Structure

```text
.
|-- demo/
|   `-- landing-page.html
|-- docs/
|   `-- labs/
|       |-- lab1/
|       |   |-- architecture-diagram.md
|       |   `-- project-plan.md
|       |-- lab2/
|       |   `-- eda-report.md
|       |-- lab3/
|       |   `-- benchmark-report.md
|       `-- lab4/
|           `-- product-documentation.md
|-- notebooks/
|   |-- lab2_eda_notebook.py
|   `-- lab3_benchmark_notebook.py
|-- src/
|   `-- SecureGenAIDemo.jsx
|-- index.html
|-- LICENSE
`-- README.md
```

## Key Files

- `index.html`: runnable single-file SecureGen AI demo
- `demo/landing-page.html`: Lab 4 landing page source
- `src/SecureGenAIDemo.jsx`: React demo component for the interactive app
- `notebooks/lab2_eda_notebook.py`: notebook-style EDA workflow
- `notebooks/lab3_benchmark_notebook.py`: notebook-style benchmarking workflow
- `docs/labs/`: organized lab deliverables and reports

## Running the Demo

Open `index.html` directly in a browser, or serve the repository root with a simple static server.

Example:

```bash
python -m http.server 4173
```

Then visit `http://127.0.0.1:4173/`.

## Notes

- The repository includes both the academic deliverables and the demo-oriented product assets.
- The Python notebook-style scripts are structured to match the lab prompt guide and can be converted into `.ipynb` notebooks later if needed.
- The React demo component is stored separately so the repo can evolve into a full Vite app without losing the current runnable single-file demo.
