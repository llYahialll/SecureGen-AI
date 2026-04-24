# SecureGen AI
## Product Documentation

## 1. What Is SecureGen AI?

SecureGen AI is a developer-facing tool that helps identify insecure patterns in AI-generated code before they are copied into real software projects. It is designed for students, developers, and engineering teams who use tools like GitHub Copilot, Amazon CodeWhisperer, and Tabnine and want a fast way to check whether generated code contains risky behavior. The product focuses on three high-priority vulnerability families in generated code: SQL injection, hardcoded secrets, and weak cryptography. Instead of only labeling code as risky, SecureGen AI also explains why the snippet was flagged and shows a safer remediation path.

## 2. How the Model Works

SecureGen AI is based on the SecurityEval benchmark, which the project guide describes as 130 Python snippets covering 75 CWE vulnerability types. These examples are mapped into four product classes: `sql_injection`, `hardcoded_secret`, `weak_crypto`, and `other_vuln`. The core model is CodeBERT (`microsoft/codebert-base`), which is fine-tuned for four-class vulnerability classification. In the project target setting, the model is expected to achieve roughly 85% macro-F1 on cross-validation, making it strong enough to support a practical academic demo and a prototype secure-coding product.

The analysis pipeline follows a simple flow:

`Code -> Tokenizer -> CodeBERT -> Softmax -> Label + Confidence`

When a user submits code, the snippet is tokenized and passed into the fine-tuned CodeBERT model. The model returns a predicted vulnerability label and a confidence score. SecureGen AI also uses attention weights to highlight risky tokens, helping users see what parts of the code influenced the model's decision. After prediction, a lightweight rule engine maps the detected issue to a concrete remediation example so the user can move quickly from detection to fixing the code.

## 3. How to Use the Product

1. Open the SecureGen AI web demo at `[URL]`.
2. Paste or type your code in the input box.
3. Select the programming language you want to inspect.
4. Click `Analyze Code`.
5. Read the risk badge shown in the result panel: `CRITICAL`, `HIGH`, `MEDIUM`, or `SAFE`.
6. Check the highlighted tokens to understand what triggered the alert.
7. Review the suggested remediation in the fix panel and apply the safer version of the code.

## 4. Interpreting Results

| Risk Level | Color | Meaning | What to do |
| --- | --- | --- | --- |
| CRITICAL | Red | A confirmed or highly likely vulnerability pattern was detected. | Apply the fix immediately and avoid merging the code unchanged. |
| HIGH | Orange | The code strongly resembles a known insecure pattern. | Review carefully and replace the risky construct with the suggested remediation. |
| MEDIUM | Yellow | A possible issue was found, but context may change the final interpretation. | Inspect the surrounding logic and verify whether the pattern is actually exploitable. |
| SAFE | Green | No known high-risk pattern was detected by the current analysis flow. | Continue reviewing the code normally, especially if it handles user input, credentials, or cryptography. |

## 5. Known Limitations

- The training benchmark is small, so model performance depends on a limited set of examples and should be interpreted carefully.
- The current project is focused on Python benchmark data, even if the product demo can simulate analysis for other languages.
- The public demo may use rule-based simulation for the interface layer, while the full production vision depends on the fine-tuned CodeBERT model.
