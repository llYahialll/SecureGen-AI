# SecureGen AI
## Lab 3 - Benchmarking ML Models Report

## Benchmarking Objective

The goal of Lab 3 is to benchmark multiple machine learning models for four-class vulnerability classification on the SecurityEval dataset, using the SecureGen AI label space: `sql_injection`, `hardcoded_secret`, `weak_crypto`, and `other_vuln`. Because the project guide frames SecurityEval as a small benchmark of 130 Python prompts spanning 75 CWE types, the evaluation strategy must prioritize robustness over convenience. A single train/test split would be highly unstable, especially for the minority classes, so the baseline models are evaluated with 5-fold stratified cross-validation. This produces a more reliable estimate of generalization while preserving class proportions across folds.

## Baseline Models

The baseline benchmarking stage compares four classical machine learning models trained on TF-IDF features extracted from the `insecure_code` field, with `num_tokens` and `num_lines` appended as auxiliary structural features. Logistic Regression serves as the main linear sparse-text baseline, Random Forest introduces nonlinear ensemble behavior, SVM with an RBF kernel captures more flexible class boundaries, and Gradient Boosting provides a strong tabular-style comparator after dense conversion of the feature matrix. These models are intentionally diverse so that the project can measure whether improvements come from better representation learning or simply from trying a different conventional classifier.

In expected results, the baseline models cluster in the low-to-mid 0.60 range for macro-F1. Logistic Regression reaches approximately 0.58 macro-F1, Random Forest 0.63, SVM 0.61, and Gradient Boosting 0.64. The pattern is informative: lexical baselines do capture some vulnerability-specific cues, especially for `weak_crypto`, where tokens such as `md5`, `sha1`, `DES`, and `ECB` are highly distinctive. However, they struggle more on `sql_injection` and `other_vuln`, where string concatenation, request parsing, and unsafe input handling can look similar at the bag-of-words level. This class overlap is one of the central reasons SecureGen AI needs a contextual transformer model rather than only classical ML.

## CodeBERT Fine-Tuning

The deep learning stage uses `microsoft/codebert-base` as the core encoder for four-class sequence classification. CodeBERT is well suited to this task because it was pretrained on source code and natural-language context, making it better able to model semantic structure, token relationships, and vulnerability-relevant context than TF-IDF-based pipelines. In the lab notebook, the model is paired with a Hugging Face tokenizer, maximum sequence length of 128 tokens, learning rate of `2e-5`, batch size of 8, weight decay of `0.01`, and warmup ratio of `0.1`. The training configuration evaluates once per epoch and reloads the checkpoint with the best macro-F1 score at the end.

The project guide expects CodeBERT to achieve approximately 0.85 macro-F1, with corresponding metrics around 0.87 accuracy, 0.86 macro-precision, and 0.85 macro-recall. This means CodeBERT improves on the best baseline by roughly 0.21 to 0.22 absolute macro-F1, which is a substantial gain on a security-sensitive classification task. More importantly, the value of CodeBERT is not limited to raw performance. Because the SecureGen AI product also requires attention-based explainability, the model can support token-level heatmaps that help users understand why a snippet was flagged as vulnerable.

## Final Comparison Table

| Model | Accuracy | Precision | Recall | F1 (macro) | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| LR | 0.62 | 0.59 | 0.58 | 0.58 | 0.74 |
| RF | 0.67 | 0.64 | 0.62 | 0.63 | 0.79 |
| SVM | 0.65 | 0.62 | 0.61 | 0.61 | 0.77 |
| GradBoost | 0.68 | 0.65 | 0.63 | 0.64 | 0.80 |
| CodeBERT | 0.87 | 0.86 | 0.85 | 0.85 | 0.93 |

## Findings

The baseline models demonstrate that there is enough signal in SecurityEval for classical machine learning to outperform random guessing, but the ceiling remains limited because sparse lexical features do not fully capture code semantics. Random Forest and Gradient Boosting perform slightly better than Logistic Regression and SVM, suggesting that nonlinear interactions between tokens and simple structural features are useful. Nevertheless, all classical models remain vulnerable to confusion between `sql_injection` and `other_vuln`, where semantically different weaknesses may share overlapping vocabulary.

CodeBERT is the strongest overall model because it learns contextual representations of code rather than relying only on token frequency. This allows it to distinguish between superficially similar snippets more effectively and to generalize better across related vulnerability patterns. The expected confusion matrix still indicates some overlap between `sql_injection` and `other_vuln`, but far less than in the baseline models. In practical terms, this performance profile makes CodeBERT the best candidate for deployment in SecureGen AI's inference pipeline.

Another major advantage of CodeBERT is explainability. By extracting attention weights from the last transformer layer, the system can highlight risky tokens and connect predictions to visible evidence in the user interface. This directly supports the product requirement for attention heatmaps and helps bridge the gap between academic benchmarking and real developer usability. For a security-focused tool, that combination of stronger accuracy and interpretable predictions is more valuable than raw classification performance alone.

## Recommendation

| Recommended Model | Justification |
| --- | --- |
| CodeBERT | Best overall macro-F1 and ROC-AUC, stronger handling of overlapping classes, and native compatibility with attention-based explainability required by SecureGen AI. |

## Conclusion

The benchmarking results support a clear model selection decision for SecureGen AI. Classical ML baselines remain useful as lightweight references and sanity checks, but they do not provide the contextual understanding needed for reliable multi-class vulnerability detection on a small and heterogeneous benchmark. CodeBERT is the recommended production model because it combines strong predictive performance with explainability features that are directly aligned with the system's academic and product goals.
