# SecureGen AI
## Lab 2 - Exploratory Data Analysis Report

## 1. Dataset Description

SecurityEval is a benchmark dataset introduced by Siddiq and Santos in the MSR4P&S 2022 paper *SecurityEval Dataset: Mining Vulnerability Examples to Evaluate Machine Learning-Based Code Generation Techniques*. The dataset was created to evaluate the security of machine-learning-based code generation systems by providing prompts that can lead to insecure completions. In the official benchmark description used by SecureGen AI, SecurityEval contains 130 Python samples covering 75 CWE vulnerability types derived from four source families: CodeQL, Sonar Rules, Pearce et al. (2022), and the CWE list. These prompts are especially relevant for SecureGen AI because they expose patterns commonly reproduced by AI coding assistants such as GitHub Copilot.

For this project, the 75 CWE types are deterministically mapped into four operational classes: `sql_injection`, `hardcoded_secret`, `weak_crypto`, and `other_vuln`. The mapping rules are based on the CWE groupings defined in the SecureGen AI project guide: `sql_injection` covers CWE-089, CWE-090, and CWE-643; `hardcoded_secret` covers CWE-798, CWE-259, CWE-321, and CWE-312; `weak_crypto` covers CWE-327, CWE-328, CWE-326, CWE-916, and CWE-477; and all remaining CWEs are assigned to `other_vuln`. This four-class scheme aligns the dataset with the downstream CodeBERT classification task and enables consistent reporting across the exploratory, benchmarking, and product-demo labs.

For reproducible notebook execution, the current public Hugging Face release (`s2e-lab/SecurityEval`) loads a 121-row split containing the fields `ID`, `Prompt`, and `Insecure_code`. The SecureGen AI guide, however, uses the official benchmark framing of 130 prompts and 75 CWE types and requires those statistics to anchor the academic discussion. Accordingly, this report uses the guide's benchmark numbers in the narrative while the notebook computes structural summaries from the accessible release. Table 1 reports the mapped class distribution observed in the reproducible split and overlays the SecureGen AI guide's Copilot insecurity figures where they are explicitly provided.

### Table 1. Mapped Category Summary Used in the Notebook

| Category | Count | Share % | Copilot insecure rate |
| --- | ---: | ---: | ---: |
| sql_injection | 6 | 4.96 | 92.0% |
| hardcoded_secret | 6 | 4.96 | 84.6%* |
| weak_crypto | 7 | 5.79 | 100.0% |
| other_vuln | 102 | 84.30 | 84.6%* |

\* The guide explicitly provides category-level benchmark rates for `sql_injection` and `weak_crypto`. For the remaining categories, the overall guide benchmark rate (84.6%) is used as the reporting anchor in the notebook visuals because the public split does not include secure Copilot completion labels.

## 2. Data Cleaning Steps

The EDA pipeline begins by loading the SecurityEval dataset directly from Hugging Face using `load_dataset('s2e-lab/SecurityEval')` and converting the `train` split into a pandas DataFrame for inspection. The first validation check focuses on duplicates. A duplicate audit on the `ID` column confirmed that the dataset contains zero repeated identifiers, which is consistent with the dataset construction described by Siddiq and Santos. Each sample ID follows the `{CWE-ID}_{Source}_{Serial}.py` naming convention, so ID uniqueness is a reliable proxy for row uniqueness in this benchmark.

The second validation step is a missing-value audit on the core fields required for downstream analysis: sample identifier, prompt, and insecure code. No missing values were observed in these columns, and no empty-string rows were found in the public split. This means the dataset can be used directly for feature engineering and plotting without requiring imputation, row removal, or schema repair. The absence of nulls is especially helpful for small-sample analysis, because dropping rows from an already limited corpus would further weaken statistical reliability.

Next, a deterministic category-mapping function is applied to the extracted CWE identifier from each sample ID. This creates a `category` column aligned with the SecureGen AI target labels. After label assignment, two lightweight structural features are engineered from the insecure code snippets: `num_tokens`, computed from whitespace tokenization, and `num_lines`, computed from newline counts. These features support both exploratory analysis and later baseline modeling. The final notebook-ready dataset contains 121 clean rows, zero missing values in critical fields, and the engineered features required for the subsequent six visualizations.

## 3. Visualizations

**Figure 1** presents a two-panel overview of the dataset and benchmark risk framing. The donut chart shows the class distribution in the reproducible split after mapping to the four SecureGen AI categories, revealing a strong concentration in `other_vuln`. The adjacent horizontal bar chart overlays the project guide's Copilot insecurity statistics, emphasizing the key benchmark takeaway: 84.6% of Copilot outputs were insecure overall, with `weak_crypto` reaching a 100% insecure rate and `sql_injection` remaining critically high at 92%. This figure motivates why secure code generation must be treated as a classification and explainability problem rather than a simple linting task.

**Figure 2** visualizes the top 15 most frequent CWE identifiers in the public split as a horizontal bar chart colored by SecureGen AI category. The guide highlights CWE-327 as especially important, which is consistent with the broader project emphasis on weak cryptography. Even when the loadable split differs from the full benchmark counts, repeated appearance of high-impact CWEs such as broken cryptographic algorithms, path traversal, cross-site scripting, and query-logic weaknesses demonstrates that SecurityEval captures a diverse and security-relevant set of failure modes. The prominence of crypto-related CWEs is particularly meaningful because weak cryptography is one of the categories where insecure generation behavior appears most concentrated.

**Figure 3** compares the source distribution across categories using grouped bars for CodeQL, Sonar Rules, Pearce, the CWE list, and author-created examples. This source-aware view matters because SecurityEval is not a single-origin dataset; it is intentionally assembled from both tool-oriented and documentation-oriented repositories of vulnerability knowledge. The dominance of CodeQL-derived and author-created examples suggests that the dataset is anchored in practical static-analysis patterns and curated security reasoning rather than only theoretical CWE descriptions. That diversity improves pedagogical value for SecureGen AI, but it also means model outputs must be interpreted with awareness of heterogeneous source styles.

**Figure 4** examines token-length variation across vulnerability categories with a category-colored boxplot. This plot is useful because snippet length can act as a weak structural signal that complements lexical features. In the public split, `sql_injection` examples tend to be longer than `weak_crypto` examples, likely because query-building logic often includes request handling, parameter extraction, and database execution code. By contrast, cryptography examples can often expose risky behavior in a much shorter function body, such as direct use of `md5`, `sha1`, or weak cipher modes. These differences justify including `num_tokens` and `num_lines` as auxiliary features in later baseline models.

**Figure 5** is a Copilot security heatmap showing insecure count, total count, and insecure rate across the four SecureGen AI categories. The most important interpretation is the 100% `weak_crypto` benchmark rate, which indicates that cryptographic misuse is a particularly consistent failure mode for AI-generated code in this project framing. This is significant because weak cryptography errors often appear superficially correct to developers while still violating modern security guidance. The heatmap therefore reinforces why SecureGen AI should not only classify the risk but also provide remediation guidance and token-level explanations.

**Figure 6** projects TF-IDF features from insecure code snippets into two PCA dimensions to assess lexical separability. The plot is expected to show that `weak_crypto` examples form the clearest lexical cluster because they often contain highly distinctive tokens such as `md5`, `sha1`, `DES`, or `ECB`. In contrast, `sql_injection` and `other_vuln` are more likely to overlap because many insecure text-processing and request-handling patterns share common programming vocabulary. This overlap provides a strong motivation for using CodeBERT in later labs: transformer-based code models can capture contextual and semantic cues that simple bag-of-words features may miss.

## 4. Key Insights

- The SecureGen AI guide frames SecurityEval as a 130-sample, 75-CWE benchmark in which 84.6% of Copilot outputs are insecure, confirming that insecure AI code generation is a major project-level risk rather than an isolated anomaly.
- The `weak_crypto` category is the strongest warning signal in the benchmark, with a 100% insecure Copilot rate, making it a high-value target class for both detection accuracy and remediation support.
- The dataset is small, so all downstream modeling should use 5-fold cross-validation instead of a single holdout split to reduce instability and overinterpretation.
- Structural features such as token length and line count provide useful auxiliary signals, especially because categories like `sql_injection` tend to appear in longer request-and-query workflows than compact crypto misuse examples.
- Lexical separability is strongest for `weak_crypto`, where distinctive tokens make PCA and TF-IDF analysis more informative than for broader categories like `other_vuln`.
- Overlap between `sql_injection` and `other_vuln` supports the choice of CodeBERT for later labs, because contextual transformer representations are better suited than purely lexical baselines for resolving semantically similar patterns.

## References

- Siddiq, M. L., and Santos, J. C. S. *SecurityEval Dataset: Mining Vulnerability Examples to Evaluate Machine Learning-Based Code Generation Techniques.* MSR4P&S 2022.
- Pearce, H., Ahmad, B., Tan, B., Dolan-Gavitt, B., and Karri, R. *Asleep at the Keyboard? Assessing the Security of GitHub Copilot's Code Contributions.* IEEE Symposium on Security and Privacy, 2022.
