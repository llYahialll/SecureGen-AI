# %% [markdown]
# # SecureGen AI - Lab 2 Exploratory Data Analysis
#
# This notebook follows the SecureGen AI prompt guide for Lab 2.
# It uses the public Hugging Face `s2e-lab/SecurityEval` release for
# reproducible code execution and overlays the benchmark statistics cited
# in the project guide where Copilot insecurity rates are discussed.

# %%
from collections import OrderedDict

from datasets import load_dataset
from IPython.display import Markdown, display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

pd.set_option("display.max_colwidth", 120)
sns.set_theme(style="whitegrid", context="talk")

PALETTE = {
    "sql_injection": "#C0392B",
    "hardcoded_secret": "#E67E22",
    "weak_crypto": "#8E44AD",
    "other_vuln": "#2E86AB",
}

GUIDE_STATS = {
    "official_samples": 130,
    "official_cwe_types": 75,
    "overall_copilot_insecure_rate": 84.6,
    "sql_injection_copilot_insecure_rate": 92.0,
    "weak_crypto_copilot_insecure_rate": 100.0,
    "pearce_2022_insecure_rate": 40.0,
}

SQLI_CWES = {"CWE-089", "CWE-090", "CWE-643"}
SECRET_CWES = {"CWE-798", "CWE-259", "CWE-321", "CWE-312"}
CRYPTO_CWES = {"CWE-327", "CWE-328", "CWE-326", "CWE-916", "CWE-477"}

CATEGORY_ORDER = [
    "sql_injection",
    "hardcoded_secret",
    "weak_crypto",
    "other_vuln",
]


def map_category(cwe: str) -> str:
    if cwe in SQLI_CWES:
        return "sql_injection"
    if cwe in SECRET_CWES:
        return "hardcoded_secret"
    if cwe in CRYPTO_CWES:
        return "weak_crypto"
    return "other_vuln"


def normalize_source(raw_source: str) -> str:
    source_map = {
        "codeql": "CodeQL",
        "sonar": "SonarRules",
        "pearce": "Pearce",
        "mitre": "CWE list",
        "author": "Author-created",
    }
    return source_map.get(str(raw_source).lower(), str(raw_source).title())


def load_securityeval_frame() -> pd.DataFrame:
    dataset = load_dataset("s2e-lab/SecurityEval")
    df = pd.DataFrame(dataset["train"]).rename(
        columns={
            "ID": "id",
            "Prompt": "prompt",
            "Insecure_code": "insecure_code",
        }
    )
    df["cwe"] = df["id"].str.extract(r"(CWE-\d+)")
    df["source_raw"] = df["id"].str.extract(r"CWE-\d+_([^_]+)_")
    df["source"] = df["source_raw"].map(normalize_source)
    df["category"] = df["cwe"].map(map_category)
    df["num_tokens"] = df["insecure_code"].str.split().str.len()
    df["num_lines"] = df["insecure_code"].str.count("\n") + 1

    # The public split stores insecure examples only, so this helper flag is
    # useful for notebook consistency but is not the same as the guide-level
    # Copilot benchmark rate statistics.
    df["copilot_insecure"] = 1
    return df


def build_benchmark_rate_frame(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df["category"]
        .value_counts()
        .reindex(CATEGORY_ORDER)
        .fillna(0)
        .astype(int)
        .rename("total_count")
    )

    # Only two category-specific guide rates are explicitly provided in the
    # prompt guide. The remaining categories use the guide's overall benchmark
    # rate so the required plots remain complete and clearly labeled.
    rates = pd.Series(
        OrderedDict(
            [
                ("sql_injection", GUIDE_STATS["sql_injection_copilot_insecure_rate"]),
                ("hardcoded_secret", GUIDE_STATS["overall_copilot_insecure_rate"]),
                ("weak_crypto", GUIDE_STATS["weak_crypto_copilot_insecure_rate"]),
                ("other_vuln", GUIDE_STATS["overall_copilot_insecure_rate"]),
            ]
        ),
        name="insecure_rate_pct",
    )

    benchmark = pd.concat([counts, rates], axis=1)
    benchmark["insecure_count"] = (
        benchmark["total_count"] * benchmark["insecure_rate_pct"] / 100.0
    ).round().astype(int)
    benchmark.index.name = "category"
    return benchmark.reset_index()


df = load_securityeval_frame()
benchmark_df = build_benchmark_rate_frame(df)

# %% [markdown]
# ## Cell 2 - Load SecurityEval
#
# This cell loads the dataset and prints the shape, columns, and first three rows.

# %%
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
display(df.head(3))

display(
    Markdown(
        f"""
**Notebook note.** The current public Hugging Face release loads **{len(df)} rows**
with the fields `id`, `prompt`, and `insecure_code`. The SecureGen AI project guide
and the original SecurityEval paper cite the broader benchmark as **130 prompts**
covering **75 CWE types**. The code below stays runnable on the public split while the
written report retains the guide's official benchmark figures.
"""
    )
)

# %% [markdown]
# ## Cell 3 - Map CWE IDs to Project Categories
#
# This cell maps each CWE to one of the four SecureGen AI classes and prints
# category counts plus the helper `copilot_insecure` mean.

# %%
category_counts = (
    df["category"].value_counts().reindex(CATEGORY_ORDER).fillna(0).astype(int)
)
print("Category counts:")
print(category_counts)
print()
print("copilot_insecure mean (public split helper flag):", df["copilot_insecure"].mean())

display(
    benchmark_df[["category", "total_count", "insecure_rate_pct"]]
    .rename(
        columns={
            "total_count": "split_count",
            "insecure_rate_pct": "guide_benchmark_insecure_rate_pct",
        }
    )
)

# %% [markdown]
# ## Cell 4 - Feature Engineering
#
# This cell computes `num_tokens` and `num_lines`, then summarizes both features
# by vulnerability category.

# %%
feature_stats = (
    df.groupby("category")[["num_tokens", "num_lines"]]
    .agg(["mean", "median", "std", "min", "max"])
    .round(2)
    .reindex(CATEGORY_ORDER)
)

print("Descriptive statistics for engineered features by category:")
display(feature_stats)

# %% [markdown]
# ## Plot 1 - Category Distribution and Copilot Insecure Rate
#
# The donut chart uses the mapped category distribution from the public
# SecurityEval split, while the horizontal bar chart uses the benchmark
# insecurity rates referenced in the SecureGen AI project guide.

# %%
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

category_share = (
    df["category"].value_counts().reindex(CATEGORY_ORDER).fillna(0).astype(int)
)
colors = [PALETTE[c] for c in CATEGORY_ORDER]

wedges, texts, autotexts = axes[0].pie(
    category_share.values,
    labels=[c.replace("_", " ").title() for c in CATEGORY_ORDER],
    colors=colors,
    startangle=90,
    autopct=lambda pct: f"{pct:.1f}%",
    pctdistance=0.82,
    wedgeprops={"width": 0.38, "edgecolor": "white"},
)
axes[0].text(
    0,
    0,
    f"{len(df)}\nexamples",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
)
axes[0].set_title("Figure 1A. Category Share in the Public SecurityEval Split")

rate_plot = benchmark_df.sort_values("insecure_rate_pct", ascending=True)
axes[1].barh(
    rate_plot["category"].str.replace("_", " ").str.title(),
    rate_plot["insecure_rate_pct"],
    color=rate_plot["category"].map(PALETTE),
)
axes[1].set_xlim(0, 110)
axes[1].set_xlabel("Copilot Insecure Rate (%)")
axes[1].set_ylabel("Category")
axes[1].set_title("Figure 1B. Guide-Level Copilot Insecure Rates by Category")
for idx, value in enumerate(rate_plot["insecure_rate_pct"]):
    axes[1].text(value + 1, idx, f"{value:.1f}%", va="center", fontsize=12)

fig.suptitle(
    "SecurityEval Category Composition and Copilot Insecurity Overview\n"
    "Guide key figures: overall 84.6%, weak_crypto 100%, sql_injection 92%",
    fontsize=18,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Plot 2 - Top 15 CWEs
#
# This plot shows the most frequent CWEs in the public split and colors each
# bar by the SecureGen AI four-class category mapping.

# %%
cwe_summary = (
    df.groupby(["cwe", "category"])
    .size()
    .reset_index(name="count")
    .sort_values(["count", "cwe"], ascending=[False, True])
    .head(15)
)

plt.figure(figsize=(16, 9))
ax = sns.barplot(
    data=cwe_summary,
    x="count",
    y="cwe",
    hue="category",
    dodge=False,
    palette=PALETTE,
)
ax.set_title("Figure 2. Top 15 CWE Types in the Public SecurityEval Split")
ax.set_xlabel("Number of Samples")
ax.set_ylabel("CWE")
for patch in ax.patches:
    width = patch.get_width()
    y = patch.get_y() + patch.get_height() / 2
    ax.text(width + 0.05, y, f"{int(width)}", va="center", fontsize=11)
ax.legend(title="Mapped Category", loc="lower right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Plot 3 - Source Distribution per Category
#
# This grouped bar chart compares how the mapped categories are distributed
# across the original example sources.

# %%
source_order = ["CodeQL", "SonarRules", "Pearce", "CWE list", "Author-created"]
source_dist = (
    df.groupby(["category", "source"])
    .size()
    .reset_index(name="count")
    .assign(
        category=lambda frame: pd.Categorical(
            frame["category"], categories=CATEGORY_ORDER, ordered=True
        ),
        source=lambda frame: pd.Categorical(
            frame["source"], categories=source_order, ordered=True
        ),
    )
    .sort_values(["category", "source"])
)

plt.figure(figsize=(16, 9))
ax = sns.barplot(
    data=source_dist,
    x="category",
    y="count",
    hue="source",
    palette="Set2",
)
ax.set_title("Figure 3. Source Distribution per Vulnerability Category")
ax.set_xlabel("Category")
ax.set_ylabel("Count")
ax.set_xticklabels([label.replace("_", " ").title() for label in CATEGORY_ORDER], rotation=0)
ax.legend(title="Source", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Plot 4 - Token Length Boxplot
#
# Token-length distributions can reveal whether some vulnerability classes tend
# to appear in longer or shorter code snippets.

# %%
plt.figure(figsize=(14, 8))
ax = sns.boxplot(
    data=df,
    x="category",
    y="num_tokens",
    order=CATEGORY_ORDER,
    palette=PALETTE,
    medianprops={"color": "white", "linewidth": 2},
)
ax.set_title("Figure 4. Token Length Distribution by Vulnerability Category")
ax.set_xlabel("Category")
ax.set_ylabel("Number of Tokens")
ax.set_xticklabels([label.replace("_", " ").title() for label in CATEGORY_ORDER], rotation=0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Plot 5 - Copilot Security Heatmap
#
# The public split does not include secure Copilot completions, so this heatmap
# overlays the guide's benchmark insecurity rates on top of the mapped split totals.

# %%
heatmap_df = benchmark_df.set_index("category").reindex(CATEGORY_ORDER)
heatmap_matrix = pd.DataFrame(
    {
        category: [
            heatmap_df.loc[category, "insecure_count"],
            heatmap_df.loc[category, "total_count"],
            heatmap_df.loc[category, "insecure_rate_pct"],
        ]
        for category in CATEGORY_ORDER
    },
    index=["Insecure count", "Total count", "Insecure rate (%)"],
)

plt.figure(figsize=(14, 5))
ax = sns.heatmap(
    heatmap_matrix,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
    linewidths=0.5,
    cbar_kws={"label": "Value"},
)
ax.set_title("Figure 5. Copilot Security Heatmap (Guide Benchmark Overlay)")
ax.set_xlabel("Category")
ax.set_ylabel("Metric")
ax.set_xticklabels([label.replace("_", " ").title() for label in CATEGORY_ORDER], rotation=0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Plot 6 - PCA of TF-IDF Features
#
# This plot projects TF-IDF features from the insecure code snippets into two
# dimensions to visualize lexical separation and overlap between classes.

# %%
vectorizer = TfidfVectorizer(max_features=300)
X_tfidf = vectorizer.fit_transform(df["insecure_code"])
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_tfidf.toarray())

pca_df = pd.DataFrame(
    {
        "pc1": X_pca[:, 0],
        "pc2": X_pca[:, 1],
        "category": df["category"].values,
    }
)

plt.figure(figsize=(14, 9))
ax = sns.scatterplot(
    data=pca_df,
    x="pc1",
    y="pc2",
    hue="category",
    palette=PALETTE,
    s=110,
    alpha=0.85,
)
ax.set_title("Figure 6. PCA Projection of TF-IDF Features")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)")
ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Optional Summary Tables for the Written Report
#
# These helper tables make it easier to transfer results into the Lab 2 report.

# %%
summary_table = (
    benchmark_df[["category", "total_count", "insecure_rate_pct"]]
    .assign(
        share_pct=lambda frame: (frame["total_count"] / frame["total_count"].sum() * 100).round(2)
    )
    .rename(
        columns={
            "category": "Category",
            "total_count": "Count",
            "share_pct": "Share (%)",
            "insecure_rate_pct": "Copilot Insecure Rate (%)",
        }
    )
)

display(summary_table)

display(
    Markdown(
        f"""
**Academic note for the report.**

- Cite Siddiq and Santos (MSR4P&S 2022) for the official benchmark definition:
  **130 prompts**, **75 CWE types**, Python-only, and four source families
  (CodeQL, Sonar Rules, Pearce et al., and the CWE list), with additional
  author-created examples mentioned in the paper.
- Cite Pearce et al. (2022) when discussing prior evidence that roughly
  **40%** of Copilot-generated programs were vulnerable.
- Use the SecureGen AI guide's benchmark figures in the narrative:
  **84.6%** overall insecure Copilot outputs and **100%** for `weak_crypto`.
"""
    )
)
