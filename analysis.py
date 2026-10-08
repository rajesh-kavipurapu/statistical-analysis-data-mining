import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import json

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score
)


# ============================================================
# 1. DATA GENERATION
# ============================================================

rng = np.random.default_rng(42)

# Number of customers
n = 600

# Three hidden customer segments
seg = rng.choice(
    [0, 1, 2],
    size=n,
    p=[0.40, 0.35, 0.25]
)

# Customer age
age = np.where(
    seg == 0,
    rng.normal(26, 4, n),
    np.where(
        seg == 1,
        rng.normal(42, 6, n),
        rng.normal(55, 7, n)
    )
)

# Annual income in thousands
income = np.where(
    seg == 0,
    rng.normal(38, 8, n),
    np.where(
        seg == 1,
        rng.normal(78, 12, n),
        rng.normal(55, 10, n)
    )
)

# Monthly visits
visits = np.where(
    seg == 0,
    rng.normal(9, 2.5, n),
    np.where(
        seg == 1,
        rng.normal(5, 1.8, n),
        rng.normal(3, 1.2, n)
    )
)

# Average basket value
basket = np.where(
    seg == 0,
    rng.normal(28, 7, n),
    np.where(
        seg == 1,
        rng.normal(95, 18, n),
        rng.normal(52, 12, n)
    )
)

# Customer tenure
tenure = np.clip(
    rng.normal(30, 14, n) + (seg == 2) * 18,
    1,
    None
)

# Satisfaction score
satis = np.clip(
    rng.normal(3.4, 0.8, n)
    + 0.012 * (basket - 50) / 5
    + 0.15 * (seg == 2),
    1,
    5
)

# Shopping channel
channel = np.where(
    rng.random(n)
    < np.where(
        seg == 0,
        0.75,
        np.where(seg == 1, 0.45, 0.20)
    ),
    "Online",
    "In-store"
)

# Loyalty membership
loyal = (
    rng.random(n)
    < np.where(
        seg == 0,
        0.25,
        np.where(seg == 1, 0.60, 0.70)
    )
).astype(int)


# Create DataFrame
df = pd.DataFrame({
    "age": age.round(0),
    "income_k": income.round(1),
    "visits_per_month": np.clip(visits, 0.5, None).round(1),
    "avg_basket": np.clip(basket, 5, None).round(2),
    "tenure_months": tenure.round(0),
    "satisfaction": satis.round(2),
    "channel": channel,
    "loyalty_member": loyal
})


# Save generated dataset
df.to_csv("customers.csv", index=False)


# Dictionary for storing results
out = {}

out["shape"] = df.shape
out["describe"] = df.describe().round(2).to_dict()
out["missing"] = int(df.isna().sum().sum())


# ============================================================
# 2. HYPOTHESIS TESTING
# ============================================================

# ------------------------------------------------------------
# Hypothesis 1:
# Online vs In-store average basket value
# ------------------------------------------------------------

online = df.loc[
    df["channel"] == "Online",
    "avg_basket"
]

instore = df.loc[
    df["channel"] == "In-store",
    "avg_basket"
]


# Levene's test for equality of variances
levene_test = stats.levene(
    online,
    instore
)


# Independent samples t-test
t_test = stats.ttest_ind(
    online,
    instore,
    equal_var=(levene_test.pvalue > 0.05)
)


# Cohen's d
pooled_sd = np.sqrt(
    (
        (len(online) - 1) * online.var()
        + (len(instore) - 1) * instore.var()
    )
    / (len(online) + len(instore) - 2)
)

cohens_d = (
    online.mean() - instore.mean()
) / pooled_sd


out["t_test"] = {
    "n_online": len(online),
    "n_instore": len(instore),
    "mean_online": online.mean(),
    "mean_instore": instore.mean(),
    "sd_online": online.std(),
    "sd_instore": instore.std(),
    "levene_p": levene_test.pvalue,
    "t_statistic": t_test.statistic,
    "p_value": t_test.pvalue,
    "cohens_d": cohens_d
}


# Shapiro-Wilk normality tests
# Shapiro supports samples up to 5000 observations.
shapiro_online = stats.shapiro(online)
shapiro_instore = stats.shapiro(instore)

out["shapiro"] = {
    "online": {
        "statistic": shapiro_online.statistic,
        "p_value": shapiro_online.pvalue
    },
    "in_store": {
        "statistic": shapiro_instore.statistic,
        "p_value": shapiro_instore.pvalue
    }
}


# Mann-Whitney U test
mann_whitney = stats.mannwhitneyu(
    online,
    instore,
    alternative="two-sided"
)

out["mann_whitney"] = {
    "statistic": mann_whitney.statistic,
    "p_value": mann_whitney.pvalue
}


# ============================================================
# Hypothesis 2:
# Satisfaction by Income Band
# ============================================================

# Divide income into three groups
df["income_band"] = pd.qcut(
    df["income_k"],
    3,
    labels=["Low", "Mid", "High"]
)


# Create groups
groups = [
    group["satisfaction"].values
    for _, group in df.groupby(
        "income_band",
        observed=True
    )
]


# One-way ANOVA
anova_test = stats.f_oneway(*groups)


# Kruskal-Wallis test
kruskal_test = stats.kruskal(*groups)


# Eta-squared effect size
overall_mean = df["satisfaction"].mean()

ss_between = sum(
    len(group)
    * (group.mean() - overall_mean) ** 2
    for group in groups
)

ss_total = (
    (df["satisfaction"] - overall_mean) ** 2
).sum()

eta_squared = ss_between / ss_total


# Group means
income_means = (
    df.groupby(
        "income_band",
        observed=True
    )["satisfaction"]
    .mean()
    .round(3)
    .to_dict()
)


# Tukey HSD post-hoc test
tukey_test = stats.tukey_hsd(*groups)


out["anova"] = {
    "F_statistic": anova_test.statistic,
    "p_value": anova_test.pvalue,
    "eta_squared": eta_squared,
    "kruskal_p": kruskal_test.pvalue,
    "means": income_means
}

out["tukey"] = str(tukey_test)


# ============================================================
# Hypothesis 3:
# Channel and Loyalty Membership
# ============================================================

# Contingency table
contingency_table = pd.crosstab(
    df["channel"],
    df["loyalty_member"]
)


# Chi-square test
chi2, chi_p, degrees_freedom, expected = (
    stats.chi2_contingency(contingency_table)
)


# Cramer's V
cramers_v = np.sqrt(
    chi2
    / (
        contingency_table.values.sum()
        * (min(contingency_table.shape) - 1)
    )
)


out["chi_square"] = {
    "table": contingency_table.to_dict(),
    "chi2": chi2,
    "p_value": chi_p,
    "degrees_of_freedom": degrees_freedom,
    "cramers_v": cramers_v
}


# ============================================================
# 3. CORRELATION ANALYSIS
# ============================================================

numeric_columns = [
    "age",
    "income_k",
    "visits_per_month",
    "avg_basket",
    "tenure_months",
    "satisfaction"
]


# Pearson correlation
pearson_corr = df[numeric_columns].corr()


# Spearman correlation
spearman_corr = df[numeric_columns].corr(
    method="spearman"
)


out["pearson"] = pearson_corr.round(3).to_dict()


# Income vs basket
income_basket_r, income_basket_p = stats.pearsonr(
    df["income_k"],
    df["avg_basket"]
)


# Visits vs age
visits_age_rho, visits_age_p = stats.spearmanr(
    df["visits_per_month"],
    df["age"]
)


# Satisfaction vs basket
satisfaction_basket_r, satisfaction_basket_p = (
    stats.pearsonr(
        df["satisfaction"],
        df["avg_basket"]
    )
)


out["income_basket"] = {
    "r": income_basket_r,
    "p_value": income_basket_p
}

out["visits_age"] = {
    "spearman_rho": visits_age_rho,
    "p_value": visits_age_p
}

out["satisfaction_basket"] = {
    "r": satisfaction_basket_r,
    "p_value": satisfaction_basket_p
}


# Fisher confidence interval for income vs basket
z_value = np.arctanh(income_basket_r)

standard_error = 1 / np.sqrt(
    len(df) - 3
)

ci_lower = np.tanh(
    z_value - 1.96 * standard_error
)

ci_upper = np.tanh(
    z_value + 1.96 * standard_error
)

out["income_basket_ci_95"] = [
    ci_lower,
    ci_upper
]


# ------------------------------------------------------------
# Correlation heatmap
# ------------------------------------------------------------

plt.figure(figsize=(7, 5.5))

sns.heatmap(
    pearson_corr,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    square=True
)

plt.title("Pearson Correlation Matrix")
plt.tight_layout()

plt.savefig(
    "fig_corr.png",
    dpi=160
)

plt.close()


# ------------------------------------------------------------
# Basket value by channel + Income vs basket
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 4)
)


# Fixed Seaborn syntax:
# hue is used together with palette.
sns.boxplot(
    data=df,
    x="channel",
    y="avg_basket",
    hue="channel",
    palette="Set2",
    legend=False,
    ax=axes[0]
)

axes[0].set_title(
    "Basket Value by Channel"
)


# Scatter plot
sns.scatterplot(
    data=df,
    x="income_k",
    y="avg_basket",
    alpha=0.5,
    ax=axes[1]
)

axes[1].set_title(
    f"Income vs Basket (r={income_basket_r:.2f})"
)


plt.tight_layout()

plt.savefig(
    "fig_tests.png",
    dpi=160
)

plt.close()


# ============================================================
# 4. CLUSTERING
# ============================================================

features = [
    "age",
    "income_k",
    "visits_per_month",
    "avg_basket",
    "tenure_months"
]


# Standardise features
scaler = StandardScaler()

X = scaler.fit_transform(
    df[features]
)


# ------------------------------------------------------------
# Test different numbers of clusters
# ------------------------------------------------------------

cluster_values = range(2, 9)

inertia = []
silhouette_scores = []
davies_bouldin_scores = []


for k in cluster_values:

    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = kmeans.fit_predict(X)

    inertia.append(
        kmeans.inertia_
    )

    silhouette_scores.append(
        silhouette_score(
            X,
            labels
        )
    )

    davies_bouldin_scores.append(
        davies_bouldin_score(
            X,
            labels
        )
    )


out["cluster_values"] = list(
    cluster_values
)

out["inertia"] = inertia

out["silhouette"] = silhouette_scores

out["davies_bouldin"] = (
    davies_bouldin_scores
)


# ------------------------------------------------------------
# Elbow + Silhouette charts
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 4)
)


# Elbow method
axes[0].plot(
    list(cluster_values),
    inertia,
    "o-"
)

axes[0].set_title(
    "Elbow Method"
)

axes[0].set_xlabel(
    "Number of Clusters (k)"
)

axes[0].set_ylabel(
    "Inertia"
)


# Silhouette score
axes[1].plot(
    list(cluster_values),
    silhouette_scores,
    "o-",
    color="green"
)

axes[1].set_title(
    "Silhouette Score"
)

axes[1].set_xlabel(
    "Number of Clusters (k)"
)

axes[1].set_ylabel(
    "Silhouette Score"
)


plt.tight_layout()

plt.savefig(
    "fig_elbow.png",
    dpi=160
)

plt.close()


# ============================================================
# FINAL K-MEANS MODEL
# ============================================================

# k = 3 is selected based on the elbow and business
# interpretability described in the report.
best_k = 3


kmeans_final = KMeans(
    n_clusters=best_k,
    n_init=10,
    random_state=42
)

kmeans_labels = kmeans_final.fit_predict(X)

df["cluster"] = kmeans_labels


# ============================================================
# AGGLOMERATIVE CLUSTERING
# ============================================================

agglomerative = AgglomerativeClustering(
    n_clusters=3
)

agg_labels = agglomerative.fit_predict(X)


# Compare K-Means and Agglomerative clustering
ari_kmeans_agg = adjusted_rand_score(
    kmeans_labels,
    agg_labels
)


# Compare K-Means with hidden true segments
ari_true = adjusted_rand_score(
    seg,
    kmeans_labels
)


# Silhouette score for Agglomerative clustering
silhouette_agg = silhouette_score(
    X,
    agg_labels
)


out["ari_kmeans_agglomerative"] = (
    ari_kmeans_agg
)

out["ari_true_segments"] = (
    ari_true
)

out["silhouette_agglomerative"] = (
    silhouette_agg
)


# ============================================================
# CLUSTER PROFILE
# ============================================================

profile = (
    df.groupby("cluster")[
        features
        + [
            "satisfaction",
            "loyalty_member"
        ]
    ]
    .mean()
    .round(2)
)


# Cluster sizes
profile["size"] = (
    df["cluster"]
    .value_counts()
    .sort_index()
)


# Online share
online_share = (
    df.groupby("cluster")["channel"]
    .agg(
        lambda values:
        (values == "Online").mean()
    )
    .round(2)
)

profile["share_online"] = online_share


out["profile"] = profile.to_dict(
    orient="index"
)


# ============================================================
# ANOVA: BASKET VALUE BY CLUSTER
# ============================================================

cluster_groups = [
    group["avg_basket"].values
    for _, group in df.groupby("cluster")
]


cluster_anova = stats.f_oneway(
    *cluster_groups
)


out["anova_cluster_basket"] = {
    "F_statistic": cluster_anova.statistic,
    "p_value": cluster_anova.pvalue
}


# ============================================================
# PCA VISUALISATION
# ============================================================

pca = PCA(
    n_components=2
)

Z = pca.fit_transform(X)


out["pca_variance"] = (
    pca.explained_variance_ratio_.tolist()
)


# PCA scatter plot
plt.figure(
    figsize=(7, 5)
)

plt.scatter(
    Z[:, 0],
    Z[:, 1],
    c=df["cluster"],
    cmap="viridis",
    alpha=0.7,
    s=18
)


# Transform K-Means centers into PCA space
cluster_centers_pca = pca.transform(
    kmeans_final.cluster_centers_
)


plt.scatter(
    cluster_centers_pca[:, 0],
    cluster_centers_pca[:, 1],
    c="red",
    marker="X",
    s=200,
    edgecolor="black",
    label="Centroids"
)


plt.xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0]:.1%})"
)

plt.ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"
)

plt.title(
    "K-Means Clusters in PCA Space"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "fig_clusters.png",
    dpi=160
)

plt.close()


# ============================================================
# 5. SAVE RESULTS
# ============================================================

with open(
    "results.json",
    "w"
) as result_file:

    json.dump(
        out,
        result_file,
        indent=1,
        default=lambda obj:
        float(obj)
        if hasattr(obj, "__float__")
        else str(obj)
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    json.dumps(
        out,
        indent=1,
        default=lambda obj:
        float(obj)
        if hasattr(obj, "__float__")
        else str(obj)
    )
)

print("\n========================================")
print("Analysis completed successfully!")
print("========================================")
print("Generated files:")
print("1. customers.csv")
print("2. fig_corr.png")
print("3. fig_tests.png")
print("4. fig_elbow.png")
print("5. fig_clusters.png")
print("6. results.json")
