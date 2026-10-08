import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, json
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

rng = np.random.default_rng(42)
# ---- 1. Simulate retail customer dataset with 3 latent segments ----
n = 600
seg = rng.choice([0,1,2], size=n, p=[0.40,0.35,0.25])
age = np.where(seg==0, rng.normal(26,4,n), np.where(seg==1, rng.normal(42,6,n), rng.normal(55,7,n)))
income = np.where(seg==0, rng.normal(38,8,n), np.where(seg==1, rng.normal(78,12,n), rng.normal(55,10,n)))
visits = np.where(seg==0, rng.normal(9,2.5,n), np.where(seg==1, rng.normal(5,1.8,n), rng.normal(3,1.2,n)))
basket = np.where(seg==0, rng.normal(28,7,n), np.where(seg==1, rng.normal(95,18,n), rng.normal(52,12,n)))
tenure = np.clip(rng.normal(30,14,n)+ (seg==2)*18, 1, None)
satis = np.clip(rng.normal(3.4,0.8,n) + 0.012*(basket-50)/5 + 0.15*(seg==2), 1, 5)
channel = np.where(rng.random(n) < np.where(seg==0,0.75,np.where(seg==1,0.45,0.20)), "Online","In-store")
loyal = (rng.random(n) < np.where(seg==0,0.25,np.where(seg==1,0.60,0.70))).astype(int)
df = pd.DataFrame(dict(age=age.round(0), income_k=income.round(1), visits_per_month=np.clip(visits,0.5,None).round(1),
    avg_basket=np.clip(basket,5,None).round(2), tenure_months=tenure.round(0), satisfaction=satis.round(2),
    channel=channel, loyalty_member=loyal))
df.to_csv("customers.csv", index=False)
out = {}
out["shape"] = df.shape
out["describe"] = df.describe().round(2).to_dict()
out["missing"] = int(df.isna().sum().sum())

# ---- 2. Hypothesis tests ----
on = df[df.channel=="Online"].avg_basket; ins = df[df.channel=="In-store"].avg_basket
lev = stats.levene(on, ins)
t = stats.ttest_ind(on, ins, equal_var=(lev.pvalue>0.05))
pooled = np.sqrt(((len(on)-1)*on.var()+(len(ins)-1)*ins.var())/(len(on)+len(ins)-2))
d = (on.mean()-ins.mean())/pooled
out["t"] = dict(n_on=len(on), n_in=len(ins), m_on=on.mean(), m_in=ins.mean(), sd_on=on.std(), sd_in=ins.std(),
   levene_p=lev.pvalue, t=t.statistic, p=t.pvalue, d=d)
sh = stats.shapiro(df.avg_basket.sample(300, random_state=1))
out["shapiro"] = (sh.statistic, sh.pvalue)
mw = stats.mannwhitneyu(on, ins)
out["mw"] = (mw.statistic, mw.pvalue)

# ANOVA income group vs satisfaction (tertiles)
df["income_band"] = pd.qcut(df.income_k, 3, labels=["Low","Mid","High"])
groups = [g.satisfaction.values for _,g in df.groupby("income_band", observed=True)]
an = stats.f_oneway(*groups)
kw = stats.kruskal(*groups)
ss_b = sum(len(g)*(g.mean()-df.satisfaction.mean())**2 for g in groups); ss_t = ((df.satisfaction-df.satisfaction.mean())**2).sum()
out["anova"] = dict(F=an.statistic, p=an.pvalue, eta2=ss_b/ss_t, kw_p=kw.pvalue,
    means=df.groupby("income_band", observed=True).satisfaction.mean().round(3).to_dict())
tk = stats.tukey_hsd(*groups)
out["tukey"] = str(tk)

# Chi-square channel x loyalty
ct = pd.crosstab(df.channel, df.loyalty_member)
chi2, p, dof, exp = stats.chi2_contingency(ct)
cramers = np.sqrt(chi2/(ct.values.sum()*(min(ct.shape)-1)))
out["chi"] = dict(table=ct.to_dict(), chi2=chi2, p=p, dof=dof, v=cramers)

# ---- 3. Correlation ----
num = ["age","income_k","visits_per_month","avg_basket","tenure_months","satisfaction"]
pear = df[num].corr(); spear = df[num].corr(method="spearman")
out["pearson"] = pear.round(3).to_dict()
r, pr = stats.pearsonr(df.income_k, df.avg_basket); rs, ps = stats.spearmanr(df.visits_per_month, df.age)
out["r_inc_basket"]=(r,pr); out["rs_visits_age"]=(rs,ps)
r2,p2 = stats.pearsonr(df.satisfaction, df.avg_basket); out["r_sat_basket"]=(r2,p2)
# CI via Fisher
z=np.arctanh(r); se=1/np.sqrt(len(df)-3); out["ci_inc_basket"]=(np.tanh(z-1.96*se), np.tanh(z+1.96*se))
plt.figure(figsize=(7,5.5)); sns.heatmap(pear, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True)
plt.title("Pearson Correlation Matrix"); plt.tight_layout(); plt.savefig("fig_corr.png", dpi=160); plt.close()

fig,ax = plt.subplots(1,2, figsize=(10,4))
sns.boxplot(data=df, x="channel", y="avg_basket", ax=ax[0], palette="Set2"); ax[0].set_title("Basket Value by Channel")
sns.scatterplot(data=df, x="income_k", y="avg_basket", alpha=.5, ax=ax[1]); ax[1].set_title(f"Income vs Basket (r={r:.2f})")
plt.tight_layout(); plt.savefig("fig_tests.png", dpi=160); plt.close()

# ---- 4. Clustering ----
feat = ["age","income_k","visits_per_month","avg_basket","tenure_months"]
X = StandardScaler().fit_transform(df[feat])
ks = range(2,9); inertia=[]; sil=[]; db=[]
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertia.append(km.inertia_); sil.append(silhouette_score(X, km.labels_)); db.append(davies_bouldin_score(X, km.labels_))
out["ks"]=list(ks); out["inertia"]=inertia; out["sil"]=sil; out["db"]=db
fig,ax=plt.subplots(1,2, figsize=(10,4))
ax[0].plot(list(ks), inertia, "o-"); ax[0].set_title("Elbow Method"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia")
ax[1].plot(list(ks), sil, "o-", color="green"); ax[1].set_title("Silhouette Score"); ax[1].set_xlabel("k")
plt.tight_layout(); plt.savefig("fig_elbow.png", dpi=160); plt.close()

best_k = 3
km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X); df["cluster"]=km.labels_
agg = AgglomerativeClustering(n_clusters=3).fit(X)
from sklearn.metrics import adjusted_rand_score
out["ari_km_agg"]=adjusted_rand_score(km.labels_, agg.labels_)
out["ari_true"]=adjusted_rand_score(seg, km.labels_)
out["sil_agg"]=silhouette_score(X, agg.labels_)
prof = df.groupby("cluster")[feat+["satisfaction","loyalty_member"]].mean().round(2)
prof["size"]=df.cluster.value_counts().sort_index(); prof["share_online"]=df.groupby("cluster").channel.apply(lambda s:(s=="Online").mean()).round(2)
out["profile"]=prof.to_dict(orient="index")
fa = stats.f_oneway(*[g.avg_basket.values for _,g in df.groupby("cluster")]); out["anova_cluster_basket"]=(fa.statistic, fa.pvalue)
pca = PCA(n_components=2).fit(X); Z = pca.transform(X); out["pca_var"]=pca.explained_variance_ratio_.tolist()
plt.figure(figsize=(7,5)); sc=plt.scatter(Z[:,0],Z[:,1], c=df.cluster, cmap="viridis", alpha=.7, s=18)
cz = pca.transform(km.cluster_centers_); plt.scatter(cz[:,0],cz[:,1], c="red", marker="X", s=200, edgecolor="k", label="Centroids")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})"); plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.title("K-Means Clusters in PCA Space"); plt.legend(); plt.tight_layout(); plt.savefig("fig_clusters.png", dpi=160); plt.close()
json.dump(out, open("results.json","w"), indent=1, default=lambda o: float(o) if hasattr(o,'__float__') else str(o))
print(json.dumps(out, indent=1, default=lambda o: float(o) if hasattr(o,'__float__') else str(o)))
