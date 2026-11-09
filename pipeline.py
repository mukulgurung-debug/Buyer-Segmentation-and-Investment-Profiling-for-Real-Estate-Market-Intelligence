"""Buyer segmentation pipeline: clean -> engineer -> encode/scale -> cluster -> profile.
Run: python pipeline.py   (writes data/clients_segmented.csv, models/*.pkl, figures/*.png, models/metrics.json)"""
import json, warnings
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
SEED = 42

# ---------- 1. Cleaning ----------
cl = pd.read_csv("data/clients.csv")
pr = pd.read_csv("data/properties.csv")
log = {"clients_raw": len(cl), "properties_raw": len(pr)}
for c in cl.select_dtypes("object"): cl[c] = cl[c].str.strip()
cl = cl.drop_duplicates("client_id")
for c in ["client_type","country","region","acquisition_purpose","loan_applied","referral_channel","gender"]:
    cl[c] = cl[c].str.title() if c not in ("country","gender") else cl[c]
log["missing_client_cells"] = int(cl.isna().sum().sum())
log["dup_listing_ids"] = int(pr.listing_id.duplicated().sum())
pr = pr.drop_duplicates("listing_id")

def parse_dob(s):  # mixed formats: '05-11-1968' (dd-mm-yyyy) and '11/26/1962' (m/d/yyyy)
    return pd.to_datetime(s, format="%d-%m-%Y", errors="coerce") if "-" in s else pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
cl["dob"] = cl.date_of_birth.map(parse_dob)
log["unparsed_dob"] = int(cl.dob.isna().sum())
pr["sale_price"] = pr.sale_price.str.replace(r"[$,]", "", regex=True).astype(float)
pr["transaction_date"] = pd.to_datetime(pr.transaction_date, format="%d-%m-%Y", errors="coerce")
ref = pr.transaction_date.max()
cl["age"] = ((ref - cl.dob).dt.days / 365.25).round(1)
log["age_range"] = [float(cl.age.min()), float(cl.age.max())]

# ---------- 2. Feature engineering (client-level behaviour from purchases) ----------
sold = pr[(pr.listing_status == "Sold") & pr.client_ref.notna()].copy()
sold["ppsf"] = sold.sale_price / sold.floor_area_sqft
g = sold.groupby("client_ref").agg(units_purchased=("listing_id","count"), total_spend=("sale_price","sum"),
        avg_price=("sale_price","mean"), avg_area=("floor_area_sqft","mean"), avg_ppsf=("ppsf","mean"),
        office_share=("unit_category", lambda s: (s=="Office").mean()))
cl = cl.merge(g, left_on="client_id", right_index=True, how="left")
log["clients_without_purchase"] = int(cl.units_purchased.isna().sum())
cl[g.columns] = cl[g.columns].fillna(0)
cl["is_international"] = np.where(cl.country == "USA", "Domestic", "International")
cl["age_group"] = pd.cut(cl.age, [0,35,50,65,120], labels=["<35","35-49","50-64","65+"]).astype(str)

NUM = ["age","satisfaction_score","units_purchased","total_spend","avg_ppsf"]
CAT = ["client_type","acquisition_purpose","loan_applied","referral_channel","is_international"]
pre = ColumnTransformer([("num", StandardScaler(), NUM),
                         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])
X = pre.fit_transform(cl)

# ---------- 3. Choosing k ----------
rows = []
for k in range(2, 11):
    km = KMeans(k, n_init=20, random_state=SEED).fit(X)
    rows.append(dict(k=k, inertia=km.inertia_, silhouette=silhouette_score(X, km.labels_), dbi=davies_bouldin_score(X, km.labels_)))
sel = pd.DataFrame(rows); sel.to_csv("models/k_selection.csv", index=False)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(sel.k, sel.inertia, "o-"); ax[0].set(title="Elbow method", xlabel="k", ylabel="Inertia")
ax[1].plot(sel.k, sel.silhouette, "o-", c="C1"); ax[1].set(title="Silhouette score", xlabel="k")
ax[2].plot(sel.k, sel.dbi, "o-", c="C2"); ax[2].set(title="Davies-Bouldin (lower=better)", xlabel="k")
plt.tight_layout(); plt.savefig("figures/k_selection.png", dpi=150); plt.close()
print(sel.round(3).to_string())
K = 4  # business-aligned (brief recommends 4 segments); see paper for silhouette trade-off
km = KMeans(K, n_init=50, random_state=SEED).fit(X)
cl["cluster"] = km.labels_

# ---------- 4. Hierarchical validation ----------
hc = AgglomerativeClustering(K, linkage="ward").fit(X)
log["silhouette_kmeans"] = float(silhouette_score(X, km.labels_))
log["silhouette_hier"] = float(silhouette_score(X, hc.labels_))
log["ARI_kmeans_vs_hier"] = float(adjusted_rand_score(km.labels_, hc.labels_))
idx = np.random.RandomState(SEED).choice(len(X), 150, replace=False)
plt.figure(figsize=(12, 5)); dendrogram(linkage(X[idx], "ward"), no_labels=True, color_threshold=0)
plt.title("Ward dendrogram (random sample of 150 clients)"); plt.tight_layout(); plt.savefig("figures/dendrogram.png", dpi=150); plt.close()

# ---------- 5. Profiling & naming ----------
prof = cl.groupby("cluster").agg(size=("client_id","count"), age=("age","mean"), sat=("satisfaction_score","mean"),
        inv=("acquisition_purpose", lambda s:(s=="Investment").mean()), loan=("loan_applied", lambda s:(s=="Yes").mean()),
        corp=("client_type", lambda s:(s=="Company").mean()), intl=("is_international", lambda s:(s=="International").mean()),
        units=("units_purchased","mean"), spend=("total_spend","mean"))
print(prof.round(2).to_string())

# data-driven naming (rules on cluster profile, deterministic)
names = {}
names[prof.spend.idxmax()] = "Multi-Unit Portfolio Buyers"
rest = prof.drop(index=list(names))
names[rest.sat.idxmin()] = "Dissatisfied Mid-Life Buyers"
rest = prof.drop(index=list(names))
names[rest.age.idxmax()] = "Senior Established Buyers"
names[[c for c in prof.index if c not in names][0]] = "Younger Satisfied Buyers"
cl["segment"] = cl.cluster.map(names)

# brief-aligned rule-based persona overlay (business lens, not learned)
q90 = cl.total_spend.quantile(.90)
def persona(r):
    if r.client_type == "Company": return "Corporate Buyer"
    if r.total_spend >= q90 and r.satisfaction_score >= 4: return "Luxury Investor"
    if r.acquisition_purpose == "Investment" and r.is_international == "International": return "Global Investor"
    if r.acquisition_purpose == "Home" and r.loan_applied == "Yes" and r.age < 40: return "First-Time Buyer"
    return "Mainstream Buyer"
cl["persona"] = cl.apply(persona, axis=1)
prof["segment"] = prof.index.map(names); prof.to_csv("models/segment_profile.csv")
cl["pc1"], cl["pc2"] = PCA(2, random_state=SEED).fit_transform(X).T
out = cl.drop(columns=["first_name","last_name","dob","date_of_birth"])
out.to_csv("data/clients_segmented.csv", index=False)
sold.to_csv("data/sold_clean.csv", index=False)
joblib.dump(dict(pre=pre, km=km, NUM=NUM, CAT=CAT, names=names, ref_date=str(ref.date())), "models/pipeline.pkl")
log.update(segment_sizes=out.segment.value_counts().to_dict(), persona_sizes=out.persona.value_counts().to_dict())
json.dump(log, open("models/metrics.json","w"), indent=1); print(prof.round(2).to_string()); print(log)

# ---------- 6. Figures ----------
def save(n): plt.tight_layout(); plt.savefig(f"figures/{n}.png", dpi=150); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(cl.age, bins=25, kde=True, ax=ax[0]).set(title="Age distribution")
sns.countplot(x="satisfaction_score", data=cl, ax=ax[1]).set(title="Satisfaction score")
cl.country.value_counts().plot.bar(ax=ax[2], title="Clients by country"); save("eda_univariate")
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, c in zip(ax, ["acquisition_purpose","loan_applied","referral_channel"]):
    pd.crosstab(cl.client_type, cl[c], normalize="index").plot.bar(stacked=True, ax=a, title=f"{c} by client type"); a.legend(fontsize=8)
save("eda_bivariate")
num = ["age","satisfaction_score","units_purchased","total_spend","avg_ppsf","office_share"]
plt.figure(figsize=(6.5, 5)); sns.heatmap(cl[num].corr(), annot=True, fmt=".2f", cmap="vlag", center=0); plt.title("Correlation matrix"); save("eda_corr")
mk = pr.assign(m=pr.transaction_date.dt.to_period("M").astype(str)).groupby(["m","listing_status"]).size().unstack()
fig, ax = plt.subplots(1, 3, figsize=(15, 4)); mk.plot(ax=ax[0], title="Listings by month & status"); ax[0].tick_params(axis="x", rotation=60)
sns.boxplot(x="unit_category", y="sale_price", data=pr, ax=ax[1]).set(title="Sale price by unit category")
sns.scatterplot(x="floor_area_sqft", y="sale_price", hue="unit_category", data=pr.sample(2500, random_state=1), s=12, ax=ax[2]).set(title="Price vs. area"); save("eda_market")
z = (prof[["age","sat","inv","loan","corp","intl","units","spend"]] - prof[["age","sat","inv","loan","corp","intl","units","spend"]].mean()) / prof[["age","sat","inv","loan","corp","intl","units","spend"]].std()
z.index = z.index.map(names)
plt.figure(figsize=(9, 3.8)); sns.heatmap(z, annot=prof[z.columns].round(2).values, fmt="", cmap="vlag", center=0, cbar_kws=dict(label="z-score across segments")); plt.title("Segment profile (cell = raw mean / share)"); save("segment_heatmap")
plt.figure(figsize=(7, 5)); sns.scatterplot(x="pc1", y="pc2", hue="segment", data=out, s=14, alpha=.7); plt.title("Segments in PCA space"); save("segment_pca")
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
out.segment.value_counts().plot.barh(ax=ax[0], title="Segment sizes")
pd.crosstab(out.persona, out.segment, normalize="index").plot.barh(stacked=True, ax=ax[1], title="Segment mix inside each brief persona"); ax[1].legend(fontsize=7); save("segment_sizes_persona")
