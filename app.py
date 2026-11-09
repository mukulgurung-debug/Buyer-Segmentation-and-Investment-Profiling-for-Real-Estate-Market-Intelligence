import json
import pandas as pd, plotly.express as px, streamlit as st

st.set_page_config(page_title="Parcl Buyer Segmentation", page_icon="🏢", layout="wide")

US = {"Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO","Connecticut":"CT","Delaware":"DE",
"Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY",
"Louisiana":"LA","Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN","Mississippi":"MS","Missouri":"MO",
"Montana":"MT","Nebraska":"NE","Nevada":"NV","New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY","North Carolina":"NC",
"North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC",
"South Dakota":"SD","Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA","West Virginia":"WV",
"Wisconsin":"WI","Wyoming":"WY"}
ISO3 = {"USA":"USA","UK":"GBR","Canada":"CAN","Germany":"DEU","France":"FRA","Belgium":"BEL","Mexico":"MEX","Australia":"AUS","Russia":"RUS","Denmark":"DNK"}

@st.cache_data
def load():
    return pd.read_csv("data/clients_segmented.csv"), pd.read_csv("models/segment_profile.csv"), json.load(open("models/metrics.json"))
df, prof, metrics = load()
seg_order = df.segment.value_counts().index.tolist()
cmap = {s: c for s, c in zip(seg_order, px.colors.qualitative.Safe)}

# ---------------- sidebar filters ----------------
st.sidebar.header("Filters")
def pick(label, col):
    opts = sorted(df[col].unique()); return st.sidebar.multiselect(label, opts, default=[])
f_country, f_type, f_purpose = pick("Country", "country"), pick("Client type", "client_type"), pick("Acquisition purpose", "acquisition_purpose")
reg_opts = sorted(df[df.country.isin(f_country)].region.unique() if f_country else df.region.unique())
f_region = st.sidebar.multiselect("Region", reg_opts, default=[])
d = df.copy()
for col, sel in [("country", f_country), ("client_type", f_type), ("acquisition_purpose", f_purpose), ("region", f_region)]:
    if sel: d = d[d[col].isin(sel)]
st.sidebar.metric("Clients in view", f"{len(d):,} / {len(df):,}")
st.sidebar.caption("Leave a filter empty to include everything.")

st.title("🏢 Buyer Segmentation & Investment Profiling")
st.caption("Parcl · K-Means (k=4) on age, satisfaction, purchase behaviour and buyer profile · 2,000 clients, 7,305 sold units")
if d.empty: st.warning("No clients match the current filters."); st.stop()

t1, t2, t3, t4 = st.tabs(["Segmentation overview", "Investor behaviour", "Geographic analysis", "Segment insights"])

with t1:
    c = st.columns(4)
    c[0].metric("Clients", f"{len(d):,}"); c[1].metric("Investment share", f"{(d.acquisition_purpose=='Investment').mean():.0%}")
    c[2].metric("Loan applied", f"{(d.loan_applied=='Yes').mean():.0%}"); c[3].metric("Avg satisfaction", f"{d.satisfaction_score.mean():.2f} / 5")
    a, b = st.columns(2)
    cnt = d.segment.value_counts().reset_index(); cnt.columns = ["segment", "clients"]
    a.plotly_chart(px.pie(cnt, names="segment", values="clients", hole=.45, color="segment", color_discrete_map=cmap, title="Cluster distribution"), width="stretch")
    b.plotly_chart(px.scatter(d, x="pc1", y="pc2", color="segment", color_discrete_map=cmap, opacity=.7, title="Segments projected to 2 principal components",
                              hover_data=["country", "acquisition_purpose", "satisfaction_score"]), width="stretch")
    p = d.persona.value_counts().reset_index(); p.columns = ["persona", "clients"]
    st.plotly_chart(px.bar(p, x="clients", y="persona", orientation="h", title="Brief-aligned buyer personas (rule-based overlay)"), width="stretch")
    with st.expander("Model quality & caveats"):
        st.write(f"Silhouette (K-Means): **{metrics['silhouette_kmeans']:.3f}**, hierarchical (Ward): {metrics['silhouette_hier']:.3f}, "
                 f"agreement K-Means vs. hierarchical (ARI): {metrics['ARI_kmeans_vs_hier']:.2f}.")
        st.write("Scores are low: the source variables are almost uncorrelated, so segments are soft groupings along age, satisfaction and purchase volume rather than sharply separated clusters. See the research paper.")

with t2:
    a, b = st.columns(2)
    inv = d.groupby("segment").acquisition_purpose.apply(lambda s: (s == "Investment").mean()).reset_index(name="share")
    a.plotly_chart(px.bar(inv, x="segment", y="share", color="segment", color_discrete_map=cmap, title="Investment-purpose share by segment").update_yaxes(tickformat=".0%").update_layout(showlegend=False), width="stretch")
    ln = d.groupby("segment").loan_applied.apply(lambda s: (s == "Yes").mean()).reset_index(name="share")
    b.plotly_chart(px.bar(ln, x="segment", y="share", color="segment", color_discrete_map=cmap, title="Loan-application share by segment").update_yaxes(tickformat=".0%").update_layout(showlegend=False), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(px.box(d, x="segment", y="total_spend", color="segment", color_discrete_map=cmap, title="Total spend per client (USD)").update_layout(showlegend=False), width="stretch")
    b.plotly_chart(px.histogram(d, x="units_purchased", color="acquisition_purpose", barmode="group", title="Units purchased, by purpose"), width="stretch")
    ch = pd.crosstab(d.segment, d.referral_channel, normalize="index").reset_index().melt("segment", value_name="share")
    st.plotly_chart(px.bar(ch, x="segment", y="share", color="referral_channel", title="Acquisition channel mix by segment").update_yaxes(tickformat=".0%"), width="stretch")

with t3:
    a, b = st.columns(2)
    g = d.groupby("country").agg(clients=("client_id", "count"), spend=("total_spend", "mean")).reset_index(); g["iso"] = g.country.map(ISO3)
    a.plotly_chart(px.choropleth(g, locations="iso", color="clients", hover_name="country", hover_data=["spend"], color_continuous_scale="Blues", title="Clients by country"), width="stretch")
    us = d[d.country == "USA"].copy(); us["st"] = us.region.map(US)
    if len(us):
        gs = us.groupby("st").agg(clients=("client_id", "count")).reset_index()
        dom = us.groupby("st").segment.agg(lambda s: s.value_counts().index[0]).rename("top_segment").reset_index()
        gs = gs.merge(dom)
        b.plotly_chart(px.choropleth(gs, locations="st", locationmode="USA-states", scope="usa", color="top_segment", color_discrete_map=cmap, hover_data=["clients"], title="Dominant segment by US state"), width="stretch")
    else: b.info("No US clients in the current selection.")
    top = d.region.value_counts().head(15).index
    hm = pd.crosstab(d[d.region.isin(top)].region, d[d.region.isin(top)].segment, normalize="index")
    st.plotly_chart(px.imshow(hm, text_auto=".0%", aspect="auto", color_continuous_scale="Blues", title="Segment mix in the 15 largest regions"), width="stretch")

with t4:
    s = st.selectbox("Segment", seg_order)
    x = d[d.segment == s]
    if x.empty: st.info("No clients of this segment in the current selection.")
    else:
        c = st.columns(5)
        c[0].metric("Clients", len(x)); c[1].metric("Mean age", f"{x.age.mean():.1f}"); c[2].metric("Satisfaction", f"{x.satisfaction_score.mean():.2f}")
        c[3].metric("Units / client", f"{x.units_purchased.mean():.1f}"); c[4].metric("Avg spend", f"${x.total_spend.mean()/1e6:.2f}M")
        st.write(f"Investment {(x.acquisition_purpose=='Investment').mean():.0%} · Loan {(x.loan_applied=='Yes').mean():.0%} · Corporate {(x.client_type=='Company').mean():.0%} · International {(x.is_international=='International').mean():.0%}")
    st.subheader("Descriptive statistics by segment")
    stats = d.groupby("segment")[["age", "satisfaction_score", "units_purchased", "total_spend", "avg_ppsf"]].agg(["mean", "median", "std"]).round(2)
    st.dataframe(stats, width="stretch")
    st.subheader("Recommended action")
    ACT = {"Multi-Unit Portfolio Buyers": "Small, high-value group (≈2× spend). Assign relationship managers; offer bulk/portfolio deals and early access to new towers.",
           "Dissatisfied Mid-Life Buyers": "Satisfaction ≈1.4/5. Prioritise service recovery and follow-up before they churn or hurt referrals.",
           "Senior Established Buyers": "Oldest group, solid satisfaction. Focus on trust, downsizing and legacy/estate-oriented offers.",
           "Younger Satisfied Buyers": "Youngest, happiest and most loan-using. Referral programmes and financing partnerships."}
    st.info(ACT.get(s, ""))
    st.download_button("Download filtered data (CSV)", d.to_csv(index=False), "segmented_clients_filtered.csv")
