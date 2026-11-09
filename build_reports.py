"""Builds deliverables/Research_Paper.pdf and deliverables/Executive_Summary.pdf from pipeline outputs."""
import json, pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether

d = pd.read_csv("data/clients_segmented.csv"); prof = pd.read_csv("models/segment_profile.csv"); m = json.load(open("models/metrics.json"))
ks = pd.read_csv("models/k_selection.csv"); sold = pd.read_csv("data/sold_clean.csv"); pr = pd.read_csv("data/properties.csv")
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=10, textColor=colors.HexColor("#1F3A5F"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=6, textColor=colors.HexColor("#1F3A5F"))
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.6, leading=13.2, spaceAfter=5)
CAP = ParagraphStyle("C", parent=B, fontSize=8, textColor=colors.grey, alignment=1)
TITLE = ParagraphStyle("T", parent=ss["Title"], fontSize=19, textColor=colors.HexColor("#1F3A5F"))
def P(t, s=B): return Paragraph(t, s)
def bullets(items): return [P("&bull; " + i) for i in items]
def fig(path, cap, w=17):
    from PIL import Image as PI
    iw, ih = PI.open(path).size
    return KeepTogether([Image(path, w*cm, w*cm*ih/iw), P(cap, CAP)])
def table(rows, widths=None, fs=8):
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1F3A5F")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTSIZE",(0,0),(-1,-1),fs),("GRID",(0,0),(-1,-1),.3,colors.grey),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#EEF2F7")]),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE")])); return t

seg = d.groupby("segment").agg(n=("client_id","count"), age=("age","mean"), sat=("satisfaction_score","mean"),
    inv=("acquisition_purpose",lambda s:(s=="Investment").mean()), loan=("loan_applied",lambda s:(s=="Yes").mean()),
    corp=("client_type",lambda s:(s=="Company").mean()), intl=("is_international",lambda s:(s=="International").mean()),
    units=("units_purchased","mean"), spend=("total_spend","mean"), tot=("total_spend","sum")).sort_values("n", ascending=False)
seg["rev_share"] = seg.tot / seg.tot.sum(); seg["cl_share"] = seg.n / seg.n.sum()
tot_rev = d.total_spend.sum(); k4 = ks[ks.k==4].iloc[0]
per = d.groupby("persona").agg(n=("client_id","count"), age=("age","mean"), sat=("satisfaction_score","mean"), units=("units_purchased","mean"), spend=("total_spend","mean")).sort_values("n", ascending=False)
dis = seg.loc["Dissatisfied Mid-Life Buyers"]; pf = seg.loc["Multi-Unit Portfolio Buyers"]
sold_rate = (pr.listing_status=="Sold").mean()

# ====================== RESEARCH PAPER ======================
S = [P("Machine Learning-Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence", TITLE),
     P("Research paper &middot; Parcl Co. Limited (Unified Mentor project)", CAP), Spacer(1, 8),
     P("Abstract", H1),
     P(f"We segment {len(d):,} property buyers of a multi-tower residential/office development using K-Means clustering, validated with Ward hierarchical clustering. "
       f"Client attributes (age, satisfaction, client type, purchase purpose, loan use, referral channel, domestic/international) are joined to behavioural purchase features "
       f"derived from {len(sold):,} sold units (units bought, total spend, average price per sq ft). Four segments emerge: <b>Younger Satisfied</b>, <b>Senior Established</b>, "
       f"<b>Dissatisfied Mid-Life</b> and a small <b>Multi-Unit Portfolio</b> group. The most actionable findings are that the dissatisfied group holds "
       f"{dis.rev_share:.0%} of revenue, and that {pf.n:.0f} portfolio buyers ({pf.cl_share:.1%} of clients) generate {pf.rev_share:.1%} of it at twice the average ticket. "
       f"Cluster separation is modest (silhouette {m['silhouette_kmeans']:.2f}) because the underlying attributes are nearly uncorrelated; we report this limitation explicitly and add a "
       f"transparent rule-based persona overlay (Global Investor, First-Time, Corporate, Luxury) that mirrors the brief's business language."),
     P("1. Background and objectives", H1),
     P("Without segmentation a developer markets identically to home buyers, institutions, foreign investors and first-time purchasers, which wastes marketing spend and produces generic recommendations. "
       "The objectives are to (i) discover hidden buyer groups, (ii) describe their investment motives, geography and financing behaviour, and (iii) deliver an interactive dashboard so marketing teams can explore segments by country, region, purpose and client type."),
     P("2. Data", H1),
     table([["Table","Rows","Content"],
            ["clients.csv", f"{m['clients_raw']:,}", "client_id, client_type, name, date_of_birth, gender, country, region, acquisition_purpose, satisfaction_score (1-5), loan_applied, referral_channel"],
            ["properties.csv", f"{m['properties_raw']:,}", f"listing_id, tower, transaction_date, unit_category (Apartment/Office), floor_area_sqft, sale_price, listing_status ({sold_rate:.0%} Sold), client_ref"]],
           [3*cm, 2*cm, 12*cm]),
     Spacer(1, 4),
     P("2.1 Cleaning", H2)] + bullets([
        f"No missing client attributes and no duplicate client_id or listing_id were found (missing cells: {m['missing_client_cells']}, duplicate listings: {m['dup_listing_ids']}); the pipeline nonetheless strips whitespace, normalises label case and de-duplicates.",
        "<b>Mixed date formats:</b> date_of_birth mixes dd-mm-yyyy (dash) and m/d/yyyy (slash). Both were parsed explicitly (0 unparsed). Age is measured at the last transaction date.",
        "sale_price was stored as text (\"$300,385.62\") and converted to float.",
        f"Only <i>Sold</i> listings with a client_ref were attributed to buyers; all 2,000 clients own at least 3 units (max {int(d.units_purchased.max())}). Unit numbers repeat within a tower (e.g. two listings of unit 4), which we treat as data-entry noise and did not use."]) + [
     P("2.2 Feature engineering", H2),
     P("Per-client aggregates from sold units: <b>units_purchased</b>, <b>total_spend</b>, avg_price, avg_area, <b>avg_ppsf</b> (price per sq ft), office_share. Derived client fields: age, age group and <b>is_international</b> (country != USA). "
       "These are the behavioural signals the brief calls for (&ldquo;multiple units&rdquo;, &ldquo;large investments&rdquo;), which the raw client table cannot provide."),
     P("3. Exploratory data analysis", H1),
     fig("figures/eda_univariate.png", "Figure 1. Univariate: ages span 24-94, satisfaction is nearly uniform, and 77% of clients are in the USA (California alone is 32%)."),
     fig("figures/eda_bivariate.png", "Figure 2. Bivariate: purpose, loan use and channel are almost identical for individuals and companies."),
     fig("figures/eda_corr.png", "Figure 3. Multivariate: only units_purchased vs. total_spend (0.71) is strongly related; client attributes are near-independent.", 9),
     fig("figures/eda_market.png", "Figure 4. Market context: listing volume by month, price distribution by unit category, and price vs. area."),
     P(f"<b>Key observations.</b> (1) {(d.acquisition_purpose=='Investment').mean():.0%} of buyers purchase for investment, {(d.loan_applied=='Yes').mean():.0%} apply for a loan, {(d.client_type=='Company').mean():.1%} are companies. "
       f"(2) Offices sell at a ~6% higher mean price (${pr.assign(p=pr.sale_price.str.replace(r'[$,]','',regex=True).astype(float)).groupby('unit_category').p.mean()['Office']/1e3:.0f}k vs ${pr.assign(p=pr.sale_price.str.replace(r'[$,]','',regex=True).astype(float)).groupby('unit_category').p.mean()['Apartment']/1e3:.0f}k for apartments). "
       "(3) Correlations between any two client attributes are below 0.12 in absolute value: investors are not systematically older, wealthier or more loan-dependent than home buyers. This is the single most important fact for interpreting the clustering."),
     P("4. Methodology", H1),
     P("<b>Encoding and scaling.</b> One-hot encoding for client_type, acquisition_purpose, loan_applied, referral_channel and is_international (country and region are one-hot-heavy with 10 and 57 levels, so they are collapsed to domestic/international for clustering and used for post-hoc geographic profiling). "
       "StandardScaler for age, satisfaction_score, units_purchased, total_spend and avg_ppsf, all wrapped in a scikit-learn ColumnTransformer saved as models/pipeline.pkl."),
     P("<b>Models.</b> K-Means (n_init=50, seed 42) as the primary, interpretable model; agglomerative (Ward) clustering as an independent check and to reveal nested structure."),
     P("<b>Choosing k.</b>"),
     fig("figures/k_selection.png", "Figure 5. Elbow shows no sharp bend; silhouette is flat (0.10-0.125) for k=2-10; Davies-Bouldin improves steadily until k=6-8."),
     table([["k","Inertia","Silhouette","Davies-Bouldin"]] + [[int(r.k), f"{r.inertia:,.0f}", f"{r.silhouette:.3f}", f"{r.dbi:.2f}"] for r in ks.itertuples()], [2*cm,3*cm,3*cm,3*cm]),
     Spacer(1,4),
     P(f"No k is statistically dominant. We chose <b>k = 4</b> because (a) silhouette at k=4 ({k4.silhouette:.3f}) is within 0.01 of the maximum, (b) the brief recommends four segments, and (c) four groups remain distinct and actionable, whereas larger k mostly splits groups on noise. This is a judgement call, not a statistical optimum."),
     P("5. Results", H1),
     P("5.1 Segment profiles", H2),
     table([["Segment","Clients","Age","Satisf.","Invest.","Loan","Corp.","Intl.","Units","Avg spend","% revenue"]] +
           [[i, f"{r.n:.0f}", f"{r.age:.1f}", f"{r.sat:.2f}", f"{r.inv:.0%}", f"{r.loan:.0%}", f"{r.corp:.0%}", f"{r.intl:.0%}", f"{r.units:.1f}", f"${r.spend/1e6:.2f}M", f"{r.rev_share:.0%}"] for i, r in seg.iterrows()],
           [4.2*cm,1.3*cm,1.1*cm,1.3*cm,1.3*cm,1.1*cm,1.1*cm,1.1*cm,1.1*cm,1.8*cm,1.6*cm], 7.5),
     Spacer(1,4), fig("figures/segment_heatmap.png", "Figure 6. Segment profile heat-map (colour = z-score across segments, text = raw value).", 13)] + bullets([
        f"<b>Multi-Unit Portfolio Buyers</b> ({pf.n:.0f} clients): {pf.units:.1f} units and ${pf.spend/1e6:.1f}M each, ~2x the average; the genuine &ldquo;large investor&rdquo; group, found by purchase behaviour, not by the self-declared investment flag.",
        f"<b>Dissatisfied Mid-Life Buyers</b> ({dis.n:.0f}): mean satisfaction {dis.sat:.2f}/5, age ~47. Purchasing is typical, so this is a retention and reputation risk worth {dis.rev_share:.0%} of revenue.",
        f"<b>Senior Established Buyers</b>: oldest (mean age {seg.loc['Senior Established Buyers'].age:.0f}), neutral-to-positive satisfaction, almost no companies ({seg.loc['Senior Established Buyers'].corp:.1%}), lowest investment share ({seg.loc['Senior Established Buyers'].inv:.0%}).",
        f"<b>Younger Satisfied Buyers</b>: youngest (~43), satisfaction {seg.loc['Younger Satisfied Buyers'].sat:.1f}, highest loan use ({seg.loc['Younger Satisfied Buyers'].loan:.0%}) and international share ({seg.loc['Younger Satisfied Buyers'].intl:.0%})."]) + [
     fig("figures/segment_pca.png", "Figure 7. Segments in PCA space: boundaries overlap, consistent with the low silhouette.", 10),
     P("5.2 Hierarchical validation", H2),
     fig("figures/dendrogram.png", "Figure 8. Ward dendrogram on a 150-client sample.", 13),
     P(f"Ward clustering with k=4 gives silhouette {m['silhouette_hier']:.3f} versus {m['silhouette_kmeans']:.3f} for K-Means, and an adjusted Rand index of {m['ARI_kmeans_vs_hier']:.2f} between the two solutions. "
       "Agreement is only partial: the dendrogram shows no deep split, so the cluster boundaries are not natural gaps. K-Means is retained for its stability and interpretability; the segments should be treated as useful, soft groupings."),
     P("5.3 Persona overlay aligned to the brief", H2),
     P("The brief names Global Investors, First-Time Buyers, Corporate Buyers and Luxury Investors. Because the data do not produce these as separate clusters, we tag them with explicit, auditable rules (Corporate: client_type=Company; Luxury: top-decile total spend and satisfaction &ge; 4; Global: investment purpose and non-US; First-Time: home purpose, loan, age &lt; 40; otherwise Mainstream), applied in that priority order."),
     table([["Persona","Clients","Mean age","Satisf.","Units","Avg spend"]] + [[i, f"{r.n:.0f}", f"{r.age:.1f}", f"{r.sat:.2f}", f"{r.units:.1f}", f"${r.spend/1e6:.2f}M"] for i, r in per.iterrows()], [4.5*cm,2*cm,2.2*cm,2*cm,2*cm,2.6*cm]),
     Spacer(1,4), fig("figures/segment_sizes_persona.png", "Figure 9. Segment sizes and the cluster mix inside each persona.", 15),
     P("6. Streamlit dashboard", H1),
     P("The app (app.py) provides four views: <b>Segmentation overview</b> (cluster distribution, PCA map, persona counts); <b>Investor behaviour</b> (purpose, loan, spend, units and channel by segment); "
       "<b>Geographic analysis</b> (country choropleth, dominant segment by US state, region-by-segment heat-map); and <b>Segment insights</b> (per-segment KPIs, descriptive statistics, recommended action, CSV export). "
       "Sidebar filters cover country, region, acquisition purpose and client type, and every chart responds to them."),
     P("7. Recommendations", H1)] + bullets([
        f"<b>Service recovery for the Dissatisfied segment.</b> Contact the {dis.n:.0f} clients rated &le; 2, find the root cause (build quality, handover, communication) and track satisfaction quarterly; ${dis.tot/1e6:,.0f}M of purchase value sits here and unhappy buyers depress referrals (only 10% of clients arrive by referral).",
        f"<b>Account management for Portfolio Buyers.</b> A dedicated relationship manager, bulk or bundle pricing and pre-launch access for {pf.n:.0f} clients who spend ${pf.spend/1e6:.1f}M each. Lookalike targeting should use units purchased, not the self-reported investment flag.",
        "<b>Financing-led offers for younger buyers,</b> who use loans most; partner with lenders and highlight monthly-payment framing.",
        "<b>Do not segment marketing on purpose or country alone:</b> these carry almost no predictive signal in this data (Section 3). Collect richer fields (income, source of funds, prior ownership, property use, budget) to make future clustering sharper.",
        "<b>Geographic focus:</b> California is a third of clients; test international campaigns on the UK/Germany/France/Canada tail, where average spend is comparable to the US."]) + [
     P("8. Limitations", H1)] + bullets([
        f"Weak cluster structure (silhouette {m['silhouette_kmeans']:.2f}; low K-Means/Ward agreement). Segments are descriptive, not sharply separated.",
        "Client attributes appear statistically independent, which suggests synthetic or heavily anonymised data; findings may not transfer to live data.",
        "No income, wealth or first-time-buyer field exists, so personas rely on proxies (age, loan, spend).",
        "Satisfaction is a single 1-5 score with no timestamp, so causal claims are not supported."]) + [
     P("9. Conclusion", H1),
     P("Behavioural purchase features reveal a small, high-value portfolio-buyer group and a large satisfaction-defined risk group that demographic fields alone would miss. The work delivers a reproducible pipeline, a saved model and a filterable dashboard, and is candid that stronger segmentation requires richer buyer data.")]

def build(path, story, margins=2*cm):
    SimpleDocTemplate(path, pagesize=A4, leftMargin=margins, rightMargin=margins, topMargin=1.8*cm, bottomMargin=1.8*cm).build(story)
build("deliverables/Research_Paper.pdf", S)

# ====================== EXECUTIVE SUMMARY ======================
E = [P("Executive Summary: Buyer Segmentation for Parcl", TITLE),
     P("What we did", H1),
     P(f"We grouped {len(d):,} buyers of {len(sold):,} sold units (${tot_rev/1e9:.2f}B total) into four segments using K-Means clustering, combining who they are (age, satisfaction, financing, channel) with what they bought (units, spend, price per sq ft). A live dashboard lets marketing filter by country, region, purpose and client type."),
     P("Four segments", H1),
     table([["Segment","Clients","% revenue","Profile","Action"]] + [
        ["Younger Satisfied", f"{seg.loc['Younger Satisfied Buyers'].n:.0f}", f"{seg.loc['Younger Satisfied Buyers'].rev_share:.0%}", "Age ~43, satisfaction 4.1, most loan use", "Referral and financing offers"],
        ["Senior Established", f"{seg.loc['Senior Established Buyers'].n:.0f}", f"{seg.loc['Senior Established Buyers'].rev_share:.0%}", "Age ~73, few companies, fewer investors", "Trust, downsizing offers"],
        ["Dissatisfied Mid-Life", f"{dis.n:.0f}", f"{dis.rev_share:.0%}", "Age ~47, satisfaction 1.4/5", "Service recovery now"],
        ["Multi-Unit Portfolio", f"{pf.n:.0f}", f"{pf.rev_share:.0%}", f"{pf.units:.1f} units, ${pf.spend/1e6:.1f}M each (2x avg)", "Account management"]],
        [3.3*cm,1.5*cm,1.8*cm,5.6*cm,4.9*cm], 7.5),
     P("Key drivers", H1)] + bullets([
        "<b>Behavioural:</b> purchase volume (units, spend) and satisfaction separate buyers far more than declared purpose, geography or channel.",
        "<b>Demographic:</b> age splits the satisfied majority into younger (loan-reliant, international) and senior (lower loan use, mostly domestic) buyers.",
        f"<b>Declared investor status is a weak signal:</b> {(d.acquisition_purpose=='Investment').mean():.0%} of buyers are investors, but they buy the same volume and spend as home buyers."]) + [
     P("Financial impact (illustrative)", H1),
     P(f"The dissatisfied segment represents about <b>${dis.tot/1e6:,.0f}M</b> of purchase value. The figures below are <b>assumptions for planning, not measured results</b>: "
       f"if a service-recovery programme lifted repeat purchase or referral value by even 2% of this base, it would be worth ~${dis.tot*0.02/1e6:,.0f}M, against a modest per-client outreach cost (e.g. $500 x {dis.n:.0f} = ${dis.n*500/1e3:,.0f}k), "
       f"an indicative return of ~{dis.tot*0.02/(dis.n*500):.0f}x. Portfolio buyers are {pf.cl_share:.1%} of clients but {pf.rev_share:.1%} of revenue; a 10% uplift in their units would add ~${pf.tot*0.10/1e6:,.0f}M. "
       "Actual uplift should be validated with an A/B pilot before scaling."),
     P("Governance and data implications", H1)] + bullets([
        "<b>Treat segments as soft groupings.</b> Cluster separation is modest (silhouette 0.12). Use them to prioritise outreach, not to automate decisions about individuals.",
        "<b>Fairness and privacy:</b> age, gender and nationality must not drive pricing, credit or access to units. Segment by behaviour and need; exclude protected attributes from any offer-eligibility logic.",
        "<b>Data investment:</b> capture income band, source of funds, prior ownership and satisfaction timestamps to sharpen segments; refresh the model quarterly and monitor segment drift.",
        "<b>Ownership:</b> assign a business owner to each segment's KPI (satisfaction, units per client, referral share)."]) + [
     P("Next steps", H1),
     P("(1) Launch a 90-day service-recovery pilot on the dissatisfied segment with a control group. (2) Assign relationship managers to portfolio buyers. (3) Pilot lender-partnered offers for younger buyers. (4) Enrich the client data and re-cluster.")]
build("deliverables/Executive_Summary.pdf", E, 1.8*cm)
print("built")
