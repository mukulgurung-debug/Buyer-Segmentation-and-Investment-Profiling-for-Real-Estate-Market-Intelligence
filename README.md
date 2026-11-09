# Buyer Segmentation & Investment Profiling – Real Estate Market Intelligence

Unified Mentor project for Parcl Co. Limited. Clusters 2,000 property buyers (K-Means, validated with Ward hierarchical clustering) using client attributes plus purchase behaviour from 7,305 sold units, and serves the results in a Streamlit dashboard.

## Contents
| Path | Purpose |
|---|---|
| `app.py` | Streamlit dashboard (overview, investor behaviour, geography, segment insights; filters for country, region, purpose, client type) |
| `pipeline.py` | Cleaning, feature engineering, encoding/scaling, k selection, clustering, profiling, figures |
| `build_reports.py` | Generates the PDFs in `deliverables/` |
| `models/pipeline.pkl` | Serialized ColumnTransformer (one-hot + StandardScaler) and K-Means model, with segment names |
| `data/` | Raw CSVs, `clients_segmented.csv` (pipeline output used by the app) |
| `deliverables/Research_Paper.pdf`, `Executive_Summary.pdf` | Written deliverables |
| `figures/` | EDA, k-selection and segment plots |

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python pipeline.py        # optional: regenerates data/clients_segmented.csv, models, figures
python build_reports.py   # optional: regenerates the PDFs
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
Push this folder to GitHub, then create an app at share.streamlit.io pointing to `app.py`. The app reads only committed files (`data/clients_segmented.csv`, `models/segment_profile.csv`, `models/metrics.json`).

## Method summary
1. **Cleaning:** mixed date formats parsed (dd-mm-yyyy vs m/d/yyyy), prices converted from text, labels normalised, duplicates removed.
2. **Features:** per-client units purchased, total spend, price/sq ft from sold listings; age; domestic/international flag.
3. **Encoding/scaling:** one-hot for categoricals, StandardScaler for numerics.
4. **Clustering:** K-Means k=4 (elbow, silhouette and Davies-Bouldin inspected for k=2–10); Ward hierarchical clustering as validation.
5. **Segments:** Younger Satisfied, Senior Established, Dissatisfied Mid-Life, Multi-Unit Portfolio, plus a rule-based persona overlay (Global Investor, First-Time, Corporate, Luxury).

## Caveat
Silhouette is ~0.12 and client attributes are nearly uncorrelated, so segments are soft groupings. See Section 8 of the research paper.
