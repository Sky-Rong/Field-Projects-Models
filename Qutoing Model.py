import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Job inputs
job = {
    "system": str(xl("B4")).strip().upper(),
    "voltage": float(xl("B5")),
    "length": float(xl("B6")),
    "circuits": float(xl("B7")),
    "joints": float(xl("B8")),
    "terms": float(xl("B9")),
    "eng": float(xl("B10")),
    "days": float(xl("B11"))
}

# Systems beginning with GIS use GIS; all others use AIS.
use_gis = job["system"].startswith("GIS")

# A14 stores how many results to show
results_to_show = pd.to_numeric(xl("A14"), errors="coerce")
if pd.isna(results_to_show):
    results_to_show = 5
results_to_show = max(1, int(results_to_show))

# Source Tables
if use_gis:
    df = xl("tblGISProjects[#All]", headers=True)
else:
    df = xl("tblAISProjects[#All]", headers=True)

# Similarity Weights
wt = xl("D4:E9", headers=False)
W = {
    str(row[0]).strip().lower(): float(row[1])
    for row in wt.values
}

# Labor Index
ix = xl("tblLaborIndex[#All]", headers=True).copy()
ix["Year"] = pd.to_numeric(ix["Year"], errors="coerce")
ix["Index"] = pd.to_numeric(ix["Index"], errors="coerce")
ix = ix.dropna(subset=["Year", "Index"])
ix = ix[ix["Index"].gt(0)].copy()

idx_map = {
    int(year): float(index_value)
    for year, index_value in zip(ix["Year"], ix["Index"])
}

if not idx_map:
    raise ValueError("No valid Year and Index values were found in tblLaborIndex.")

CUR = max(idx_map)
base = idx_map[CUR]

# Source-specific columns
if use_gis:
    cols = {
        "Project Name": "project",
        "System": "system",
        "Rated (kV)": "voltage",
        "Joints": "joints",
        "Terms": "terms",
        "# Eng": "eng",
        "# Days": "days",
        "Value ($)": "value",
        "Start Date": "date"
    }
    similarity_features = ["voltage", "joints", "terms"]
    model_features = ["voltage", "joints", "terms", "eng", "days"]
else:
    cols = {
        "Project Name": "project",
        "System": "system",
        "Rated (kV)": "voltage",
        "Total ft": "length",
        "Circuits": "circuits",
        "Joints": "joints",
        "Terms": "terms",
        "# Eng": "eng",
        "# Days": "days",
        "Value ($)": "value",
        "Start Date": "date"
    }
    similarity_features = ["voltage", "length", "circuits", "joints", "terms"]
    model_features = ["voltage", "length", "circuits", "joints", "terms", "eng", "days"]

missing = [name for name in cols if name not in df.columns]
if missing:
    source = "GIS" if use_gis else "AIS"
    raise ValueError(f"Missing {source} headers: {missing}")

df = df.rename(columns=cols)[list(cols.values())].copy()

# Remove blank and total rows
df["project"] = df["project"].astype(str).str.strip()
df = df[
    df["project"].ne("")
    & df["project"].str.upper().ne("NAN")
    & df["project"].str.upper().ne("TOTAL PROJECTS")
].copy()
df["system"] = df["system"].astype(str).str.upper().str.strip()

# Clean Numbers
def clean_number(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.extract(r"(-?\d+(?:\.\d+)?)")[0],
        errors="coerce"
    )

for column in model_features + ["value"]:
    df[column] = clean_number(df[column])

# Project Year
serial = pd.to_numeric(df["date"], errors="coerce")
df["year"] = pd.to_datetime(
    serial,
    unit="D",
    origin="1899-12-30",
    errors="coerce"
).dt.year

df = df[df["value"].notna() & df["value"].gt(0)].reset_index(drop=True)
if df.empty:
    source = "GIS" if use_gis else "AIS"
    raise ValueError(f"No quoted {source} projects remain after cleaning.")

# Inflation Adjustment
def to_today(row):
    year = row["year"]
    if pd.isna(year) or int(year) not in idx_map:
        return row["value"]
    return row["value"] * base / idx_map[int(year)]

df["value_adj"] = df.apply(to_today, axis=1)

# Similarity and Match Perecent
ranges = {}
for feature in similarity_features:
    low = df[feature].min()
    high = df[feature].max()
    if high == low:
        high = low + 1
    ranges[feature] = (low, high)

def normalize(value, feature):
    low, high = ranges[feature]
    return (value - low) / (high - low)

distance = W["system"] * df["system"].ne(job["system"]).astype(float)
for feature in similarity_features:
    historical = df[feature].apply(lambda value: normalize(value, feature))
    entered = normalize(job[feature], feature)
    distance = distance + W[feature] * (historical - entered).abs()

active_weight = W["system"] + sum(W[feature] for feature in similarity_features)
if active_weight == 0:
    active_weight = 1

df["match_distance"] = distance
df["Match %"] = (100 * (1 - distance / active_weight)).clip(lower=0).round(1)

top_results = df.sort_values("match_distance").head(results_to_show).copy()
lo = top_results["value"].min()
hi = top_results["value"].max()

# Random Forest
model_df = df.dropna(subset=model_features + ["value_adj", "system"]).copy()
if model_df.shape[0] in range(5):
    source = "GIS" if use_gis else "AIS"
    raise ValueError(
        f"Only {model_df.shape[0]} complete quoted {source} projects remain for training."
    )

X = pd.get_dummies(
    model_df[model_features + ["system"]],
    columns=["system"]
)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    min_samples_leaf=2
)
model.fit(X, model_df["value_adj"])

new_job = {feature: job[feature] for feature in model_features}
new_job["system"] = job["system"]
new_X = pd.get_dummies(
    pd.DataFrame([new_job]),
    columns=["system"]
).reindex(columns=X.columns, fill_value=0)

estimate = float(model.predict(new_X)[0])

# Output
if use_gis:
    out = top_results[
        ["project", "system", "voltage", "joints", "terms", "value", "Match %"]
    ].copy()
    out.columns = [
        "Project Name", "System", "Rated kV", "Joints", "Terms", "Value ($)", "Match %"
    ]
else:
    out = top_results[
        ["project", "system", "voltage", "length", "terms", "value", "Match %"]
    ].copy()
    out.columns = [
        "Project Name", "System", "Rated kV", "Total ft", "Terms", "Value ($)", "Match %"
    ]

blank = {column: "" for column in out.columns}
range_row = blank.copy()
range_row["Project Name"] = "COMPARABLE RANGE (nominal)"
range_row["Value ($)"] = "$" + format(lo, ",.0f") + "-$" + format(hi, ",.0f")

estimate_row = blank.copy()
estimate_row["Project Name"] = "MODEL ESTIMATE (" + str(CUR) + " $)"
estimate_row["Value ($)"] = "$" + format(estimate, ",.0f")

out = pd.concat(
    [out, pd.DataFrame([range_row, estimate_row], columns=out.columns)],
    ignore_index=True
)

out
