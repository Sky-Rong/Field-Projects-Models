import pandas as pd, numpy as np

# Data
sysv = str(xl("B4")).upper()
if sysv.startswith("GIS"):
    df = xl("tblGISProjects[#All]", headers=True)
else:
    df = xl("tblAISProjects[#All]", headers=True)

# New job
job = {
    "system": xl("B4"),
    "voltage": xl("B5"),
    "length": xl("B6"),
    "circuits": xl("B7"),
    "joints": xl("B8"),
    "terms": xl("B9")
}

# Weights with on/off switch
wt = xl("D5:F10", headers=False)
W = {}
for row in wt.values:
    p = str(row[0]).strip().lower()
    on = str(row[2]).strip().lower() in ("on", "yes", "true", "1")
    W[p] = float(row[1]) if on else 0.0

# Clean
cols = {
    "System": "system",
    "Rated (kV)": "voltage",
    "Total ft": "length",
    "Circuits": "circuits",
    "Joints": "joints",
    "Terms": "terms",
    "Value ($)": "value"
}

keep = {k: v for k, v in cols.items() if k in df.columns}
df = df.rename(columns=keep)
name_col = "Project Name" if "Project Name" in df.columns else df.columns[0]
use = [
    c for c in ["system", "voltage", "length", "circuits", "joints", "terms", "value"]
    if c in df.columns
]
df = df[[name_col] + use].copy()

for c in [c for c in use if c != "system"]:
    df[c] = pd.to_numeric(
        df[c]
        .astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.extract(r"(-?\d+\.?\d*)")[0],
        errors="coerce"
    )

df["system"] = df["system"].astype(str).str.upper().str.strip()

# Similarity
def is_num(x):
    try:
        float(x)
        return True
    except:
        return False

num = [
    f for f in ["voltage", "length", "circuits", "joints", "terms"]
    if W.get(f, 0) > 0 and f in df.columns and is_num(job[f])
]

rng = {
    f: (df[f].min(), max(df[f].max(), df[f].min() + 1))
    for f in num
}

nrm = lambda v, f: (v - rng[f][0]) / (rng[f][1] - rng[f][0])

dist = W.get("system", 0) * (
    df["system"] != str(job["system"]).upper()
).astype(float)

for f in num:
    dist = dist + W[f] * (
        df[f].apply(lambda v: nrm(v, f)) - nrm(float(job[f]), f)
    ).abs()

# Match %
active_w = W.get("system", 0) + sum(W[f] for f in num)
if active_w <= 0:
    active_w = 1

df["Match %"] = (100 * (1 - dist / active_w)).clip(lower=0).round(1)
df["dist"] = dist

# Output
# A12 stores how many results
raw = str(xl("A12")).strip().lower()
if raw == "all":
    n = len(df)
elif raw in ("", "none", "nan"):
    n = 10
else:
    try:
        n = int(float(raw))
    except ValueError:
        n = 10

out = df.sort_values("dist").head(n).reset_index(drop=True)
out.insert(0, "Rank", range(1, len(out) + 1))

out = out[
    ["Rank", name_col, "system", "voltage", "joints", "terms", "value", "Match %"]
].copy()

out = out.rename(columns={
    name_col: "Project Name",
    "system": "System",
    "voltage": "Rated kV",
    "joints": "Joints",
    "terms": "Terms",
    "value": "Value ($)"
})

out["Value ($)"] = out["Value ($)"].apply(
    lambda value: "" if pd.isna(value) else f"${value:,.0f}"
)

out["Match %"] = out["Match %"].apply(
    lambda value: "" if pd.isna(value) else f"{value:.1f}%"
)

out
