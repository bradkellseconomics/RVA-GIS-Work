# %%
# Imports + load minimalist EDA file
import pandas as pd
from pathlib import Path
import numpy as np
path = Path("data/analysis/servicelines_house_with_attributes.csv")


# %%
import pandas as pd

# --- Normalization helpers ---
def normalize_status_material(val):
    if pd.isna(val):
        return "Unknown"
    s = str(val).strip().lower()

    # Lead or galvanized requiring replacement → Lead/GRR
    if "lead" == s or "galvanized requiring replacement" in s:
        return "Lead/GRR"

    # Any mention of unknown
    if "unknown" in s:
        return "Unknown"

    # Clear non-lead materials
    if (
        "non-lead" in s
        or "copper" in s
        or "plastic" in s
        or "other" in s
        or s == "galvanized"
    ):
        return "Non-Lead"

    # Default fallback
    return "Unknown"


def normalize_status_both(val):
    if pd.isna(val):
        return "Unknown"
    s = str(val).strip().lower()

    # Check for non-lead first to avoid catching "non-lead" as "lead"
    if "non-lead" in s:
        return "Non-Lead"

    # Lead or galvanized requiring replacement → Lead/GRR
    if "lead" in s or "galvanized requiring replacement" in s:
        return "Lead/GRR"

    return "Unknown"


# --- Main code ---
df = pd.read_csv(path)

# Convert timestamps if present
if "CreationDate" in df.columns:
    df["CreationDate_dt"] = pd.to_datetime(df["CreationDate"], unit="ms", errors="coerce")
if "EditDate" in df.columns:
    df["EditDate_dt"] = pd.to_datetime(df["EditDate"], unit="ms", errors="coerce")

# Normalize materials and statuses
if "custmaterial" in df.columns:
    df["custmaterial_cat"] = df["custmaterial"].apply(normalize_status_material)
if "utilmaterial" in df.columns:
    df["utilmaterial_cat"] = df["utilmaterial"].apply(normalize_status_material)
if "bothsidesstatus" in df.columns:
    df["bothsidesstatus_cat"] = df["bothsidesstatus"].apply(normalize_status_both)

# Print summaries
for col in ["custmaterial_cat", "utilmaterial_cat", "bothsidesstatus_cat"]:
    if col in df.columns:
        print(f"\nValue counts for {col}:")
        counts = df[col].value_counts(dropna=False)
        perc = (counts / len(df) * 100).round(1)
        print(pd.DataFrame({"count": counts, "percent": perc}))


# %%
# When was this file last updated? 
print("Data last edited on:", df["EditDate_dt"].max())
# Show the ten most recent edit dates
print("Ten most recent edit dates:")
print(df["EditDate_dt"].nlargest(10))

# %%
# Print value counts for pipe material variables

for col in ["custmaterial", "custmaterial_cat", "utilmaterial", "utilmaterial_cat", "bothsidesstatus","bothsidesstatus_cat"]:
    print(f"\nValue counts for {col}:")
    counts = df[col].value_counts(dropna=False)
    percents = df[col].value_counts(normalize=True, dropna=False) * 100
    result = pd.DataFrame({"count": counts, "percent": percents.round(1)})
    print(result)


# %%
# Print buckets for lead score by model classification for customer pipes

bins = (df.groupby("ModelClassification_Cust")["ModelScoreForLeadClassification_Cust"]
          .agg(["min","max","mean"]))
print(bins)


# %%
df.groupby("custmaterial_cat")[["ModelScoreForLeadClassification_Cust", "ModelClassification_Cust"]].agg(["min", "max", "mean"])


# %%
#Impute missing customer material categories based on model classification

# # Copy your clean category column
df["custmaterial_cat_imputed"] = df["custmaterial_cat"]

# Only impute for unknowns
mask = df["custmaterial_cat"] == "Unknown"

# Define mapping from model class to imputed category
class_map = {0: "Non-Lead", 1: "Non-Lead", 2: "Lead/GRR", 3: "Lead/GRR"}

df.loc[mask, "custmaterial_cat_imputed"] = (
    df.loc[mask, "ModelClassification_Cust"]
    .map(class_map)
    .fillna("Unknown")
)

#Impute missing utility material categories based on model classification

# # Copy your clean category column
df["utilmaterial_cat_imputed"] = df["utilmaterial_cat"]

# Only impute for unknowns
mask = df["utilmaterial_cat"] == "Unknown"

# Define mapping from model class to imputed category
class_map = {0: "Non-Lead", 1: "Non-Lead", 2: "Lead/GRR", 3: "Lead/GRR"}

df.loc[mask, "utilmaterial_cat_imputed"] = (
    df.loc[mask, "ModelClassification_City"]
    .map(class_map)
    .fillna("Unknown")
)

print()

def impute_bothsides(row):
    if "Lead/GRR" in (row["custmaterial_cat_imputed"], row["utilmaterial_cat_imputed"]):
        return "Lead/GRR"
    elif "Non-Lead" in (row["custmaterial_cat_imputed"], row["utilmaterial_cat_imputed"]):
        return "Non-Lead"
    else:
        return "Unknown"

df["bothsidesstatus_imputed"] = df.apply(impute_bothsides, axis=1)

# %%
# Comparing imputed vs original categories for lead pipes

for col in ["custmaterial_cat","custmaterial_cat_imputed", "utilmaterial_cat", "utilmaterial_cat_imputed","bothsidesstatus","bothsidesstatus_imputed"]:
    print(f"\nValue counts for {col}:")
    counts = df[col].value_counts(dropna=False)
    percents = df[col].value_counts(normalize=True, dropna=False) * 100
    result = pd.DataFrame({"count": counts, "percent": percents.round(1)})
    print(result)

# %%
#Using our three new imputed columns, custmaterial_cat_imputed, utilmaterial_cat_imputed, and bothsidesstatus_imputed   
# calculate the number and percent of lead pipes by neighborhood (neighborhood_name)


#custmaterial
lead_counts_custmaterial_imputed = df[df["custmaterial_cat_imputed"] == "Lead/GRR"].groupby("neighborhood_name").size()
total_counts = df.groupby("neighborhood_name").size()
lead_percents_custmaterial_imputed = (lead_counts_custmaterial_imputed / total_counts * 100).round(1)
non_lead_counts_custmaterial_imputed = df[df["custmaterial_cat_imputed"] == "Non-Lead"].groupby("neighborhood_name").size()
non_lead_percents_custmaterial_imputed = (non_lead_counts_custmaterial_imputed / total_counts * 100).round(1)

#utilmaterial
lead_counts_utilmaterial_imputed = df[df["utilmaterial_cat_imputed"] == "Lead/GRR"].groupby("neighborhood_name").size()
lead_percents_utilmaterial_imputed = (lead_counts_utilmaterial_imputed / total_counts * 100).round(1)
non_lead_counts_utilmaterial_imputed = df[df["utilmaterial_cat_imputed"] == "Non-Lead"].groupby("neighborhood_name").size()
non_lead_percents_utilmaterial_imputed = (non_lead_counts_utilmaterial_imputed / total_counts * 100).round(1)

#Bothsides
lead_counts_bothsides_imputed = df[df["bothsidesstatus_imputed"] == "Lead/GRR"].groupby("neighborhood_name").size()
lead_percents_bothsides_imputed = (lead_counts_bothsides_imputed / total_counts * 100).round(1)
non_lead_counts_bothsides_imputed = df[df["bothsidesstatus_imputed"] == "Non-Lead"].groupby("neighborhood_name").size()
non_lead_percents_bothsides_imputed = (non_lead_counts_bothsides_imputed / total_counts * 100).round(1)

lead_imputed_summary = pd.DataFrame({"total_count": total_counts, 
                                      "lead_count_bothsides_imputed": lead_counts_bothsides_imputed, "lead_percent_bothsides_imputed": lead_percents_bothsides_imputed,
                                      "lead_count_custmaterial_imputed": lead_counts_custmaterial_imputed, "lead_percent_custmaterial_imputed": lead_percents_custmaterial_imputed,
                                      "lead_count_utilmaterial_imputed": lead_counts_utilmaterial_imputed, "lead_percent_utilmaterial_imputed": lead_percents_utilmaterial_imputed,
                                      "non_lead_count_bothsides_imputed": non_lead_counts_bothsides_imputed, "non_lead_percent_bothsides_imputed": non_lead_percents_bothsides_imputed,
                                      "non_lead_count_custmaterial_imputed": non_lead_counts_custmaterial_imputed, "non_lead_percent_custmaterial_imputed": non_lead_percents_custmaterial_imputed,
                                      "non_lead_count_utilmaterial_imputed": non_lead_counts_utilmaterial_imputed, "non_lead_percent_utilmaterial_imputed": non_lead_percents_utilmaterial_imputed})
lead_imputed_summary = lead_imputed_summary.fillna(0)
print(lead_imputed_summary)


#Output to CSV
output_path = "data/analysis/neighborhoods_lead_imputed_summary.csv"
lead_imputed_summary.to_csv(output_path)


# %%
# Output df as CSV
output_path = "data/analysis/servicelines_with_imputed_materials.csv"
df.to_csv(output_path, index=False)


