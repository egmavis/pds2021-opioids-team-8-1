# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np


# %%
# Read in the data
dfc = pd.read_csv(
    "/Users/weilianghu/Downloads/arcos-co-statewide-itemized.csv.gz", compression="gzip"
)
dfc.head()


# %%
# Subset the data for the corresponding variables that we need
cols = [
    "BUYER_STATE",
    "BUYER_COUNTY",
    "MME_Conversion_Factor",
    "CALC_BASE_WT_IN_GM",
    "TRANSACTION_DATE",
]

dfc1 = dfc[cols]
dfc1


# %%
# Check for missing data
assert not dfc1["BUYER_STATE"].isnull().any()
assert not dfc1["BUYER_COUNTY"].isnull().any()
assert not dfc1["MME_Conversion_Factor"].isnull().any()
assert not dfc1["CALC_BASE_WT_IN_GM"].isnull().any()
assert not dfc1["TRANSACTION_DATE"].isnull().any()


# %%
# Change date variable to year
dfc1["YEAR"] = pd.to_datetime(dfc1["TRANSACTION_DATE"], format="%m%d%Y")
dfc1["YEAR"] = pd.DatetimeIndex(dfc1["YEAR"]).year

dfc1


# %%
# Group by County and year and calculate opioid quantity.
dfc_by_county = (
    dfc1.groupby(["BUYER_STATE", "BUYER_COUNTY", "YEAR"]).sum().reset_index()
)
dfc_by_county["MME"] = (
    dfc_by_county["CALC_BASE_WT_IN_GM"] * 1000 * dfc_by_county["MME_Conversion_Factor"]
)
dfc_by_county
final_columns = ["BUYER_STATE", "BUYER_COUNTY", "YEAR", "MME"]
dfc_final = dfc_by_county[final_columns]
dfc_final


# %%
dfc_final.to_csv("Opioid_CO.csv")
dfc_final.to_parquet("Opioid_CO.gzip", compression="gzip")
