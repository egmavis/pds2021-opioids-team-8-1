# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np


# %%
# Read in the data
dfl = pd.read_csv(
    "/Users/weilianghu/Downloads/arcos-la-statewide-itemized.csv.gz", compression="gzip"
)
dfl.head()


# %%
# Subset the data for the corresponding variables that we need
cols = [
    "BUYER_STATE",
    "BUYER_COUNTY",
    "MME_Conversion_Factor",
    "CALC_BASE_WT_IN_GM",
    "TRANSACTION_DATE",
]

dfl1 = dfl[cols]
dfl1


# %%
# Check for missing data
assert not dfl1["BUYER_STATE"].isnull().any()
assert not dfl1["BUYER_COUNTY"].isnull().any()
assert not dfl1["MME_Conversion_Factor"].isnull().any()
assert not dfl1["CALC_BASE_WT_IN_GM"].isnull().any()
assert not dfl1["TRANSACTION_DATE"].isnull().any()


# %%
# Change date variable to year
dfl1["TRANS_TIME"] = pd.to_datetime(dfl1["TRANSACTION_DATE"], format="%m%d%Y")
dfl1["YEAR"] = pd.DatetimeIndex(dfl1["TRANS_TIME"]).year
dfl1["MONTH"] = pd.DatetimeIndex(dfl1["TRANS_TIME"]).month
dfl1


# %%
# Group by County and year and calculate opioid quantity.
dfl_by_county = (
    dfl1.groupby(["BUYER_STATE", "BUYER_COUNTY", "YEAR", "MONTH"]).sum().reset_index()
)
dfl_by_county["MME_monthly"] = (
    dfl_by_county["CALC_BASE_WT_IN_GM"] * 1000 * dfl_by_county["MME_Conversion_Factor"]
)
dfl_by_county
final_columns = ["BUYER_STATE", "BUYER_COUNTY", "MONTH", "YEAR", "MME_monthly"]
dfl_final = dfl_by_county[final_columns]
dfl_final


# %%
dfl_final.to_csv("Opioid_LA.csv")
dfl_final.to_parquet("Opioid_LA.gzip", compression="gzip")
