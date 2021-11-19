# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np


# %%
# Read in the data
dfw = pd.read_csv(
    "/Users/weilianghu/Downloads/arcos-wa-statewide-itemized.csv.gz", compression="gzip"
)
dfw.head()


# %%
# Subset the data for the corresponding variables that we need
cols = [
    "BUYER_STATE",
    "BUYER_COUNTY",
    "MME_Conversion_Factor",
    "CALC_BASE_WT_IN_GM",
    "TRANSACTION_DATE",
]

dfw1 = dfw[cols]
dfw1


# %%
# Check for missing data
assert not dfw1["BUYER_STATE"].isnull().any()
assert not dfw1["BUYER_COUNTY"].isnull().any()
assert not dfw1["MME_Conversion_Factor"].isnull().any()
assert not dfw1["CALC_BASE_WT_IN_GM"].isnull().any()
assert not dfw1["TRANSACTION_DATE"].isnull().any()


# %%
# Change date variable to year
dfw1["TRANS_TIME"] = pd.to_datetime(dfw1["TRANSACTION_DATE"], format="%m%d%Y")
dfw1["YEAR"] = pd.DatetimeIndex(dfw1["TRANS_TIME"]).year
dfw1["MONTH"] = pd.DatetimeIndex(dfw1["TRANS_TIME"]).month
dfw1


# %%
# Group by County and year and calculate opioid quantity.
dfw_by_county = (
    dfw1.groupby(["BUYER_STATE", "BUYER_COUNTY", "YEAR", "MONTH"]).sum().reset_index()
)
dfw_by_county["MME_monthly"] = (
    dfw_by_county["CALC_BASE_WT_IN_GM"] * 1000 * dfw_by_county["MME_Conversion_Factor"]
)
dfw_by_county
final_columns = ["BUYER_STATE", "BUYER_COUNTY", "MONTH", "YEAR", "MME_monthly"]
dfw_final = dfw_by_county[final_columns]
dfw_final


# %%
dfw_final.to_csv("Opioid_WA.csv")
dfw_final.to_parquet("Opioid_WA.gzip", compression="gzip")
