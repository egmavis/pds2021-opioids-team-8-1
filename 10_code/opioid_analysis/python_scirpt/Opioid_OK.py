# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np


# %%
# Read in the data
dfo = pd.read_csv(
    "/Users/weilianghu/Downloads/arcos-ok-statewide-itemized.csv.gz", compression="gzip"
)
dfo.head()


# %%
# Subset the data for the corresponding variables that we need
cols = [
    "BUYER_STATE",
    "BUYER_COUNTY",
    "MME_Conversion_Factor",
    "CALC_BASE_WT_IN_GM",
    "TRANSACTION_DATE",
]

dfo1 = dfo[cols]
dfo1


# %%
# Check for missing data
assert not dfo1["BUYER_STATE"].isnull().any()
assert not dfo1["BUYER_COUNTY"].isnull().any()
assert not dfo1["MME_Conversion_Factor"].isnull().any()
assert not dfo1["CALC_BASE_WT_IN_GM"].isnull().any()
assert not dfo1["TRANSACTION_DATE"].isnull().any()


# %%
# Change date variable to year
dfo1["TRANS_TIME"] = pd.to_datetime(dfo1["TRANSACTION_DATE"], format="%m%d%Y")
dfo1["YEAR"] = pd.DatetimeIndex(dfo1["TRANS_TIME"]).year
dfo1["MONTH"] = pd.DatetimeIndex(dfo1["TRANS_TIME"]).month
dfo1


# %%
# Group by County and year and calculate opioid quantity.
dfo_by_county = (
    dfo1.groupby(["BUYER_STATE", "BUYER_COUNTY", "YEAR", "MONTH"]).sum().reset_index()
)
dfo_by_county["MME_monthly"] = (
    dfo_by_county["CALC_BASE_WT_IN_GM"] * 1000 * dfo_by_county["MME_Conversion_Factor"]
)
dfo_by_county
final_columns = ["BUYER_STATE", "BUYER_COUNTY", "MONTH", "YEAR", "MME_monthly"]
dfo_final = dfo_by_county[final_columns]
dfo_final


# %%
dfo_final.to_csv("Opioid_OK.csv")
dfo_final.to_parquet("Opioid_OK.gzip", compression="gzip")
