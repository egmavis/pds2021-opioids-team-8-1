# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np


# %%
# Read in the data
dfa = pd.read_csv(
    "/Users/weilianghu/Downloads/arcos-az-statewide-itemized.csv.gz", compression="gzip"
)
dfa.head()


# %%
# Subset the data for the corresponding variables that we need
cols = [
    "BUYER_STATE",
    "BUYER_COUNTY",
    "MME_Conversion_Factor",
    "CALC_BASE_WT_IN_GM",
    "TRANSACTION_DATE",
]

dfa1 = dfa[cols]
dfa1


# %%
# Check for missing data
assert not dfa1["BUYER_STATE"].isnull().any()
assert not dfa1["BUYER_COUNTY"].isnull().any()
assert not dfa1["MME_Conversion_Factor"].isnull().any()
assert not dfa1["CALC_BASE_WT_IN_GM"].isnull().any()
assert not dfa1["TRANSACTION_DATE"].isnull().any()


# %%
# Change date variable to year
dfa1["TRANS_TIME"] = pd.to_datetime(dfa1["TRANSACTION_DATE"], format="%m%d%Y")
dfa1["YEAR"] = pd.DatetimeIndex(dfa1["TRANS_TIME"]).year
dfa1["MONTH"] = pd.DatetimeIndex(dfa1["TRANS_TIME"]).month
dfa1.drop(columns=["TRANSACTION_DATE"], axis=1)
dfa1


# %%
# Group by County and year and calculate opioid quantity.
dfa_by_county = (
    dfa1.groupby(["BUYER_STATE", "BUYER_COUNTY", "YEAR", "MONTH"]).sum().reset_index()
)
dfa_by_county["MME_monthly"] = (
    dfa_by_county["CALC_BASE_WT_IN_GM"] * 1000 * dfa_by_county["MME_Conversion_Factor"]
)
dfa_by_county
final_columns = ["BUYER_STATE", "BUYER_COUNTY", "MONTH", "YEAR", "MME_monthly"]
dfa_final = dfa_by_county[final_columns]
dfa_final


# %%
dfa_final.to_csv("Opioid_AZ.csv")
dfa_final.to_parquet("Opioid_AZ.gzip", compression="gzip")
