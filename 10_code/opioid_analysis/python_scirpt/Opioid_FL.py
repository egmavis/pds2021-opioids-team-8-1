# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np


# %%
# Read in the data
dff = pd.read_csv(
    "/Users/weilianghu/Downloads/arcos-fl-statewide-itemized.csv.gz", compression="gzip"
)
dff.head()


# %%
# After exploring the data, we only keep the number of columns that we care about.
cols = [
    "BUYER_STATE",
    "BUYER_COUNTY",
    "MME_Conversion_Factor",
    "CALC_BASE_WT_IN_GM",
    "TRANSACTION_DATE",
]

dff1 = dff[cols]
dff1


# %%
# Drop data that are invalid
dff1 = dff1.drop(dff1[dff1["BUYER_COUNTY"].isnull()].index)


# %%
# Check the validity of data
assert not dff1["BUYER_STATE"].isnull().any()
assert not dff1["BUYER_COUNTY"].isnull().any()
assert not dff1["MME_Conversion_Factor"].isnull().any()
assert not dff1["CALC_BASE_WT_IN_GM"].isnull().any()
assert not dff1["TRANSACTION_DATE"].isnull().any()


# %%
dff1["TRANSACTION_DATE"].isna().sum()


# %%
# Change date to year and month
dff1["TRANS_TIME"] = pd.to_datetime(dff1["TRANSACTION_DATE"], format="%m%d%Y")
dff1["YEAR"] = pd.DatetimeIndex(dff1["TRANS_TIME"]).year
dff1["MONTH"] = pd.DatetimeIndex(dff1["TRANS_TIME"]).month
dff1.drop(columns=["TRANSACTION_DATE"], axis=1)
dff1


# %%
# Group by County and year and month and calculate opioid quantity.
dff_by_county = (
    dff1.groupby(["BUYER_STATE", "BUYER_COUNTY", "YEAR", "MONTH"]).sum().reset_index()
)
# MME is the equivalent quantity of opioid shipped that we calculate.
dff_by_county["MME_yearly"] = (
    dff_by_county["CALC_BASE_WT_IN_GM"] * 1000 * dff_by_county["MME_Conversion_Factor"]
)
dff_by_county
final_columns = ["BUYER_STATE", "BUYER_COUNTY", "MONTH", "YEAR", "MME_yearly"]
dff_final = dff_by_county[final_columns]
dff_final


# %%
dff_final.to_csv("Opioid_FL.csv")
dff_final.to_parquet("Opioid_FL.gzip", compression="gzip")
