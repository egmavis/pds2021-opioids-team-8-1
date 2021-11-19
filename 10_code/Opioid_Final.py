# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd


# %%
# Concatenate the four datasets that we need
states = ["AZ", "WA", "CO", "FL", "LA", "OK"]


def read_all_files():
    data_list = []
    for state in states:
        df_temp = pd.read_csv(
            f"/Users/weilianghu/Downloads/pds2021-opioids-team-8-1/20_intermediate_files/Opioid_{state}.csv"
        )
        data_list.append(df_temp)
    return pd.concat(data_list)


# %%
df = read_all_files()
df.sample(10)


# %%
df.to_csv("Opioid_Final.csv")
df.to_parquet("Opioid_Final.gzip", compression="gzip")
