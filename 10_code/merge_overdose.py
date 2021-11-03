import pandas as pd
import pathlib


def read_all_files():
    data_list = []
    for year in range(2003, 2016):
        df_temp = pd.read_csv(f"{pathlib.Path.cwd()}/00_source_data/US_VitalStatistics/Underlying Cause of Death, {year}.txt", delimiter="\t")
        data_list.append(df_temp)
    return pd.concat(data_list)


def basic_data_validity_check(df):
    assert not df["County"].isnull().any()
    assert not df["Year"].isnull().any()
    assert not df["Drug/Alcohol Induced Cause"].isnull().any()
    assert not df["Deaths"].isnull().any()
    assert not (df["Deaths"] == 0).any()


# If we need to change the target states in the future, change this method
def check_state(county):
    target_list = ["Fl", "TX", "WA", "AZ", "LA", "CO", "WI", "MS", "KS", "OK"]
    state = county.split(", ")[1]
    return 1 if state in target_list else 0


if __name__ == "__main__":
    # Read all overdose datafile from the folder
    df = read_all_files()

    # Remove abnormal values from data scripting
    df = df.drop(df[df["County"].isnull()].index)

    # Basic Data Validity Check
    basic_data_validity_check(df)

    # Subset: only keep target states
    df["target_state"] = df.apply(lambda row: check_state(row.County), axis=1)
    df_sub = df[df["target_state"] == 1]

    # Write to output files
    df_sub.to_csv(f"{pathlib.Path.cwd()}/20_intermediate_files/Underlying Cause of Death, 2003-2015.csv")
    df_sub.to_parquet(f"{pathlib.Path.cwd()}/20_intermediate_files/Underlying Cause of Death, 2003-2015.gzip", compression="gzip")