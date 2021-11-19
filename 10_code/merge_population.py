import pandas as pd
import pathlib


def move_first_row_to_header(population_dataframe):
    df = population_dataframe.copy()
    new_header = df.iloc[0]
    df = df[1:]
    new_header[0] = "0"
    new_header.astype(int)
    new_header = [int(ele) for ele in new_header]
    new_header[0] = "County"
    df.columns = new_header
    return df


# This function only applies for data format of year 2000-2009
def structure_modification_for_old_xls(population_dataframe, state_name):

    df = population_dataframe.copy()
    # Drop unecessary columns and rows
    df.drop(df.columns[[1, 12, 13]], axis=1, inplace=True)
    df.drop(df.index[[0, 1, 3]], inplace=True)

    # Make row0 the header with a little modification tricks
    df = move_first_row_to_header(df)

    # Reset Index
    df.reset_index(inplace=True)
    df.drop("index", axis=1, inplace=True)

    # Some finetune
    df["County"] = df["County"].str[1:]
    df["State"] = state_name
    df = df.drop(df[df[2000].isnull()].index)

    return df


# This function only applies for data format of year 2010-2019
def structure_modification_for_new_xlsx(population_dataframe, state_name):

    df = population_dataframe.copy()

    # Drop unecessary columns and rows
    df.drop(df.columns[[1, 2]], axis=1, inplace=True)
    df.drop(df.index[[0, 1, 3]], inplace=True)

    # Make row0 the header with a little modification tricks
    df = move_first_row_to_header(df)

    # Reset Index
    df.reset_index(inplace=True)
    df.drop("index", axis=1, inplace=True)

    # Some finetune
    df["County"] = df["County"].str.split(", ").str[0]
    df["County"] = df["County"].str[1:]
    df["State"] = state_name
    df = df.drop(df[df[2010].isnull()].index)

    return df


def validity_check(df):
    for column in df.columns:
        assert not df[column].isnull().any()


if __name__ == "__main__":
    # Read population data for all target states
    state_list = ["FL", "TX", "WA", "AZ", "LA", "CO", "WI", "MS", "KS", "OK"]

    old_xls_list = []
    new_xlsx_list = []
    curr_path = pathlib.Path.cwd()
    for state in state_list:
        # Read data from 2000 to 2009
        old_state_population = pd.read_excel(f"{curr_path}/00_source_data/Population/{state}_population_2000_2009.xls")
        old_state_population_modified = structure_modification_for_old_xls(old_state_population, state)
        old_xls_list.append(old_state_population_modified)
        # Read data from 2010 to 2019
        new_state_population = pd.read_excel(f"{curr_path}/00_source_data/Population/{state}_population_2010_2019.xlsx")
        new_state_population_modified = structure_modification_for_new_xlsx(new_state_population, state)
        new_xlsx_list.append(new_state_population_modified)

    popuplation_2000 = pd.concat(old_xls_list)
    popuplation_2010 = pd.concat(new_xlsx_list)

    validity_check(popuplation_2000)
    validity_check(popuplation_2010)

    # Merge population from 2000 to 2019
    merged_population = popuplation_2000.merge(popuplation_2010, validate="1:1", indicator=True)
    assert merged_population[merged_population["_merge"] != "both"].empty
    merged_population.columns = merged_population.columns.astype(str)
    
    # Write output files
    merged_population.to_parquet(f"{curr_path}/20_intermediate_files/Population_2000-2019.gzip", compression="gzip")
    merged_population.to_csv(f"{curr_path}/20_intermediate_files/Population_2000-2019.csv")