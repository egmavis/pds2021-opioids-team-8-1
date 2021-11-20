from altair.vegalite.v4.schema.channels import Order
import pandas as pd
import numpy as np
import pathlib

# read in files to merge
population = pd.read_csv(
    "~/720/pds2021-opioids-team-8-1/20_intermediate_files/Population_2000-2019.csv"
)
cause_of_death = pd.read_csv(
    "~/720/pds2021-opioids-team-8-1/20_intermediate_files/Underlying Cause of Death, 2003-2015.csv"
)

# extract state from county column in cause_of_death
cause_of_death["State"] = ""
cause_of_death.State = [
    cause_of_death.County[i][-2:] for i in range(len(cause_of_death.County))
]

# extract county from county column in cause_of_death
cause_of_death["temp_County"] = ""
cause_of_death["temp_County"] = [
    cause_of_death.County[i][:-4] for i in range(len(cause_of_death.County))
]
cause_of_death.drop(
    labels=["County", "Unnamed: 0", "Notes", "Year Code"], axis=1, inplace=True
)
cause_of_death.rename({"temp_County": "County"}, axis=1, inplace=True)

# melt year headers into a Year column in population
population.drop(labels=["Unnamed: 0", "_merge"], axis=1, inplace=True)
population_melted = population.melt(
    id_vars=["County", "State"],
    value_vars=population.filter(like="20").columns.tolist(),
    var_name="Year",
    value_name="Population",
)
population_melted.Year = population_melted.Year.astype(np.float64)

# merge population_melted and cause_of_death
merged = cause_of_death.merge(
    population_melted,
    how="outer",
    on=["Year", "State", "County"],
    validate="m:1",
    indicator=True,
)


# subset to just D1, D2, D4 drug overdose deaths and missing values
overdose_deaths = merged[
    (merged["Drug/Alcohol Induced Cause Code"] == "D1")
    | (merged["Drug/Alcohol Induced Cause Code"] == "D2")
    | (merged["Drug/Alcohol Induced Cause Code"] == "D4")
    | (merged["Drug/Alcohol Induced Cause Code"].isnull())
].copy()

# remove everything later than 2015
overdose_deaths = overdose_deaths[overdose_deaths.Year <= 2015]

# calculate death rate
overdose_deaths["Death_Rate"] = (
    overdose_deaths["Deaths"] / overdose_deaths["Population"] * 100
)

# grab averages for each state-year to place into counties with missing values
overdose_deaths["Interpolated_Rate"] = overdose_deaths.groupby(["State", "Year"])[
    "Death_Rate"
].transform(np.mean)

# replace missing values with average death rate for that state-year
overdose_deaths.loc[
    overdose_deaths.Death_Rate.isnull(), "Death_Rate"
] = overdose_deaths.loc[overdose_deaths.Death_Rate.isnull(), "Interpolated_Rate"]


# total all drug overdose deaths for each state-county-year
overdose_deaths["Rate"] = overdose_deaths.groupby(["State", "County", "Year"])[
    "Death_Rate"
].transform(np.sum)


overdose_deaths.drop_duplicates(
    subset=["Year", "State", "County", "Rate"], inplace=True
)

# validity check for duplicates
assert ~overdose_deaths.duplicated(["Year", "State", "County"]).any()

# drop remaining unnecessary columns
overdose_deaths.drop(
    labels=[
        "Drug/Alcohol Induced Cause",
        "Drug/Alcohol Induced Cause Code",
        "Deaths",
        "_merge",
        "Death_Rate",
        "Interpolated_Rate",
    ],
    axis=1,
    inplace=True,
)

overdose_deaths.drop(labels=["County Code", "target_state"], axis=1, inplace=True)

overdose_deaths.isnull().any()

# write output file
overdose_deaths.to_csv(
    "~/720/pds2021-opioids-team-8-1/20_intermediate_files/Death_and_Population.csv"
)
overdose_deaths.to_parquet(
    "~/720/pds2021-opioids-team-8-1/20_intermediate_files/Death_and_Population.gzip"
)
