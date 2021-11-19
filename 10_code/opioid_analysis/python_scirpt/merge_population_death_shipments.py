import pandas as pd
import numpy as np

population_death = pd.read_csv(
    "/Users/emeliamavis/720/pds2021-opioids-team-8-1/20_intermediate_files/Death_and_Population.csv"
)
shipments = pd.read_csv("/Users/emeliamavis/Downloads/Opioid_Final.csv")

# Rename shipments columns to match population_death
shipments = shipments.rename(
    {
        "BUYER_STATE": "State",
        "BUYER_COUNTY": "County",
        "YEAR": "Year",
        "MONTH": "Month",
    },
    axis=1,
).copy()

# Drop unnecessary columns
shipments = shipments.drop(labels=["Unnamed: 0", "Unnamed: 0.1"], axis=1).copy()

# Change County entries to sentence-case
counties = list(shipments.County)
new_counties = [i.capitalize() for i in counties]
shipments.County = new_counties

# Change types to match
shipments.Year = shipments.Year.astype("float64")

# Modify county observations to include "County"
population_death.County = [
    population_death.County[i].replace("Parish", "County")
    for i in range(len(population_death))
]

shipments["temp_county"] = ""
shipments.temp_county = [shipments.County[i] + " County" for i in range(len(shipments))]
shipments = shipments.drop(labels="County", axis=1).copy()
shipments = shipments.rename({"temp_county": "County"}, axis=1).copy()

# Great, now can merge on state-county-year
merged = pd.merge(
    shipments,
    population_death,
    on=["Year", "State", "County"],
    validate="m:1",
    indicator=True,
)

# last validity checks
assert ~merged.isna().any().sum()
assert ~merged.duplicated(["Year", "County", "State"]).any().sum()

# write output file
merged.to_csv(
    "~/720/pds2021-opioids-team-8-1/20_intermediate_files/Death_Population_Shipments.csv"
)
merged.to_parquet(
    "~/720/pds2021-opioids-team-8-1/20_intermediate_files/Death_Population_Shipments.gzip"
)
