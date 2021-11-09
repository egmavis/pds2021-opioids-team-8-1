import pandas as pd
import numpy as np

population_death = pd.read_csv(
    "/Users/emeliamavis/720/pds2021-opioids-team-8-1/20_intermediate_files/Death_and_Population.csv"
)
shipments = pd.read_csv(
    "/Users/emeliamavis/720/pds2021-opioids-team-8-1/20_intermediate_files/Opioid_Final.csv"
)

# Rename shipments columns to match population_death
shipments.rename(
    {"BUYER_STATE": "State", "BUYER_COUNTY": "County", "YEAR": "Year"},
    axis=1,
    inplace=True,
)

# Drop unnecessary columns
shipments.drop(labels=["Unnamed: 0", "Unnamed: 0.1"], axis=1, inplace=True)

# Change County entries to sentence-case
counties = list(shipments.County)
new_counties = [i.capitalize() for i in counties]
shipments.County = new_counties

# Change types to match
shipments.Year = shipments.Year.astype("float64")

# Great, now can merge on state-county-year
merged = pd.merge(
    population_death,
    shipments,
    on=["Year", "State", "County"],
    validate="1:1",
    indicator=True,
)
