import pandas as pd
import numpy as np

death = pd.read_csv(
    "~/720/pds2021-opioids-team-8-1/30_results/summary_stats/Death_and_Population.csv"
)
shipment = pd.read_csv(
    "~/720/pds2021-opioids-team-8-1/30_results/summary_stats/Death_Population_Shipments.csv"
)

# stats for overdose death

# WA and pooled states
death[death.State == "WA"]["Death Rate By Overdose"].describe()
death[(death.State == "AZ") | (death.State == "OK") | (death.State == "CO")][
    "Death Rate By Overdose"
].describe()

# FL and pooled states
death[death.State == "FL"]["Death Rate By Overdose"].describe()
death[(death.State == "AZ") | (death.State == "LA") | (death.State == "CO")][
    "Death Rate By Overdose"
].describe()

# TX and pooled states
death[death.State == "TX"]["Death Rate By Overdose"].describe()
death[(death.State == "WI") | (death.State == "MS") | (death.State == "KS")][
    "Death Rate By Overdose"
].describe()


# stats for shipments

# FL and pooled states
shipment[shipment.State == "FL"]["Monthly Shipment Rate Per Capita"].describe()
shipment[
    (shipment.State == "AZ") | (shipment.State == "LA") | (shipment.State == "CO")
]["Monthly Shipment Rate Per Capita"].describe()

# WA and pooled states
shipment[shipment.State == "WA"]["Monthly Shipment Rate Per Capita"].describe()
shipment[
    (shipment.State == "AZ") | (shipment.State == "OK") | (shipment.State == "CO")
]["Monthly Shipment Rate Per Capita"].describe()
