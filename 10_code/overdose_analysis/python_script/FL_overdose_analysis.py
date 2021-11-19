import pandas as pd
import numpy as np
import altair as alt
import statsmodels.formula.api as smf
import pathlib

from TX_overdose_analysis import get_vertical_line, pre_post_analysis


if __name__ == "__main__":
    data = pd.read_parquet(f"{pathlib.Path.cwd()}/20_intermediate_files/Death_and_Population.gzip")
    data["Overdose_death_per_capita"] = data["Deaths"] / data["Population"]

    # Select target states
    FL = data[data.State == "FL"] # Florida
    AZ = data[data.State == "AZ"] # Arizona
    LA = data[data.State == "LA"] # Louisiana
    CO = data[data.State == "CO"] # Colorado

    # Where to save files
    path_prefix = f"{pathlib.Path.cwd()}/30_results/overdose_death/"

    # Pre-post analysis
    year = 2010
    line = get_vertical_line(year)
    FL_pre, FL_post = pre_post_analysis(FL, year, "blue")
    FL_pre_post_chart = alt.layer(FL_pre, FL_post, line).properties(
        title="Overdose Death Rates in Florida"
    )
    FL_pre_post_chart.save(f"{path_prefix}FL.png")

    # The diff-in-diff charts between each comparison state
    # Texas vs. Arizona
    ref_color = "green"
    AZ_pre, AZ_post = pre_post_analysis(AZ, year, ref_color)
    FL_vs_AZ = alt.layer(FL_pre_post_chart, AZ_pre + AZ_post, line).properties(
        title="Overdose Death Rates in Florida vs. Arizona"
    )
    FL_vs_AZ.save(f"{path_prefix}FL_vs_AZ.png")

    # Texas vs. Louisiana
    LA_pre, LA_post = pre_post_analysis(LA, year, ref_color)
    FL_vs_LA = alt.layer(FL_pre_post_chart, LA_pre + LA_post, line).properties(
        title="Overdose Death Rates in Florida vs. Louisiana"
    )
    FL_vs_LA.save(f"{path_prefix}FL_vs_LA.png")

    # Texas vs. Colorado
    CO_pre, CO_post = pre_post_analysis(CO, year, ref_color)
    FL_vs_CO = alt.layer(FL_pre_post_chart, CO_pre + CO_post, line).properties(
        title="Overdose Death Rates in Florida vs. Colorado"
    )
    FL_vs_CO.save(f"{path_prefix}FL_vs_CO.png")