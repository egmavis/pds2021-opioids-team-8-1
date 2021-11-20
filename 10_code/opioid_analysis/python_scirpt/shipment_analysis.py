import pandas as pd
import numpy as np
import altair as alt
from altair_saver import save
import statsmodels.formula.api as smf

data = pd.read_csv(
    "~/720/pds2021-opioids-team-8-1/20_intermediate_files/Death_Population_Shipments.csv"
)

# double check that there are no duplicates and no missing values


assert ~data.duplicated(["Year", "Month", "State", "County"]).any()
assert ~data.isna().any().sum()

# subset to target states
florida = data[data.State == "FL"]
washington = data[data.State == "WA"]

arizona = data[data.State == "AZ"]
colorado = data[data.State == "CO"]
louisiana = data[data.State == "LA"]

# break washington into pre and post subsets for pre-post analysis
flo_pre = florida[florida.Year < 2010]
flo_post = florida[florida.Year >= 2010]

# add vertical line for year = 2012 (policy change)
line = (
    alt.Chart(pd.DataFrame({"Year": [2010]})).mark_rule(color="red").encode(x="Year:Q")
)

# same arguments for each chart
yvar = "MME_per_Capita_in_milligram"
xvar = "Year"
alpha = 0.05

"""
PRE-POST ANALYSIS
"""

"""
Pre-Policy Charts
"""
# Grid for predicted values
flo_pre_x = flo_pre.loc[pd.notnull(flo_pre[yvar]), xvar]
flo_pre_xmin = flo_pre_x.min()
flo_pre_xmax = flo_pre_x.max()
flo_pre_step = (flo_pre_xmax - flo_pre_xmin) / 100
flo_pre_grid = np.arange(flo_pre_xmin, flo_pre_xmax + flo_pre_step, flo_pre_step)
flo_pre_predictions = pd.DataFrame({xvar: flo_pre_grid})

# Fit model, get predictions
flo_pre_model = smf.ols(f"{yvar} ~ {xvar}", data=flo_pre).fit()
flo_pre_model_predict = flo_pre_model.get_prediction(flo_pre_predictions[xvar])
flo_pre_predictions[yvar] = flo_pre_model_predict.summary_frame()["mean"]
flo_pre_predictions[["ci_low", "ci_high"]] = flo_pre_model_predict.conf_int(alpha=alpha)

# Build chart
flo_pre_reg = (
    alt.Chart(flo_pre_predictions)
    .mark_line()
    .encode(x=xvar, y=alt.Y(yvar, scale=alt.Scale(zero=False)))
)
flo_pre_ci = (
    alt.Chart(flo_pre_predictions)
    .mark_errorband()
    .encode(
        x=xvar,
        y=alt.Y("ci_low", title=yvar),
        y2="ci_high",
    )
)

"""
Post-Policy Charts
"""
# Grid for predicted values
flo_post_x = flo_post.loc[pd.notnull(flo_post[yvar]), xvar]
flo_post_xmin = flo_post_x.min()
flo_post_xmax = flo_post_x.max()
flo_post_step = (flo_post_xmax - flo_post_xmin) / 100
flo_post_grid = np.arange(flo_post_xmin, flo_post_xmax + flo_post_step, flo_post_step)
flo_post_predictions = pd.DataFrame({xvar: flo_post_grid})

# Fit model, get predictions
flo_post_model = smf.ols(f"{yvar} ~ {xvar}", data=flo_post).fit()
flo_post_model_predict = flo_post_model.get_prediction(flo_post_predictions[xvar])
flo_post_predictions[yvar] = flo_post_model_predict.summary_frame()["mean"]
flo_post_predictions[["ci_low", "ci_high"]] = flo_post_model_predict.conf_int(
    alpha=alpha
)

# Build chart
flo_post_reg = (
    alt.Chart(flo_post_predictions)
    .mark_line()
    .encode(x=xvar, y=alt.Y(yvar, scale=alt.Scale(zero=False)))
)
flo_post_ci = (
    alt.Chart(flo_post_predictions)
    .mark_errorband()
    .encode(
        x=xvar,
        y=alt.Y("ci_low", title=yvar),
        y2="ci_high",
    )
)

pre_chart = flo_pre_ci + flo_pre_reg
post_chart = flo_post_ci + flo_post_reg

# pre-post analysis chart
pre_post_chart = alt.layer(pre_chart, post_chart, line).properties(
    title="Opioid Shipment Rate in Florida"
)
pre_post_chart


# saving charts as png files
save(
    pre_post_chart,
    "/Users/weilianghu/Downloads/pds2021-opioids-team-8-1/30_results/shipment_FL.png",
)
