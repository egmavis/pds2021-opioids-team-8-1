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

# adding columns for unique year-month x values and a one-word y value
data.Month = data.Month.astype("float64")
data["to_add"] = data.Month / 100
data["Date"] = data.Year + data.to_add
data["Rate"] = data["Monthly Shipment Rate Per Capita"]

# subset to target states
florida = data[data.State == "FL"]
pooled = data[
    (data["State"] == "LA") | (data["State"] == "AZ") | (data["State"] == "CO")
]

# break florida into pre and post subsets for pre-post analysis
flo_pre = florida[florida.Year < 2010]
flo_post = florida[florida.Year >= 2010]

# break pooled states into pre and post subsets for diff-in-diff analysis
pooled_pre = pooled[pooled.Year < 2010]
pooled_post = pooled[pooled.Year >= 2010]

# add vertical line for year = 2010 (policy change)
line = (
    alt.Chart(pd.DataFrame({"Date": [2010.01]}))
    .mark_rule(color="red")
    .encode(x="Date:Q")
)

# same arguments for each chart
yvar = "Rate"
xvar = "Date"
alpha = 0.05

"""
PRE-POST ANALYSIS
"""

"""
Florida Pre-Policy Chart
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
Florida Post-Policy Chart
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

"""
Pooled Pre-Policy Charts
"""
# Grid for predicted values
pooled_pre_x = pooled_pre.loc[pd.notnull(pooled_pre[yvar]), xvar]
pooled_pre_xmin = pooled_pre_x.min()
pooled_pre_xmax = pooled_pre_x.max()
pooled_pre_step = (pooled_pre_xmax - pooled_pre_xmin) / 100
pooled_pre_grid = np.arange(
    pooled_pre_xmin, pooled_pre_xmax + pooled_pre_step, pooled_pre_step
)
pooled_pre_predictions = pd.DataFrame({xvar: pooled_pre_grid})

# Fit model, get predictions
pooled_pre_model = smf.ols(f"{yvar} ~ {xvar}", data=pooled_pre).fit()
pooled_pre_model_predict = pooled_pre_model.get_prediction(pooled_pre_predictions[xvar])
pooled_pre_predictions[yvar] = pooled_pre_model_predict.summary_frame()["mean"]
pooled_pre_predictions[["ci_low", "ci_high"]] = pooled_pre_model_predict.conf_int(
    alpha=alpha
)

# Build chart
pooled_pre_reg = (
    alt.Chart(pooled_pre_predictions)
    .mark_line(color="green")
    .encode(x=xvar, y=alt.Y(yvar, scale=alt.Scale(zero=False)))
)
pooled_pre_ci = (
    alt.Chart(pooled_pre_predictions)
    .mark_errorband(color="green")
    .encode(
        x=xvar,
        y=alt.Y("ci_low", title=yvar),
        y2="ci_high",
    )
)

"""
Pooled Post-Policy Charts
"""
# Grid for predicted values
pooled_post_x = pooled_post.loc[pd.notnull(pooled_post[yvar]), xvar]
pooled_post_xmin = pooled_post_x.min()
pooled_post_xmax = pooled_post_x.max()
pooled_post_step = (pooled_post_xmax - pooled_post_xmin) / 100
pooled_post_grid = np.arange(
    pooled_post_xmin, pooled_post_xmax + pooled_post_step, pooled_post_step
)
pooled_post_predictions = pd.DataFrame({xvar: pooled_post_grid})

# Fit model, get predictions
pooled_post_model = smf.ols(f"{yvar} ~ {xvar}", data=pooled_post).fit()
pooled_post_model_predict = pooled_post_model.get_prediction(
    pooled_post_predictions[xvar]
)
pooled_post_predictions[yvar] = pooled_post_model_predict.summary_frame()["mean"]
pooled_post_predictions[["ci_low", "ci_high"]] = pooled_post_model_predict.conf_int(
    alpha=alpha
)

# Build chart
pooled_post_reg = (
    alt.Chart(pooled_post_predictions)
    .mark_line(color="green")
    .encode(x=xvar, y=alt.Y(yvar, scale=alt.Scale(zero=False)))
)
pooled_post_ci = (
    alt.Chart(pooled_post_predictions)
    .mark_errorband(color="green")
    .encode(
        x=xvar,
        y=alt.Y("ci_low", title=yvar),
        y2="ci_high",
    )
)

# put the charts together
flo_pre_chart = flo_pre_ci + flo_pre_reg
flo_post_chart = flo_post_ci + flo_post_reg
pooled_pre_chart = pooled_pre_ci + pooled_pre_reg
pooled_post_chart = pooled_post_ci + pooled_post_reg

"""
PRE-POST ANALYSIS
"""
pre_post_chart = alt.layer(flo_pre_chart, flo_post_chart, line).properties(
    title="Opioid Shipment Rate in Florida Before and After 2010 (Policy Change)"
)

"""
DIFFERENCE-IN-DIFFERENCE ANALYSIS
"""
flor_vs_pooled = alt.layer(
    flo_pre_chart, flo_post_chart, pooled_pre_chart, pooled_post_chart, line
).properties(
    title="Pre- and Post- Policy Opioid Shipment Rates In Flordia and Control States"
)


# saving charts as png files
save(
    pre_post_chart,
    "/Users/emeliamavis/720/pds2021-opioids-team-8-1/30_results/shipment/NEW_PLOTS_FOR_FINAL/florida_shipment_pre_post.png",
)

save(
    flor_vs_pooled,
    "/Users/emeliamavis/720/pds2021-opioids-team-8-1/30_results/shipment/NEW_PLOTS_FOR_FINAL/florida_shipment_vs_pooled.png",
)
