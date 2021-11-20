import pandas as pd
import numpy as np
import altair as alt
from altair_saver import save
import statsmodels.formula.api as smf

data = pd.read_csv(
    "~/720/pds2021-opioids-team-8-1/20_intermediate_files/Death_and_Population.csv"
)

# double check that there are no duplicates and no missing values
assert ~data.duplicated(["Year", "State", "County"]).any()
assert ~data.isna().any().sum()

# subset to target states
flor = data[data.State == "FL"]
pooled = data[
    (data["State"] == "LA") | (data["State"] == "AZ") | (data["State"] == "CO")
]

# break washington into pre and post subsets for pre-post and diff-in-diff analyses
flor_pre = flor[flor.Year < 2010]
flor_post = flor[flor.Year >= 2010]

# break pooled states into pre and post subsets for diff-in-diff analysis
pooled_pre = pooled[pooled.Year < 2010]
pooled_post = pooled[pooled.Year >= 2010]

# add vertical line for year = 2012 (policy change)
line = (
    alt.Chart(pd.DataFrame({"Year": [2010]})).mark_rule(color="red").encode(x="Year:Q")
)

# same arguments for each chart
yvar = "Rate"
xvar = "Year"
alpha = 0.05

"""
PRE-POST CHART BUILDING
"""

"""
Florida Pre-Policy Charts
"""
# Grid for predicted values
flor_pre_x = flor_pre.loc[pd.notnull(flor_pre[yvar]), xvar]
flor_pre_xmin = flor_pre_x.min()
flor_pre_xmax = flor_pre_x.max()
flor_pre_step = (flor_pre_xmax - flor_pre_xmin) / 100
flor_pre_grid = np.arange(flor_pre_xmin, flor_pre_xmax + flor_pre_step, flor_pre_step)
flor_pre_predictions = pd.DataFrame({xvar: flor_pre_grid})

# Fit model, get predictions
flor_pre_model = smf.ols(f"{yvar} ~ {xvar}", data=flor_pre).fit()
flor_pre_model_predict = flor_pre_model.get_prediction(flor_pre_predictions[xvar])
flor_pre_predictions[yvar] = flor_pre_model_predict.summary_frame()["mean"]
flor_pre_predictions[["ci_low", "ci_high"]] = flor_pre_model_predict.conf_int(
    alpha=alpha
)

# Build chart
flor_pre_reg = (
    alt.Chart(flor_pre_predictions)
    .mark_line()
    .encode(x=xvar, y=alt.Y(yvar, scale=alt.Scale(zero=False)))
)
flor_pre_ci = (
    alt.Chart(flor_pre_predictions)
    .mark_errorband()
    .encode(
        x=xvar,
        y=alt.Y("ci_low", title=yvar),
        y2="ci_high",
    )
)

"""
Florida Post-Policy Charts
"""
# Grid for predicted values
flor_post_x = flor_post.loc[pd.notnull(flor_post[yvar]), xvar]
flor_post_xmin = flor_post_x.min()
flor_post_xmax = flor_post_x.max()
flor_post_step = (flor_post_xmax - flor_post_xmin) / 100
flor_post_grid = np.arange(
    flor_post_xmin, flor_post_xmax + flor_post_step, flor_post_step
)
flor_post_predictions = pd.DataFrame({xvar: flor_post_grid})

# Fit model, get predictions
flor_post_model = smf.ols(f"{yvar} ~ {xvar}", data=flor_post).fit()
flor_post_model_predict = flor_post_model.get_prediction(flor_post_predictions[xvar])
flor_post_predictions[yvar] = flor_post_model_predict.summary_frame()["mean"]
flor_post_predictions[["ci_low", "ci_high"]] = flor_post_model_predict.conf_int(
    alpha=alpha
)

# Build chart
flor_post_reg = (
    alt.Chart(flor_post_predictions)
    .mark_line()
    .encode(x=xvar, y=alt.Y(yvar, scale=alt.Scale(zero=False)))
)
flor_post_ci = (
    alt.Chart(flor_post_predictions)
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
flor_pre_chart = flor_pre_ci + flor_pre_reg
flor_post_chart = flor_post_ci + flor_post_reg
pooled_pre_chart = pooled_pre_ci + pooled_pre_reg
pooled_post_chart = pooled_post_ci + pooled_post_reg


"""
PRE-POST ANALYSIS
"""
pre_post_chart = alt.layer(flor_pre_chart, flor_post_chart, line).properties(
    title="Overdose Deaths in Florida Before and After 2010 (Policy Change)"
)

"""
DIFFERENCE-IN-DIFFERENCE ANALYSIS
"""
flor_vs_pooled = alt.layer(
    flor_pre_chart, flor_post_chart, pooled_pre_chart, pooled_post_chart, line
).properties(title="Pre- and Post- Policy Trends In Flordia and Control States")


# saving charts as png files
save(
    pre_post_chart,
    "/Users/emeliamavis/720/pds2021-opioids-team-8-1/30_results/overdose_death/NEW_PLOTS_FOR_FINAL/flor_pre_post.png",
)
save(
    flor_vs_pooled,
    "/Users/emeliamavis/720/pds2021-opioids-team-8-1/30_results/overdose_death/NEW_PLOTS_FOR_FINAL/flor_vs_pooled.png",
)
