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
wash = data[data.State == "WA"]
pooled = data[
    (data["State"] == "OK") | (data["State"] == "AZ") | (data["State"] == "CO")
]

# break washington into pre and post subsets for pre-post and diff-in-diff analyses
wash_pre = wash[wash.Year < 2012]
wash_post = wash[wash.Year >= 2012]

# break pooled states into pre and post subsets for diff-in-diff analysis
pooled_pre = pooled[pooled.Year < 2012]
pooled_post = pooled[pooled.Year >= 2012]

# add vertical line for year = 2012 (policy change)
line = (
    alt.Chart(pd.DataFrame({"Year": [2012]})).mark_rule(color="red").encode(x="Year:Q")
)

# same arguments for each chart
yvar = "Rate"
xvar = "Year"
alpha = 0.05

"""
PRE-POST CHART BUILDING
"""

"""
Washington Pre-Policy Charts
"""
# Grid for predicted values
wash_pre_x = wash_pre.loc[pd.notnull(wash_pre[yvar]), xvar]
wash_pre_xmin = wash_pre_x.min()
wash_pre_xmax = wash_pre_x.max()
wash_pre_step = (wash_pre_xmax - wash_pre_xmin) / 100
wash_pre_grid = np.arange(wash_pre_xmin, wash_pre_xmax + wash_pre_step, wash_pre_step)
wash_pre_predictions = pd.DataFrame({xvar: wash_pre_grid})

# Fit model, get predictions
wash_pre_model = smf.ols(f"{yvar} ~ {xvar}", data=wash_pre).fit()
wash_pre_model_predict = wash_pre_model.get_prediction(wash_pre_predictions[xvar])
wash_pre_predictions[yvar] = wash_pre_model_predict.summary_frame()["mean"]
wash_pre_predictions[["ci_low", "ci_high"]] = wash_pre_model_predict.conf_int(
    alpha=alpha
)

# Build chart
wash_pre_reg = (
    alt.Chart(wash_pre_predictions)
    .mark_line()
    .encode(x=xvar, y=alt.Y(yvar, scale=alt.Scale(zero=False)))
)
wash_pre_ci = (
    alt.Chart(wash_pre_predictions)
    .mark_errorband()
    .encode(
        x=xvar,
        y=alt.Y("ci_low", title=yvar),
        y2="ci_high",
    )
)

"""
Washington Post-Policy Charts
"""
# Grid for predicted values
wash_post_x = wash_post.loc[pd.notnull(wash_post[yvar]), xvar]
wash_post_xmin = wash_post_x.min()
wash_post_xmax = wash_post_x.max()
wash_post_step = (wash_post_xmax - wash_post_xmin) / 100
wash_post_grid = np.arange(
    wash_post_xmin, wash_post_xmax + wash_post_step, wash_post_step
)
wash_post_predictions = pd.DataFrame({xvar: wash_post_grid})

# Fit model, get predictions
wash_post_model = smf.ols(f"{yvar} ~ {xvar}", data=wash_post).fit()
wash_post_model_predict = wash_post_model.get_prediction(wash_post_predictions[xvar])
wash_post_predictions[yvar] = wash_post_model_predict.summary_frame()["mean"]
wash_post_predictions[["ci_low", "ci_high"]] = wash_post_model_predict.conf_int(
    alpha=alpha
)

# Build chart
wash_post_reg = (
    alt.Chart(wash_post_predictions)
    .mark_line()
    .encode(x=xvar, y=alt.Y(yvar, scale=alt.Scale(zero=False)))
)
wash_post_ci = (
    alt.Chart(wash_post_predictions)
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
wash_pre_chart = wash_pre_ci + wash_pre_reg
wash_post_chart = wash_post_ci + wash_post_reg
pooled_pre_chart = pooled_pre_ci + pooled_pre_reg
pooled_post_chart = pooled_post_ci + pooled_post_reg


"""
PRE-POST ANALYSIS
"""
pre_post_chart = alt.layer(wash_pre_chart, wash_post_chart, line).properties(
    title="Overdose Deaths in Washington Before and After 2012 (Policy Change)"
)

"""
DIFFERENCE-IN-DIFFERENCE ANALYSIS
"""
wash_vs_pooled = alt.layer(
    wash_pre_chart, wash_post_chart, pooled_pre_chart, pooled_post_chart, line
).properties(title="Pre- and Post- Policy Trends In Washington and Control States")


# saving charts as png files
save(
    pre_post_chart,
    "/Users/emeliamavis/720/pds2021-opioids-team-8-1/30_results/overdose_death/NEW_PLOTS_FOR_FINAL/wash_pre_post.png",
)
save(
    pre_post_chart,
    "/Users/emeliamavis/720/pds2021-opioids-team-8-1/30_results/overdose_death/NEW_PLOTS_FOR_FINAL/wash_vs_pooled.png",
)
