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
texas = data[data.State == "TX"]
pooled = data[
    (data["State"] == "WI") | (data["State"] == "MS") | (data["State"] == "KS")
]

# break washington into pre and post subsets for pre-post and diff-in-diff analyses
texas_pre = texas[texas.Year < 2007]
texas_post = texas[texas.Year >= 2007]

# break pooled states into pre and post subsets for diff-in-diff analysis
pooled_pre = pooled[pooled.Year < 2007]
pooled_post = pooled[pooled.Year >= 2007]

# add vertical line for year = 2012 (policy change)
line = (
    alt.Chart(pd.DataFrame({"Year": [2007]})).mark_rule(color="red").encode(x="Year:Q")
)

# same arguments for each chart
yvar = "Rate"
xvar = "Year"
alpha = 0.05

"""
PRE-POST CHART BUILDING
"""

"""
Texas Pre-Policy Charts
"""
# Grid for predicted values
texas_pre_x = texas_pre.loc[pd.notnull(texas_pre[yvar]), xvar]
texas_pre_xmin = texas_pre_x.min()
texas_pre_xmax = texas_pre_x.max()
texas_pre_step = (texas_pre_xmax - texas_pre_xmin) / 100
texas_pre_grid = np.arange(
    texas_pre_xmin, texas_pre_xmax + texas_pre_step, texas_pre_step
)
texas_pre_predictions = pd.DataFrame({xvar: texas_pre_grid})

# Fit model, get predictions
texas_pre_model = smf.ols(f"{yvar} ~ {xvar}", data=texas_pre).fit()
texas_pre_model_predict = texas_pre_model.get_prediction(texas_pre_predictions[xvar])
texas_pre_predictions[yvar] = texas_pre_model_predict.summary_frame()["mean"]
texas_pre_predictions[["ci_low", "ci_high"]] = texas_pre_model_predict.conf_int(
    alpha=alpha
)

# Build chart
texas_pre_reg = (
    alt.Chart(texas_pre_predictions)
    .mark_line()
    .encode(x=xvar, y=alt.Y(yvar, scale=alt.Scale(zero=False)))
)
texas_pre_ci = (
    alt.Chart(texas_pre_predictions)
    .mark_errorband()
    .encode(
        x=xvar,
        y=alt.Y("ci_low", title=yvar),
        y2="ci_high",
    )
)

"""
Texas Post-Policy Charts
"""
# Grid for predicted values
texas_post_x = texas_post.loc[pd.notnull(texas_post[yvar]), xvar]
texas_post_xmin = texas_post_x.min()
texas_post_xmax = texas_post_x.max()
texas_post_step = (texas_post_xmax - texas_post_xmin) / 100
texas_post_grid = np.arange(
    texas_post_xmin, texas_post_xmax + texas_post_step, texas_post_step
)
texas_post_predictions = pd.DataFrame({xvar: texas_post_grid})

# Fit model, get predictions
texas_post_model = smf.ols(f"{yvar} ~ {xvar}", data=texas_post).fit()
texas_post_model_predict = texas_post_model.get_prediction(texas_post_predictions[xvar])
texas_post_predictions[yvar] = texas_post_model_predict.summary_frame()["mean"]
texas_post_predictions[["ci_low", "ci_high"]] = texas_post_model_predict.conf_int(
    alpha=alpha
)

# Build chart
texas_post_reg = (
    alt.Chart(texas_post_predictions)
    .mark_line()
    .encode(x=xvar, y=alt.Y(yvar, scale=alt.Scale(zero=False)))
)
texas_post_ci = (
    alt.Chart(texas_post_predictions)
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
texas_pre_chart = texas_pre_ci + texas_pre_reg
texas_post_chart = texas_post_ci + texas_post_reg
pooled_pre_chart = pooled_pre_ci + pooled_pre_reg
pooled_post_chart = pooled_post_ci + pooled_post_reg


"""
PRE-POST ANALYSIS
"""
pre_post_chart = alt.layer(texas_pre_chart, texas_post_chart, line).properties(
    title="Overdose Deaths in Texas Before and After 2007 (Policy Change)"
)

"""
DIFFERENCE-IN-DIFFERENCE ANALYSIS
"""
texas_vs_pooled = alt.layer(
    texas_pre_chart, texas_post_chart, pooled_pre_chart, pooled_post_chart, line
).properties(title="Pre- and Post- Policy Trends In Texas and Control States")


# saving charts as png files
save(
    pre_post_chart,
    "/Users/emeliamavis/720/pds2021-opioids-team-8-1/30_results/overdose_death/NEW_PLOTS_FOR_FINAL/texas_pre_post.png",
)
save(
    texas_vs_pooled,
    "/Users/emeliamavis/720/pds2021-opioids-team-8-1/30_results/overdose_death/NEW_PLOTS_FOR_FINAL/texas_vs_pooled.png",
)
