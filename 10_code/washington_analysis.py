import pandas as pd
import numpy as np
import altair as alt
import statsmodels.formula.api as smf

data = pd.read_csv(
    "https://raw.githubusercontent.com/MIDS-at-Duke/pds2021-opioids-team-8-1/main/20_intermediate_files/Death_and_Population.csv?token=AQURHZAMNI56MOEM2UQPDVLBRZOA6"
)

# double check that there are no duplicates and no missing values
assert ~data.duplicated(["Year", "State", "County"]).any()
assert ~data.isna().any().sum()

data["Overdose_Deaths_per_Capita"] = (
    data["Total Deaths By Overdose"] / data.Population
) * 100

# subset to target states
wash = data[data.State == "WA"]
okl = data[data.State == "OK"]
arz = data[data.State == "AZ"]
col = data[data.State == "CO"]

# break washington into pre and post subsets for pre-post analysis
wash_pre = wash[wash.Year < 2012]
wash_post = wash[wash.Year >= 2012]

# add vertical line for year = 2012 (policy change)
line = (
    alt.Chart(wash)
    .transform_quantile("Year", probs=[0.75], as_=["prob", "value"])
    .mark_rule()
    .encode(x="value:Q")
)

# same arguments for each chart
yvar = "Overdose_Deaths_per_Capita"
xvar = "Year"
alpha = 0.05

"""
PRE-POST ANALYSIS
"""

"""
Pre-Policy Charts
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
Post-Policy Charts
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

pre_chart = wash_pre_ci + wash_pre_reg
post_chart = wash_post_ci + wash_post_reg

"""
DIFFERENCE-IN-DIFFERENCE ANALYSIS
"""

"""
Washington Chart
"""

# Grid for predicted values
wa_x = wash.loc[pd.notnull(wash[yvar]), xvar]
wa_xmin = wa_x.min()
wa_xmax = wa_x.max()
wa_step = (wa_xmax - wa_xmin) / 100
wa_grid = np.arange(wa_xmin, wa_xmax + wa_step, wa_step)
wa_predictions = pd.DataFrame({xvar: wa_grid})

# Fit model, get predictions
wa_model = smf.ols(f"{yvar} ~ {xvar}", data=wash).fit()
wa_model_predict = wa_model.get_prediction(wa_predictions[xvar])
wa_predictions[yvar] = wa_model_predict.summary_frame()["mean"]
wa_predictions[["ci_low", "ci_high"]] = wa_model_predict.conf_int(alpha=alpha)

# Build chart
wa_reg = (
    alt.Chart(wa_predictions)
    .mark_line()
    .encode(x=xvar, y=alt.Y(yvar, scale=alt.Scale(zero=False)))
)
wa_ci = (
    alt.Chart(wa_predictions)
    .mark_errorband()
    .encode(
        x=xvar,
        y=alt.Y("ci_low", title=yvar),
        y2="ci_high",
    )
)

"""
Oklahoma Chart
"""

# Grid for predicted values
okl_x = okl.loc[pd.notnull(okl[yvar]), xvar]
okl_xmin = okl_x.min()
okl_xmax = okl_x.max()
okl_step = (okl_xmax - okl_xmin) / 100
okl_grid = np.arange(okl_xmin, okl_xmax + okl_step, okl_step)
okl_predictions = pd.DataFrame({xvar: okl_grid})

# Fit model, get predictions
okl_model = smf.ols(f"{yvar} ~ {xvar}", data=okl).fit()
okl_model_predict = okl_model.get_prediction(okl_predictions[xvar])
okl_predictions[yvar] = okl_model_predict.summary_frame()["mean"]
okl_predictions[["ci_low", "ci_high"]] = okl_model_predict.conf_int(alpha=alpha)

# Build chart
okl_reg = (
    alt.Chart(okl_predictions)
    .mark_line(color="red")
    .encode(x=xvar, y=alt.Y(yvar, scale=alt.Scale(zero=False)))
)
okl_ci = (
    alt.Chart(okl_predictions)
    .mark_errorband(color="red")
    .encode(
        x=xvar,
        y=alt.Y("ci_low", title=yvar),
        y2="ci_high",
    )
)

"""
Arizona Chart
"""

# Grid for predicted values
arz_x = arz.loc[pd.notnull(arz[yvar]), xvar]
arz_xmin = arz_x.min()
arz_xmax = arz_x.max()
arz_step = (arz_xmax - arz_xmin) / 100
arz_grid = np.arange(arz_xmin, arz_xmax + arz_step, arz_step)
arz_predictions = pd.DataFrame({xvar: arz_grid})

# Fit model, get predictions
arz_model = smf.ols(f"{yvar} ~ {xvar}", data=arz).fit()
arz_model_predict = arz_model.get_prediction(arz_predictions[xvar])
arz_predictions[yvar] = arz_model_predict.summary_frame()["mean"]
arz_predictions[["ci_low", "ci_high"]] = arz_model_predict.conf_int(alpha=alpha)

# Build chart
arz_reg = (
    alt.Chart(arz_predictions)
    .mark_line(color="yellow")
    .encode(x=xvar, y=alt.Y(yvar, scale=alt.Scale(zero=False)))
)
arz_ci = (
    alt.Chart(arz_predictions)
    .mark_errorband(color="yellow")
    .encode(
        x=xvar,
        y=alt.Y("ci_low", title=yvar),
        y2="ci_high",
    )
)

"""
Colorado Chart
"""

# Grid for predicted values
col_x = col.loc[pd.notnull(col[yvar]), xvar]
col_xmin = col_x.min()
col_xmax = col_x.max()
col_step = (col_xmax - col_xmin) / 100
col_grid = np.arange(col_xmin, col_xmax + col_step, col_step)
col_predictions = pd.DataFrame({xvar: col_grid})

# Fit model, get predictions
col_model = smf.ols(f"{yvar} ~ {xvar}", data=col).fit()
col_model_predict = col_model.get_prediction(col_predictions[xvar])
col_predictions[yvar] = col_model_predict.summary_frame()["mean"]
col_predictions[["ci_low", "ci_high"]] = col_model_predict.conf_int(alpha=alpha)

# Build chart
col_reg = (
    alt.Chart(col_predictions)
    .mark_line(color="green")
    .encode(x=xvar, y=alt.Y(yvar, scale=alt.Scale(zero=False)))
)
col_ci = (
    alt.Chart(col_predictions)
    .mark_errorband(color="green")
    .encode(
        x=xvar,
        y=alt.Y("ci_low", title=yvar),
        y2="ci_high",
    )
)

# all four charts for each state, ready to be layered
col_chart = col_ci + col_reg
arz_chart = arz_ci + arz_reg
okl_chart = okl_ci + okl_reg
wa_chart = wa_ci + wa_reg

# pre-post analysis chart
pre_post_chart = alt.layer(pre_chart, post_chart, line).properties()

# the diff-in-diff charts between each comparison state
wash_vs_okl = alt.layer(wa_chart, okl_chart, line).properties(
    title="Overdose Death Rates in Washington vs. Oklahoma"
)
wash_vs_arz = alt.layer(wa_chart, arz_chart, line).properties(
    title="Overdose Death Rates in Washington vs. Arizona"
)
wash_vs_col = alt.layer(wa_chart, col_chart, line).properties(
    title="Overdose Death Rates in Washington vs. Colorado"
)
