import pandas as pd
import numpy as np
import altair as alt
import statsmodels.formula.api as smf
import pathlib

# Mark a vertical line for a certain year
def get_vertical_line(year):
    line = alt.Chart(
        pd.DataFrame({"Year": [year]})
    ).mark_rule(
        color="red"
    ).encode(
        x="Year:Q"
    )
    return line


# Get a grid for future prediction
def get_plot_grid(df, x, y):
    X = df.loc[pd.notnull(df[y]), x]
    X_min = X.min()
    X_max = X.max()
    X_step = (X_max - X_min) / 100
    return np.arange(X_min, X_max + X_step, X_step)


# Get prediction result using ols
def get_ols_prediction(df, grid, x, y):
    predictions = pd.DataFrame({x: grid})
    model = smf.ols(f"{y} ~ {x}", data=df).fit()
    model_predict = model.get_prediction(predictions[x])
    predictions[y] = model_predict.summary_frame()["mean"]
    predictions[["ci_low", "ci_high"]] = model_predict.conf_int(alpha=0.05)
    return predictions


def get_regression_chart(df, x, y):
    regression_chart = (
        alt.Chart(df)
        .mark_line()
        .encode(x=x, y=alt.Y(y, scale=alt.Scale(zero=False)))
    )
    return regression_chart


def get_ci_chart(df, x, y):
    ci_chart = (
        alt.Chart(df)
        .mark_errorband()
        .encode(
            x=x,
            y=alt.Y("ci_low", title=y),
            y2="ci_high",
        )
    )
    return ci_chart


def pre_post_analysis(dataframe, year):
    x = "Year"
    y = "Overdose_death_per_capita"
    before = dataframe[dataframe["Year"] < year]
    after = dataframe[dataframe["Year"] >= year]
    
    before_grid = get_plot_grid(before, x, y)
    before_predictions = get_ols_prediction(before, before_grid, x, y)
    before_reg_chart = get_regression_chart(before_predictions, x, y)
    before_ci_chart = get_ci_chart(before_predictions, x, y)

    after_grid = get_plot_grid(after, x, y)
    after_predictions = get_ols_prediction(after, after_grid, x, y)
    after_reg_chart = get_regression_chart(after_predictions, x, y)
    after_ci_chart = get_ci_chart(after_predictions, x, y)
    return before_reg_chart + before_ci_chart, after_reg_chart + after_ci_chart


if __name__ == "__main__":
    data = pd.read_parquet(f"{pathlib.Path.cwd()}/20_intermediate_files/Death_and_Population.gzip")
    data["Overdose_death_per_capita"] = data["Deaths"] / data["Population"]

    # Select target states
    TX = data[data.State == "TX"] # Texas
    WI = data[data.State == "WI"] # Wisconsin
    MS = data[data.State == "MS"] # Mississippi
    KS = data[data.State == "KS"] # Kansus

    # Where to save files
    path_prefix = f"{pathlib.Path.cwd()}/30_results/overdose_death/"

    # Pre-post analysis
    line = get_vertical_line(2007)
    TX_pre, TX_post = pre_post_analysis(TX, 2007)
    TX_pre_post_chart = alt.layer(TX_pre, TX_post, line).properties()
    TX_pre_post_chart.save(f"{path_prefix}TX.png")

    # The diff-in-diff charts between each comparison state
    # Texas vs. Wisconsin
    WI_pre, WI_post = pre_post_analysis(WI, 2007)
    TX_vs_WI = alt.layer(TX_pre_post_chart, WI_pre + WI_post, line).properties(
        title="Overdose Death Rates in Texas vs. Wisconsin"
    )
    TX_vs_WI.save(f"{path_prefix}TX_vs_WI.png")

    # Texas vs. Mississippi
    MS_pre, MS_post = pre_post_analysis(MS, 2007)
    TX_vs_MS = alt.layer(TX_pre_post_chart, MS_pre + MS_post, line).properties(
        title="Overdose Death Rates in Texas vs. Mississippi"
    )
    TX_vs_MS.save(f"{path_prefix}TX_vs_MS.png")

    # Texas vs. Kansas
    KS_pre, KS_post = pre_post_analysis(KS, 2007)
    TX_vs_KS = alt.layer(TX_pre_post_chart, KS_pre + KS_post, line).properties(
        title="Overdose Death Rates in Texas vs. Kansas"
    )
    TX_vs_KS.save(f"{path_prefix}TX_vs_KS.png")

