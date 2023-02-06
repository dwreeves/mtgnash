from typing import TypeVar

import altair as alt
import pandas as pd


T = TypeVar("T")


def pivot_and_format(
        base_df: pd.DataFrame,
        attr_df: pd.DataFrame,
        col_name: str
):

    df = base_df.stack().rename("value").reset_index()
    df["column"] = col_name
    df = df.merge(
        attr_df,
        left_on="log2(step)",
        right_index=True
    )

    # Interpolated column must be duplicated to last non-duplicated.
    latest_values = (
        df
        .loc[(df["_interpolated"] == False), :]  # noqa
        .sort_values("log2(step)")
        .groupby("deck")
        .last()
        .reset_index()
    )
    latest_values["_interpolated"] = True

    df = pd.concat([
        df,
        latest_values
    ]).reset_index(drop=True)

    df = df.sort_values(["deck", "log2(step)", "_interpolated"])
    df["value_d1"] = (
        df
        .groupby(["column"])
        ["value"]
        .diff()
    )
    df["value_d1"] = df["value_d1"].fillna(0)
    df.loc[df["log2(step)"] == 0, "value_d1"] = 0

    return df


def create_evolution_chart_dataframe(
        payoff_matrix: pd.DataFrame,
        evolution_df: pd.DataFrame,
        show_metagame_share: bool = True,
        show_win_rate: bool = True
) -> pd.DataFrame:
    df_list: list[pd.DataFrame] = []

    # We need to show at least one
    show_metagame_share |= not show_win_rate
    deck_cols = filter(lambda _: not _.startswith("_"), evolution_df.columns)
    attr_cols = filter(lambda _: _.startswith("_"), evolution_df.columns)

    if show_metagame_share:
        stacked_metagame_df = pivot_and_format(
            base_df=evolution_df[deck_cols],
            attr_df=evolution_df[attr_cols],
            col_name="Metagame Share"
        )
        df_list.append(stacked_metagame_df)

    if show_win_rate:
        stacked_win_rate_df = pivot_and_format(
            base_df=payoff_matrix.dot(evolution_df[deck_cols].T).T,
            attr_df=evolution_df[attr_cols],
            col_name="Win Rate"
        )
        df_list.append(stacked_win_rate_df)

    df = (
        pd.concat(df_list, axis=0)
        .sort_values(["column", "deck", "_interpolated", "log2(step)"])
    )

    pd.set_option("display.max_rows", 200)

    return df


def evolution_chart(
        payoff_matrix: pd.DataFrame,
        evolution_df: pd.DataFrame,
        show_metagame_share: bool = True,
        show_win_rate: bool = True
) -> alt.Chart:

    df = create_evolution_chart_dataframe(
        payoff_matrix=payoff_matrix,
        evolution_df=evolution_df,
        show_metagame_share=show_metagame_share,
        show_win_rate=show_win_rate
    )

    on_click_vertical_rule_selector = alt.selection_interval(
        name="on_click_vertical_rule_selector",
        encodings=["x"],
    )

    mouseover_vertical_rule_selector = alt.selection(
        name="mouseover_vertical_rule_selector",
        type="single",
        nearest=True,
        on="mouseover",
        fields=["log2(step)"],
        empty="none"
    )

    base = alt.Chart(df)

    # The vertical select thing is pretty unfriendly to the edges.
    # Adding a little "fudge" makes it easier to select first and last values.
    width = 850
    # width = "container"

    # The lines that represent the time series
    lines_base = (
        base
        .mark_line(
            interpolate="basis",
        )
        .encode(
            x=alt.X(
                "log2(step):Q"
            ),
            y="value:Q",
            color="deck:N",
            strokeDash="_interpolated:N",
            # TODO:
            #  I'd like to be more exlicit about defining the style here.
            #  However, Altair seems to be limited here. (Vega Lite seems fine though.)
            #  Maybe put in a feature req / bug report to Altair.
            # strokeDash=alt.condition(
            #     "datum._interpolated",
            #     alt.StrokeDash([1, 0]),
            #     alt.StrokeDash([4, 4])
            # ),
            opacity=alt.Opacity(
                "_interpolated:N",
                scale=alt.Scale(
                    domain=[False, True],
                    range=[1.0, 0.4]
                ),
                legend=None
            )
        )
    )

    # We need to separate out the steps up to the encoding from all
    # the interactive stuff because we want to .mark_points() on top of
    # the original,
    lines = lines_base.add_selection(on_click_vertical_rule_selector)  # .interactive(bind_y=False)

    # df.groupby(["deck", "log2(step)"])[""].first()
    _df = evolution_df.copy()
    _df.columns = list(i.replace("'", "") for i in _df.columns)
    _df = _df.applymap(lambda _: "{:.3f}".format(_ * 100) + "%")
    _df = _df.reset_index()
    _df["interpolated"] = evolution_df["_interpolated"]

    # Used to bind the selector to a location
    hidden_vertical_rules = (
        alt.Chart(_df)
        .mark_rule()
        .encode(
            x="log2(step):Q",
            # tooltip=alt.Tooltip(field="value", bin="binned"),
            tooltip=list(i for i in _df.columns if not i.startswith("_")),
            opacity=alt.value(0)
        )
        .add_selection(
            mouseover_vertical_rule_selector
        )
    )

    # Draw points on the line, and highlight based on selection
    points = (
        lines_base
        .mark_point()
        .encode(
            opacity=alt.condition(
                mouseover_vertical_rule_selector,
                alt.value(1),
                alt.value(0)
            )
        )
    )

    # Draw text labels near the points, and highlight based on selection
    text = (
        lines_base
        .mark_text(align="left", dx=5, dy=-5)
        .encode(
            text=alt.condition(
                mouseover_vertical_rule_selector,
                "deck:N",
                alt.value(" ")
           )
        )
    )
    # Draw a rule at the location of the selection
    rules = (
        base
        .mark_rule(color="gray")
        .encode(
            x="log2(step):Q",
        )
        .transform_filter(
            mouseover_vertical_rule_selector
        )
    )

    final_pre_format = alt.layer(
        lines,
        hidden_vertical_rules,
        points,
        rules,
        text
    )

    deltas_chart = (
        lines_base
        .mark_bar()  # (orient="horizontal")
        .encode(
            x=alt.X("sum(value_d1)", scale=alt.Scale(domainMin=-1, domainMax=1, type="symlog")),
            y="deck:N",
            tooltip=alt.value('none'),
        )
        .transform_filter(
            on_click_vertical_rule_selector
        )
        .properties(
            width=width,
            height=300
        )
    )
    alt.Chart()

    # Put the five layers into a chart and bind the data
    final = (
        final_pre_format
        .properties(
            width=width,
            height=500
        )
        .resolve_scale(
            y="independent"
        )
    )

    final &= deltas_chart
    final = final.configure_scale(
            bandPaddingOuter=0.5
        )

    return final
