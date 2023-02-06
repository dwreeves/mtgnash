import numpy as np
import pandas as pd
import streamlit as st

from app.visualization import evolution_chart


chart_empty = None


def update_df_states(
        evolution_matrix: np.matrix,
        payoff_matrix: pd.DataFrame,
        actual_solution: pd.Series,
        steps: int
):
    global chart_empty
    if chart_empty is None:
        chart_empty = st.empty()

    evolution_df = pd.DataFrame(
        evolution_matrix,
        columns=payoff_matrix.columns,
    )
    evolution_df.index.name = "log2(step)"

    evolution_df["_interpolated"] = False

    # INTERPOLATE REMAINING ROWS
    last_row = evolution_df.iloc[-1]
    log2_steps = int(np.log2(steps)) + 1
    if log2_steps != last_row.name:
        for interpolated_step in range(last_row.name + 1, log2_steps + 1):

            offset = interpolated_step - last_row.name
            proportion = offset / (log2_steps - last_row.name)

            interpolated_row = proportion * actual_solution + (1 - proportion) * last_row
            interpolated_row.name = interpolated_step
            interpolated_row["_interpolated"] = True

            evolution_df.loc[interpolated_step, :] = interpolated_row

    chart_empty.altair_chart(evolution_chart(
        payoff_matrix=payoff_matrix,
        evolution_df=evolution_df,
        show_win_rate=st.session_state.get("show_win_rate", True),
        show_metagame_share=st.session_state.get("show_metagame_share", False)
    ), use_container_width=True, theme=None)
