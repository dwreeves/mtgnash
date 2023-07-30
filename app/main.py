# ruff: noqa: E402
import sys


sys.path.append(".")

import nashpy as nash
import numpy as np
import pandas as pd
import streamlit as st

from app.algo import get_nash_equilibrium
from app.data_lotr_protour import deck_count
from app.data_lotr_protour import payoff_matrix
from app.state import update_df_states
from app.utils import st_markdown_file


def get_default_initial_weights(payoff_matrix: pd.DataFrame) -> pd.Series:
    n = payoff_matrix.shape[0]
    weights_t0 = np.array([1 / n for _ in range(n)])
    return pd.Series(weights_t0, index=payoff_matrix.columns)


def main():
    st.set_page_config(layout="wide")
    st.header("`mtgnash`: Metagame Solver - MTG PT LotR 2023 - Swiss Rounds")
    st.markdown("---")
    with st.expander("Description", expanded=False):
        st_markdown_file("overview.md")
    st.markdown("---")

    with st.sidebar:
        data_type = st.selectbox("Show data for:", ["Metagame Share", "Win Rate"])

    st.session_state["show_metagame_share"] = data_type == "Metagame Share"
    st.session_state["show_win_rate"] = not (data_type == "Metagame Share")

    actual_solution = pd.Series(
        nash.Game(payoff_matrix).lemke_howson(initial_dropped_label=0)[0],
        index=payoff_matrix.index,
        name="Lemke-Howson Solution"
    )

    weights_t0 = deck_count / deck_count.sum()
    update_df_states(
        evolution_matrix=np.matrix([weights_t0]),
        payoff_matrix=payoff_matrix,
        actual_solution=actual_solution,
        steps=16
    )

    evolution_df = get_nash_equilibrium(
        payoff_matrix=payoff_matrix,
        weights_t0=(deck_count / deck_count.sum()),
        actual_solution=actual_solution
    )

    # ##########################################################################
    # NOTE: The below doesn't seem to work on Streamlist's managed service.
    # Even though this works locally.
    # (It makes the Altair chart away to render the dataframe.)
    # ##########################################################################

    # st.dataframe(
    #     pd.concat([
    #         evolution_df.iloc[-1].rename("Gradient Descent Solution"),
    #         actual_solution
    #     ], axis=1)
    # )


if __name__ == "__main__":
    main()
