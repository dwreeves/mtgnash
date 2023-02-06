import typing as t

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import grad
from jax import jit

from app.state import update_df_states


def get_nash_equilibrium(
        payoff_matrix: pd.DataFrame,
        weights_t0: t.Optional[pd.Series] = None,
        steps: int = 2 ** 15,
        learning_rate: float = 0.02,
        learning_rate_decay: float = 0.0002,
        actual_solution: pd.Series = None
):
    payoff_matrix_np = payoff_matrix.to_numpy()

    weights_t0 = weights_t0.to_numpy()

    evolution_matrix = np.matrix([weights_t0])
    update_df_states(
        evolution_matrix=evolution_matrix,
        payoff_matrix=payoff_matrix,
        actual_solution=actual_solution,
        steps=steps
    )

    def minimax_loss_function(
            weights: np.array
    ):
        """Minimize the maximum win rate"""
        # Bound to 0 and re-normalize
        weights = jnp.maximum(weights, 0.0)
        weights = weights / weights.sum()

        win_rate_vector = jnp.dot(payoff_matrix_np, weights)
        max_win_rate = jnp.max(win_rate_vector)
        return max_win_rate

    d_loss_d_weight = jit(grad(minimax_loss_function, argnums=0))

    w = weights_t0

    # Gradient descent
    for step in range(steps):
        w_grad = d_loss_d_weight(w)
        w -= w_grad * learning_rate / (1 + learning_rate_decay * step)
        w = np.maximum(w, 0.0)
        w /= w.sum()
        if np.log2(1 + step) // 1 == np.log2(1 + step):
            evolution_matrix = np.append(evolution_matrix, [w], axis=0)
            update_df_states(
                evolution_matrix=evolution_matrix,
                payoff_matrix=payoff_matrix,
                actual_solution=actual_solution,
                steps=steps
            )

    update_df_states(
        evolution_matrix=evolution_matrix,
        payoff_matrix=payoff_matrix,
        actual_solution=actual_solution,
        steps=steps
    )

    df = pd.DataFrame(
        evolution_matrix,
        columns=payoff_matrix.columns,
    )
    df.index.name = "log2(step)"
    return df
