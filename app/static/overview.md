### Overview
This tool calculates the Nash equilibrium of a Magic: the Gathering metagame given
a match-up matrix and starting values for the metagame share.

The two main variables that this tool outputs:

- **`metagame_share`:** what % of the metagame is represented by each deck
- **`win_rate`:** what % of the time a deck will win

These values are represented as an evolution over time. The left-most part of each graph
represents the metagame share according to the input vector; to the right, hopefully we
converge to a Nash equilibrium.

### How it works

This tool uses a minimax algorithm and gradient descent implemented with JAX to solve
for the Nash equilibrium. Although this is far less efficient than Lemke-Howson,
this approach allows us to visualize an evolution of a metagame over time.

While the dashboard is loading, I use the Lemke-Howson solution as a terminating value
to log-linearly interpolate the values. (Because the domain is also log, this is shown as
linear in the visualization).

### Things to add / to-do list

It's very unlikely I'll ever get around to the below list. I had a lot on my plate! I'm releasing this to the public after a few months of having not worked on it.

In any case, these are the things I would love to have here:

- Priors
- Parametrization of:
    - Payoff matrix + w0. Data is currently hardcoded to te pre-Lurrus meta
    - Learning rate + decay + steps
    - Priors
    - Scale (log vs linear)
- Analysis of eigenvectors of payoff matrix, which have interesting interpretations (i.e. eigenvectors corresponding with complex valued eigenvalues show how the meta evolves over time when it is out of equilibrium).
- Code is a mess. Clean it up!
- "Rerun" button, since it's so fun to watch the evolution fill in
- Smarter configurable scales + option to choose how many steps to show
