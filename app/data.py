import re
import typing as t

import pandas as pd


# This is Jan 1 2022 to Feb 28 2022 tournament matchups, modern format (pre-Lurrus ban)
# This data is pulled from here:
#
# https://mtgmelee.com/Decklist/ArchetypeMatrix#?Format=Modern&Limit=10&Date%5BStart%5D=2022-01-01T05%3A00%3A00.000Z&Date%5BEnd%5D=2022-03-01T04%3A59%3A59.999Z

deck_info_raw = """Four-Color Blink (Yorion)  Decklists: 36 | MWP: 49.5 %
Grixis Death's Shadow  Decklists: 34 | MWP: 46.1 %
Mono-White Hammer  Decklists: 25 | MWP: 54.5 %
Burn  Decklists: 26 | MWP: 51.3 %
Azorius Control (Kaheera)  Decklists: 22 | MWP: 46.4 %
Temur Rhinos  Decklists: 15 | MWP: 67.0 %
Amulet Titan  Decklists: 18 | MWP: 43.8 %
Izzet Murktide  Decklists: 20 | MWP: 45.5 %
Jund  Decklists: 11 | MWP: 46.4 %
Eldrazi Tron  Decklists: 11 | MWP: 50.9 %"""

tsv_percents = """0.5	0.385	0.308	0.571	0.714	0.5	0.2	0.833	0.333	0.5
0.615	0.5	0.467	0.167	0.75	0.4	0.75	0.333	0.667	0.333
0.692	0.533	0.5	0.6	0.667	0.125	0.6	0.5	0.667	0.5
0.429	0.833	0.4	0.5	0.25	0.125	0.667	0.75	0	0
0.286	0.25	0.333	0.75	0.5	0	0.5	1	0.5	0.333
0.5	0.6	0.875	0.875	1	0.5	0.5	0.333	1	0.333
0.8	0.25	0.4	0.333	0.5	0.5	0.5	0.4	0.5	0.5
0.167	0.667	0.5	0.25	0	0.667	0.6	0.5	0.667	0
0.667	0.333	0.333	1	0.5	0	0.5	0.333	0.5	0
0.5	0.667	0.5	1	0.667	0.667	0.5	1	1	0.5"""

tsv_matches = """1000	13	13	14	7	8	5	6	6	1000
13	1000	15	12	4	5	8	9	6	3
13	15	1000	10	3	8	5	4	3	1000
14	12	10	1000	8	8	3	4	2	2
7	4	3	8	1000	1	2	1	1000	6
8	5	8	8	1	1000	2	3	2	3
5	8	5	3	2	2	1000	5	2	1000
6	9	4	4	1	3	5	1000	3	1
6	6	3	2	1000	2	2	3	1000	1
1000	3	1000	2	6	3	1000	1	1	1000"""


_parse_decks_regexp = re.compile("([A-Z].*?)\s+Decklists: ([0-9]+) .*?")

deck_names = [
    re.match(_parse_decks_regexp, row).group(1)
    for row in deck_info_raw.split("\n")
]


deck_count_full = pd.Series([
    int(re.match(_parse_decks_regexp, row).group(2))
    for row in deck_info_raw.split("\n")
], index=deck_names, name="deck_count")


def tsv_to_matrix(
        tsv: str,
        labels: t.Sequence[str],
        func: t.Optional[t.Callable[[t.Any], t.Any]] = None
) -> pd.DataFrame:
    """Take a string of a tsv, and return an N by N matrix."""
    if func is None:
        def func(x): return x

    df = pd.DataFrame([
        [func(cell) for cell in row.split("\t")]
        for row in tsv.split("\n")
    ], index=labels, columns=labels)

    return df


matchups = tsv_to_matrix(tsv_percents, labels=deck_names, func=float)
matches = tsv_to_matrix(tsv_matches, labels=deck_names, func=int)


# Add beta(1,1) prior
matchups_regularized = (matches * matchups + 1) / (matches + 2)

# We need to treat the full df a little differently for now.
# Basically, Eldrazi Tron is all positives.
payoff_matrix_full = pd.DataFrame(
    matchups_regularized,
    columns=deck_names,
    index=deck_names
)
payoff_matrix_full.columns.name = "deck"
payoff_matrix_full.index.name = "deck"

payoff_matrix = payoff_matrix_full.loc[
    payoff_matrix_full.index != "Eldrazi Tron",
    payoff_matrix_full.columns != "Eldrazi Tron"
]

deck_count = deck_count_full.reindex(payoff_matrix.index)


assert payoff_matrix_full.sum().sum() == payoff_matrix_full.shape[0] ** 2 / 2
