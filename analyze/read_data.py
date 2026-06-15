import json
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")


# ----------------------------
# Helpers: configs and tables
# ----------------------------
def get_latest_config(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT config
        FROM sessions
        ORDER BY created_at DESC
        LIMIT 1
    """)
    row = cur.fetchone()
    latest_config = row[0] if row else None
    return json.loads(latest_config) if latest_config else None


def build_actual_distribution_table(config):
    rows = []
    for dist in config["distribution_cues"].values():
        rows.append(pd.DataFrame({
            "id": dist["id"],
            "magnitude": dist["magnitude_values"],
            "probability": dist["probabilities"],
            "dist_image": dist["image"],
            "mag_image": [config["stimulus_sets"]["3"][str(mag)] for mag in dist["magnitude_values"]]
        }))
    actual = pd.concat(rows, ignore_index=True)
    return actual


def summarize_distributions(actual_distribution_table):
    def _summary(group):
        x = group["magnitude"].to_numpy()
        p = group["probability"].to_numpy()
        mu = np.sum(x * p)
        var = np.sum(p * (x - mu) ** 2)
        return pd.Series({
            "expected_value": mu,
            "variance": var,
        })

    return actual_distribution_table.groupby("id", sort=False).apply(_summary)


def prepare_choice_data(df, time_col="date"):
    choice_df = df.copy()
    choice_df[time_col] = pd.to_datetime(choice_df[time_col])

    choice_df = choice_df.query('outcome == "choice"').copy()

    choice_df["distribution_options_sorted"] = choice_df["distribution_options"].apply(
        lambda x: tuple(sorted(x))
    )
    choice_df["sorted_first_chosen"] = choice_df.apply(
        lambda row: row["chosen_distribution"] == row["distribution_options_sorted"][0],
        axis=1
    )
    return choice_df


def build_choice_matrix(choice_df):
    outcomes = choice_df.groupby("distribution_options_sorted")["sorted_first_chosen"].mean()

    pairs = []
    values = []
    for pair, p in outcomes.items():
        pairs.extend([pair, pair[::-1]])
        values.extend([p, 1 - p])

    s = pd.Series(
        values,
        index=pd.MultiIndex.from_tuples(pairs, names=["chosen", "alternative"])
    )
    return s.unstack()


def sort_square_table(table, summary, sort_col="expected_value", ascending=False):
    order = summary.sort_values(sort_col, ascending=ascending).index
    return table.reindex(index=order, columns=order)


# ----------------------------
# Plot 1: choice heatmaps by time window
# ----------------------------
def plot_choice_heatmaps_by_time(
    choice_df,
    distribution_summary,
    time_col="date",
    freq="2D",
    sort_col="expected_value",
    ascending=False,
    cmap="coolwarm",
):
    order = distribution_summary.sort_values(sort_col, ascending=ascending).index

    grouped = [
        (timestamp, group)
        for timestamp, group in choice_df.groupby(pd.Grouper(key=time_col, freq=freq))
        if not group.empty
    ]

    if not grouped:
        raise ValueError("No non-empty time groups were found. Check time_col and freq.")

    n_panels = len(grouped)
    ncols = min(3, n_panels)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.5 * ncols, 4.8 * nrows),
        squeeze=False
    )

    for ax in axes.flat[n_panels:]:
        ax.axis("off")

    for ax, (timestamp, group) in zip(axes.flat, grouped):
        mat = build_choice_matrix(group)
        mat = mat.reindex(index=order, columns=order)

        sns.heatmap(
            mat,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=0,
            vmax=1,
            cbar=False,
            square=True,
        )
        ax.set_title(f"{timestamp:%Y-%m-%d}  |  n={len(group)}")
        ax.set_xlabel("Alternative")
        ax.set_ylabel("Chosen")

    fig.tight_layout()
    return fig, axes


def add_date_bin(
    df,
    time_col,
    bin_edges,
    bin_labels=None,
    right=False,
):
    """
    Assign each row to a custom date bin.

    bin_edges: list of datetime-like boundaries, e.g.
        ["2024-01-01", "2024-01-05", "2024-01-10", "2024-01-20"]
    bin_labels: optional list of labels for each interval
    """
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    edges = pd.to_datetime(bin_edges)

    out["date_bin"] = pd.cut(
        out[time_col],
        bins=edges,
        labels=bin_labels,
        right=right,
        include_lowest=True,
    )
    return out


def plot_choice_heatmaps_by_bin(
    choice_df,
    distribution_summary,
    bin_col="date_bin",
    sort_col="expected_value",
    ascending=False,
    cmap="coolwarm",
):
    order = distribution_summary.sort_values(sort_col, ascending=ascending).index
    grouped = [
        (bin_name, group)
        for bin_name, group in choice_df.groupby(bin_col, observed=False)
        if pd.notna(bin_name) and not group.empty
    ]

    if not grouped:
        raise ValueError("No non-empty bins found. Check your bin edges and time column.")

    n_panels = len(grouped)
    ncols = min(3, n_panels)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.5 * ncols, 4.8 * nrows),
        squeeze=False
    )

    for ax in axes.flat[n_panels:]:
        ax.axis("off")

    for ax, (bin_name, group) in zip(axes.flat, grouped):
        mat = build_choice_matrix(group)
        mat = mat.reindex(index=order, columns=order)

        sns.heatmap(
            mat,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=0,
            vmax=1,
            cbar=False,
            square=True,
        )
        ax.set_title(f"{bin_name}  |  n={len(group)}")
        ax.set_xlabel("Alternative")
        ax.set_ylabel("Chosen")

    fig.tight_layout()
    return fig, axes

# ----------------------------
# Plot 2: actual vs experienced distributions
# ----------------------------
def build_experienced_distribution_table(
    distribution_df,
    id_col="chosen_distribution",
    sampled_col="sampled_magnitude",
):
    experienced = (
        distribution_df
        .groupby([id_col, sampled_col])
        .size()
        .rename("n")
        .reset_index()
    )
    experienced["probability"] = experienced["n"] / experienced.groupby(id_col)["n"].transform("sum")
    experienced = experienced.rename(columns={id_col: "id", sampled_col: "magnitude"})
    return experienced[["id", "magnitude", "probability", "n"]]

def plot_actual_vs_experienced_distributions(
    actual_distribution_table,
    experienced_distribution_table,
    id_order=None,
    ncols=2,
    bar=True,
):
    ids = (
        list(id_order)
        if id_order is not None
        else actual_distribution_table["id"].drop_duplicates().tolist()
    )

    n_panels = len(ids)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7 * ncols, 4 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True
    )

    for ax in axes.flat[n_panels:]:
        ax.axis("off")

    for ax, dist_id in zip(axes.flat, ids):
        actual = (
            actual_distribution_table[actual_distribution_table["id"] == dist_id]
            .groupby("magnitude")["probability"]
            .sum()
            .sort_index()
        )

        exp_subset = experienced_distribution_table[experienced_distribution_table["id"] == dist_id]

        experienced = (
            exp_subset
            .groupby("magnitude")["probability"]
            .sum()
            .sort_index()
        )

        values = np.array(sorted(set(actual.index).union(set(experienced.index))))
        x = np.arange(len(values))
        width = 0.38

        actual_vals = actual.reindex(values, fill_value=0).to_numpy()
        experienced_vals = experienced.reindex(values, fill_value=0).to_numpy()

        if bar:
            ax.bar(x - width / 2, actual_vals, width=width, label="Actual")
            ax.bar(x + width / 2, experienced_vals, width=width, label="Experienced")
        else:
            ax.plot(x, actual_vals, marker="o", label="Actual")
            ax.plot(x, experienced_vals, marker="o", label="Experienced")

        trial_count = exp_subset["n"].sum() if "n" in exp_subset.columns else len(exp_subset)

        ax.text(
            0.98,
            0.96,
            f"Trials: {trial_count}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none"),
        )

        ax.set_xticks(x)
        ax.set_xticklabels(values)
        ax.set_ylim(0, 1)
        ax.set_title(f"Distribution {dist_id}")
        ax.set_xlabel("Magnitude")
        ax.set_ylabel("Probability")
        ax.legend(frameon=False)

    fig.tight_layout()
    return fig, axes

# ----------------------------
# Example usage
# ----------------------------
import sqlite3
conn = sqlite3.connect("data.db")
df = pd.read_sql_query("SELECT * FROM data", conn)

magnitude_data = []
distribution_data = []
for block, blockdata in df.groupby("block"):
    assert isinstance(block, str), f"Expected block to be a string, got {type(block)}"
    if block.endswith('magnitude'):
        magnitude_data.append(blockdata)
    else:
        distribution_data.append(blockdata)
magnitude_data = pd.concat(magnitude_data)
distribution_data = pd.concat(distribution_data)
distribution_data = distribution_data.query('outcome == "choice"')
distribution_data = distribution_data.join(
    pd.DataFrame(distribution_data.data.apply(json.loads).tolist(), index=distribution_data.index)
)


config = get_latest_config(conn)
assert config is not None, "No configuration found in the database."

actual_distribution_table = build_actual_distribution_table(config)
distribution_summary = summarize_distributions(actual_distribution_table)

choice_df = prepare_choice_data(distribution_data, time_col="date")   # change if needed
experienced_distribution_table = build_experienced_distribution_table(
    choice_df,
    id_col="chosen_distribution",      # change if needed
    sampled_col="sampled_magnitude",
)

# Facet choice heatmaps by time window
fig1, axes1 = plot_choice_heatmaps_by_bin(
    choice_df=add_date_bin(choice_df, time_col="date", bin_edges=["2026-05-19", "2026-05-29", "2026-06-04", "2026-06-08", "2026-06-10", "2026-06-13"]),
    distribution_summary=distribution_summary,
    bin_col="date_bin",
    sort_col="expected_value",
    ascending=False,
)

# Separate plot: actual vs experienced distributions
fig2, axes2 = plot_actual_vs_experienced_distributions(
    actual_distribution_table=actual_distribution_table,
    experienced_distribution_table=experienced_distribution_table,
    id_order='abcd',
    ncols=2,
    bar=False
)

plt.show()