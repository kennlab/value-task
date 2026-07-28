import json
import math
import sqlite3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

sns.set_theme(style="whitegrid")
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'

# ----------------------------
# Helpers: config and tables
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
    stimulus_map = config["stimulus_sets"]["3"]

    for dist in config["distribution_cues"].values():
        dist_name = dist.get("name", dist["id"])
        rows.append(pd.DataFrame({
            "id": dist["id"],
            "name": dist_name,
            "magnitude": dist["magnitude_values"],
            "probability": dist["probabilities"],
            "dist_image": dist["image"],
            "mag_image": [stimulus_map[str(mag)] for mag in dist["magnitude_values"]],
        }))

    return pd.concat(rows, ignore_index=True)


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


def add_date_bin(
    df,
    time_col,
    bin_edges,
    bin_labels=None,
    right=False,
):
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
    experienced["probability"] = (
        experienced["n"] / experienced.groupby(id_col)["n"].transform("sum")
    )
    experienced = experienced.rename(columns={id_col: "id", sampled_col: "magnitude"})
    return experienced[["id", "magnitude", "probability", "n"]]


# ----------------------------
# Helpers: image drawing
# ----------------------------
def _load_image(img_ref):
    if img_ref is None or (isinstance(img_ref, float) and np.isnan(img_ref)):
        return None
    return Image.open(img_ref)


def _add_image_box(ax, img_ref, xy, xycoords, zoom=0.22):
    img = _load_image(img_ref)
    if img is None:
        return

    ab = AnnotationBbox(
        OffsetImage(img, zoom=zoom),
        xy,
        xycoords=xycoords,
        frameon=False,
        box_alignment=(0.5, 0.5),
    )
    ab.set_clip_on(False)
    ax.add_artist(ab)


def _distribution_maps(actual_distribution_table):
    dist_to_image = (
        actual_distribution_table[["id", "dist_image"]]
        .drop_duplicates(subset=["id"])
        .set_index("id")["dist_image"]
        .to_dict()
    )

    dist_to_name = (
        actual_distribution_table[["id", "name"]]
        .drop_duplicates(subset=["id"])
        .set_index("id")["name"]
        .to_dict()
    )

    mag_to_image = {}
    for dist_id, sub in actual_distribution_table.groupby("id"):
        mag_to_image[dist_id] = (
            sub[["magnitude", "mag_image"]]
            .drop_duplicates(subset=["magnitude"])
            .set_index("magnitude")["mag_image"]
            .to_dict()
        )

    return dist_to_image, dist_to_name, mag_to_image


def _decorate_heatmap_axes_with_images(ax, order, dist_to_image):
    n = len(order)
    x_positions = np.arange(n) + 0.5
    y_positions = np.arange(n) + 0.5

    ax.set_xticks(x_positions)
    ax.set_yticks(y_positions)
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_yticklabels(order, rotation=0)

    ax.tick_params(axis="x", pad=24)
    ax.tick_params(axis="y", pad=34)

    for x, dist_id in zip(x_positions, order):
        _add_image_box(
            ax,
            dist_to_image.get(dist_id),
            (x, -0.18),
            xycoords=("data", "axes fraction"),
            zoom=0.14,
        )

    for y, dist_id in zip(y_positions, order):
        _add_image_box(
            ax,
            dist_to_image.get(dist_id),
            (-0.15, y),
            xycoords=("axes fraction", "data"),
            zoom=0.14,
        )


def _decorate_distribution_axis_with_mag_images(ax, values, mag_images, show_tick_labels=False, offset=.5):
    x = np.arange(len(values)) + offset
    ax.set_xticks(x)

    if show_tick_labels:
        ax.set_xticklabels([str(v) for v in values], rotation=0)
        ax.tick_params(axis="x", labelbottom=True, pad=18)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", labelbottom=False)

    for xi, img_ref in zip(x, mag_images):
        _add_image_box(
            ax,
            img_ref,
            (xi, -0.18),
            xycoords=("data", "axes fraction"),
            zoom=0.18,
        )


# ----------------------------
# Plot 1: choice heatmaps by bin + legend axis
# ----------------------------
def _plot_heatmap_legend(ax, order, actual_distribution_table, distribution_summary, sort_col="expected_value", ascending=False):
    dist_to_image, dist_to_name, _ = _distribution_maps(actual_distribution_table)
    summary_sorted = distribution_summary.sort_values(sort_col, ascending=ascending)

    ax.axis("off")
    ax.set_title("Distribution key", loc="left", fontsize=12, pad=8)

    y = 0.92
    dy = 0.14 if len(summary_sorted) <= 6 else 0.11

    for dist_id in summary_sorted.index:
        name = dist_to_name.get(dist_id, dist_id)
        ev = summary_sorted.loc[dist_id, "expected_value"]
        var = summary_sorted.loc[dist_id, "variance"]

        _add_image_box(
            ax,
            dist_to_image.get(dist_id),
            (0.04, y),
            xycoords=ax.transAxes,
            zoom=0.16,
        )

        ax.text(
            0.10,
            y,
            f"{name}   |   EV = {ev:.2f}   |   Var = {var:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10,
        )
        y -= dy


def plot_choice_heatmaps_by_bin(
    choice_df,
    distribution_summary,
    actual_distribution_table,
    bin_col="date_bin",
    sort_col="expected_value",
    ascending=False,
    cmap="coolwarm",
):
    order = distribution_summary.sort_values(sort_col, ascending=ascending).index.tolist()

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

    figsize=(5 * ncols, 4 * (nrows))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
    )
    flat_axes = axes.flatten()
    for ax in flat_axes[n_panels:-1]:
        ax.axis("off")
    legend_ax = flat_axes[-1]

    dist_to_image, _, _ = _distribution_maps(actual_distribution_table)

    for ax, (bin_name, group) in zip(flat_axes, grouped):
        mat = build_choice_matrix(group).reindex(index=order, columns=order)

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

        _decorate_heatmap_axes_with_images(ax, order, dist_to_image)

        ax.set_title(f"{bin_name}  |  n={len(group)}", fontsize=11, pad=14)
        ax.set_xlabel("")
        ax.set_ylabel("")

    _plot_heatmap_legend(legend_ax, order, actual_distribution_table, distribution_summary, sort_col=sort_col, ascending=ascending)

    fig.subplots_adjust(bottom=0.04, left=0.08, right=0.98, top=0.96)
    return fig, axes, legend_ax


# ----------------------------
# Plot 2: actual vs experienced distributions
# ----------------------------
def plot_actual_vs_experienced_distributions(
    actual_distribution_table,
    experienced_distribution_table,
    id_order=None,
    ncols=2,
    bar=True,
):
    dist_to_image, dist_to_name, mag_to_image = _distribution_maps(actual_distribution_table)

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
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for ax in axes.flat[n_panels:]:
        ax.axis("off")

    for idx, (ax, dist_id) in enumerate(zip(axes.flat, ids)):
        actual_subset = actual_distribution_table[actual_distribution_table["id"] == dist_id]
        exp_subset = experienced_distribution_table[experienced_distribution_table["id"] == dist_id]

        actual = (
            actual_subset
            .groupby("magnitude")["probability"]
            .sum()
            .sort_index()
        )

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

        dist_image = dist_to_image.get(dist_id)
        dist_name = dist_to_name.get(dist_id, dist_id)

        if dist_image is not None:
            _add_image_box(
                ax,
                dist_image,
                (0.5, 1.16),
                xycoords=ax.transAxes,
                zoom=0.24,
            )

        ax.text(
            0.98,
            0.96,
            f"Trials: {trial_count}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                alpha=0.8,
                edgecolor="none",
            ),
        )

        row = idx // ncols
        is_bottom_row = row == nrows - 1
        mag_images = [mag_to_image.get(dist_id, {}).get(m, None) for m in values]
        _decorate_distribution_axis_with_mag_images(
            ax,
            values,
            mag_images,
            show_tick_labels=is_bottom_row,
            offset=0
        )

        ax.set_ylim(0, 1)
        ax.set_title(f"Distribution {dist_name}", fontsize=11, pad=34, y=1.02)
        ax.set_ylabel("Probability")
        ax.legend(frameon=False)

    fig.supxlabel("Magnitude", y=0.03)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig, axes


# ----------------------------
# Example usage
# ----------------------------
def main():
    conn = sqlite3.connect("data.db")
    df = pd.read_sql_query("SELECT * FROM data", conn)

    magnitude_data = []
    distribution_data = []

    for block, blockdata in df.groupby("block"):
        assert isinstance(block, str), f"Expected block to be a string, got {type(block)}"
        if block.endswith("magnitude"):
            magnitude_data.append(blockdata)
        else:
            distribution_data.append(blockdata)

    magnitude_data = pd.concat(magnitude_data, ignore_index=True)
    distribution_data = pd.concat(distribution_data, ignore_index=True)

    distribution_data = distribution_data.query('outcome == "choice"').copy()
    distribution_data = distribution_data.join(
        pd.DataFrame(distribution_data.data.apply(json.loads).tolist(), index=distribution_data.index)
    )

    config = get_latest_config(conn)
    assert config is not None, "No configuration found in the database."

    actual_distribution_table = build_actual_distribution_table(config)
    distribution_summary = summarize_distributions(actual_distribution_table)

    choice_df = prepare_choice_data(distribution_data, time_col="date")

    choice_df_binned = add_date_bin(
        choice_df,
        time_col="date",
        bin_edges=[
            "2026-05-19",
            "2026-05-29",
            "2026-06-04",
            "2026-06-08",
            "2026-06-10",
            "2026-06-15",
        ],
        bin_labels=[
            "May 19 to May 28",
            "May 29 to Jun 3",
            "Jun 4 to Jun 7",
            "Jun 8 to Jun 9",
            "Jun 10 to Jun 15",
        ],
        right=True,
    )

    experienced_distribution_table = build_experienced_distribution_table(
        choice_df,
        id_col="chosen_distribution",
        sampled_col="sampled_magnitude",
    )

    fig1, axes1, legend_ax1 = plot_choice_heatmaps_by_bin(
        choice_df=choice_df_binned,
        distribution_summary=distribution_summary,
        actual_distribution_table=actual_distribution_table,
        bin_col="date_bin",
        sort_col="expected_value",
        ascending=False,
    )
    fig1.tight_layout()

    fig2, axes2 = plot_actual_vs_experienced_distributions(
        actual_distribution_table=actual_distribution_table,
        experienced_distribution_table=experienced_distribution_table,
        id_order=distribution_summary.sort_values("expected_value", ascending=False).index.tolist(),
        ncols=2,
        bar=False,
    )
    fig2.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()