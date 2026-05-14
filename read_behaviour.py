
import pandas as pd
import sqlite3
import json
import seaborn as sns
import matplotlib.pyplot as plt

def draw_heatmap(data, **kwargs):
    # Pivot for heatmap values
    value_pivot = data.pivot(
        index="locs",
        columns="abs_value_diff",
        values="accuracy_mean"
    )

    # Pivot for annotations
    annot_pivot = (
        data.assign(
            annot=lambda d: d.apply(
                lambda r: f"{r['accuracy_mean']:.2f}\n(n={int(r['n_trials'])})",
                axis=1
            )
        )
        .pivot(
            index="locs",
            columns="abs_value_diff",
            values="annot"
        )
    )

    sns.heatmap(
        value_pivot,
        annot=annot_pivot,
        fmt="",
        vmin=0,
        vmax=1,
        cmap="viridis",
        cbar=False,
        **kwargs
    )
def get_value_difference(x):
    x = json.loads(x)
    op_left, op_right = x['magnitudes']
    loc1, loc2 = x['locations']
    if loc1[0] > loc2[1]:
        op_left, op_right = op_right, op_left
    return op_left - op_right
def get_chose_left(x):
    x = json.loads(x)
    loc1, loc2 = x['locations']
    return (loc1[0] < loc2[1]) == (x['chosen'] == 'option1')

def get_data():
    conn = sqlite3.connect(r"C:\Users\akeeler\data.db")
    query = """
    SELECT * 
    FROM data 
    WHERE outcome in ("correct", "incorrect")
    """
    data = pd.read_sql(query, conn)
    data['datetime']=pd.to_datetime(data['date'] + ' ' + data['time'])
    condition_split = data['condition'].str.split('_')
    data['stimulus_set'] = condition_split.str[0]
    data['values'] = condition_split.str[2]
    data['locs'] = condition_split.str[3]
    data['option1_value'], data['option2_value'] = data['values'].str[1].astype(int), data['values'].str[3].astype(int)
    data['option1_loc'], data['option2_loc'] = data['locs'].str[3].astype(int), data['locs'].str[5].astype(int)
    data['abs_value_diff'] = abs(data['option1_value'] - data['option2_value'])
    data['accuracy'] = data['outcome'].map({'correct': 1, 'incorrect': 0})
    data['value_left_minus_right'] = data['data'].apply(get_value_difference)
    data['chose_left'] = data['data'].apply(get_chose_left)
    return data

def plot_choice_probability(data, period="4D"):
    # Create 2-day bins
    data["datetime_bin"] = data["datetime"].dt.floor(period)
    fg = sns.relplot(
        data=data,
        x='value_left_minus_right',
        y='chose_left',
        col='datetime_bin',
        col_wrap=3,
        kind='line',
    )
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.regplot(
        data=data,
        x='value_left_minus_right',
        y='chose_left',
        logistic=True,
        scatter=False,
        ci=None,
        ax=ax
    )
    sns.scatterplot(
        data=data.groupby(['value_left_minus_right', 'datetime_bin'])['chose_left'].mean().reset_index(),
        x='value_left_minus_right',
        y='chose_left',
        ax=ax,
        alpha=0.5,
    )
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel("Value Left - Value Right")
    ax.set_ylabel("P(Choose Left)")
    ax.axhline(0.5, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')

def plot_heatmaps(data, period="2D"):
    # Create bins
    data["datetime_bin"] = data["datetime"].dt.floor(period)

    # Aggregate mean accuracy + trial counts
    agg = (
        data.groupby(["datetime_bin", "locs", "abs_value_diff"])
            .agg(
                accuracy_mean=("accuracy", "mean"),
                n_trials=("accuracy", "size")
            )
            .reset_index()
    )

    # Facet grid
    g = sns.FacetGrid(
        agg,
        col="datetime_bin",
        col_wrap=3,
        height=4,
        sharex=False,
        sharey=False
    )
    g.map_dataframe(draw_heatmap)
    # Add total trials to facet titles
    total_trials = (
        data.groupby("datetime_bin")
            .size()
            .to_dict()
    )

    for ax, dt_bin in zip(g.axes.flat, g.col_names):
        n = total_trials.get(pd.Timestamp(dt_bin), 0)
        ax.set_title(f"{dt_bin}\nTotal N={n}")
    # Shared colorbar
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    g.figure.colorbar(sm, ax=g.axes, label="Mean Accuracy")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze TwoAFC behavior data")
    parser.add_argument("--period", type=str, default="2D", help="Time period for binning (e.g., '2D' for 2 days)")
    args = parser.parse_args()
    data = get_data()
    plot_choice_probability(data, period=args.period)
    plt.show()
    plot_heatmaps(data, period=args.period)
    plt.show()