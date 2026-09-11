import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Dict, List, Tuple
def construct_colourmap(labels: List[str] | np.ndarray, cmap: str, values=None) -> Dict[str, Tuple[float,float,float]]:
    nlevels = len(labels)
    if values is None:
        values = np.linspace(0, 1, nlevels)
    return dict(zip(labels, mpl.colormaps[cmap](values)))

import seaborn as sns
def plot_distribution_choices(data, labels, dist_info):
    plt.style.use('dark_background')
    # custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    # sns.set_theme(style="ticks", rc=custom_params)
    fig, axes = plt.subplots(4,3,sharex=True,sharey=True)
    # distribution_cmap = construct_colourmap(labels, 'cool')
    # distribution_cmap = dict(zip(labels, mpl.colormaps.get_cmap('tab10')(np.arange(len(labels)))))
    for distribution, ax in zip(labels, axes.flatten()):
        mask = data['distribution_options_sorted'].map(lambda x: distribution in x)
        dist_data = data.loc[mask]
        dist_data['other_distribution'] = dist_data['distribution_options_sorted'].map(lambda x: x[0] if x[1] == distribution else x[1])
        dist_data['distribution_chosen'] = dist_data['chosen_distribution'] == distribution
        choice_prob = dist_data.groupby(['date','other_distribution']).distribution_chosen.mean()
        from matplotlib.colors import CenteredNorm
        distribution_cmap = construct_colourmap(labels, 'bwr', values = CenteredNorm()(dist_info - dist_info.loc[distribution]))

        sns.lineplot(data=choice_prob.reset_index(), x='date', y='distribution_chosen', hue='other_distribution', palette=distribution_cmap, ax=ax, legend=False)
        ax.set_title(distribution)
        ax.set_ylim(0, 1)
        ax.axhline(0.5)
        ax.tick_params(labelrotation=45)
        # legend_elements = [Line2D([0],[0], color=distribution_cmap[label], label=f"{label} = {dist_info.loc[label].item():.2f}") for label in labels]
        # ax.legend(handles=legend_elements, loc='lower left', ncols=3,fontsize='xx-small')
    legend_ax = axes.flatten()[-1]
    from matplotlib.lines import Line2D
    
    legend_elements = [Line2D([0],[0], color=distribution_cmap[label], label=f"{label} = {dist_info.loc[label].item():.2f}") for label in labels]
    legend_ax.legend(handles=legend_elements, loc='center')
    legend_ax.axis('off')