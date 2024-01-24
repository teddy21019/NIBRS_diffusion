from typing import Self
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class Stepwise:
    def __init__(self, log:list[dict[str, int]]):
        self.log = log
        self.orig_cols = list(log[0].keys())
        self.remove_order  = self.get_removal_order()       # the order of removal
    def get_removal_order(self):
        remove_order = []
        for iteration in self.log:
            rank_by_se = dict(sorted(iteration.items(), key=lambda item: item[1]))
            worst_col =  list(rank_by_se.keys())[0] ## Add this columns yields the best performance improve
            remove_order.append(worst_col)
        return remove_order

    def score_evolution(self, include_not_removed = False):

        display_order  = list(
                set(self.orig_cols) - set(self.remove_order) if include_not_removed else set()
            ) + self.remove_order

        score_lists = []
        for iteration in self.log:
            score_list = [iteration.get(col, 1) for col in display_order]
            score_lists.append(score_list)

        return score_lists, display_order

    def plot_heatmap(self, top_n=10, include_not_removed = False, ax:plt.Axes = None, **kwargs):
        import seaborn as sns

        if ax == None:
            fig, ax = plt.subplots(1,1)
        else:
            fig = ax.get_figure()

        score, col_names = self.score_evolution(include_not_removed)
        sns.heatmap(
            np.array(score[:top_n]).T,
            yticklabels=col_names,
            ax=ax,
            **kwargs
            );
        return fig

    def compare_order_with(self, other:Self, ax:plt.Axes|None):

        if ax == None:
            fig, ax = plt.subplots(1,1)
        else:
            fig = ax.get_figure()

        A = self.score_evolution()[1]
        B = other.score_evolution()[1]
        # Plot elements of A on the left
        ax.scatter(np.zeros(len(A)), np.arange(len(A)), marker='o', color='blue', label='0.8', s=2)

        # Plot elements of B on the right
        ax.scatter(np.ones(len(B)), np.arange(len(B)), marker='o', color='green', label='0.7', s=2)

        # Draw lines between similar elements
        for a in A:
            try:
                idx_a = A.index(a)
                idx_b = B.index(a)
                ax.plot([0, 1], [idx_a, idx_b], color='gray', linestyle='-')
            except:
                continue
        # Set labels and legend
        ax.set_yticks(np.arange(max(len(A), len(B))))
        ax.set_yticklabels(A)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['A', 'B'])
        ax.legend()

        return fig