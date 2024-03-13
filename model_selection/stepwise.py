from collections import Counter
from functools import reduce
from math import floor
from operator import index
from re import U
from typing import Any, Callable, Self, Iterable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import simnet.similarity
import networkx as nx
from diffusion.simple_diff import get_diffusion_diff, random_rewire
from joblib import delayed, Parallel

class BackwardStepwise:
    def __init__(self,
                traits_df: pd.DataFrame,
                dynamic_df: pd.DataFrame,
                var_pool: list[str],
                similarity_metric: simnet.similarity.CosineSimilarity,*,
                num_var_remain: int = 10
                ):
        if list(traits_df.index) != list(dynamic_df.index):
            raise ValueError("Trait and dynamic dataframe should share same indices")
        self.traits_df = traits_df
        self.dynamic_df = dynamic_df
        self.dynamic_np = self.dynamic_df.to_numpy(dtype=np.float_)
        self.var_pool = var_pool
        self.similarity_metric = similarity_metric

        self.diffusion_diff_fn = lambda m1, m2: get_diffusion_diff(m1, m2, 0.5)     # default diffusion diff function


        self.remaining_vars  = num_var_remain

    def set_diffusion_diff_function(self, fn:Callable[[Any, Any], float]) -> Self:
        """
        Default is get_diffusion_diff with threshold = 0.5
        """
        self.diffusion_diff_fn = fn
        return self

    def get_origin_p_value(self, bin:int = 10):
        """
        Under the given list of variables, calculated the number of random rewiring in which the square error is smaller tha that of the baseline cosine similarity network.
        """

        baseline_sn = simnet.similarity.SimilarityNetwork(
                    self.traits_df,
                    self.similarity_metric,
                    self.var_pool
                    )
        baseline_sn.fit_transform()
        baseline_G = baseline_sn.network
        baseline_W = baseline_sn.adjacency_matrix


        def random_wire_diff(G:nx.Graph, th:float):
            # random_G = nx.random_degree_sequence_graph([d for n, d in G.degree()])    # This preserves the distribution
            random_G = nx.gnm_random_graph(len(G.nodes), len(G.edges))
            rand_adj_matrix = nx.adjacency_matrix(random_G).toarray().astype(np.float_)
            d = get_diffusion_diff(
                rand_adj_matrix,
                self.dynamic_np,
                th
            )
            return d

        th_change:list[float] = []
        for i in tqdm(range(1,bin)):
            th = float(i/bin)
            benchmark_diff = get_diffusion_diff(baseline_W, self.dynamic_np, th)
            list_of_random = [random_wire_diff(baseline_G, th) for _ in range(100)]
            th_change.append(
                sum(benchmark_diff > diff for diff in list_of_random) / len(list_of_random)
            )
        self.th_change = th_change
        return th_change

    def run(self,parallel = False):

        self.removal_dict:dict[str, int] = {}
        self.logging:list[dict[str, int]] = []
        cols_to_include_for_removal = self.var_pool.copy()

        while (n:=len(cols_to_include_for_removal) )>=self.remaining_vars:
            if not parallel:
                diff_dict = dict(self.compute_diffusion_diff(cat_col_i, col_name, cols_to_include_for_removal) for cat_col_i, col_name in enumerate(cols_to_include_for_removal))
            else:
                diff_dict = self.__parallel_runner(cols_to_include_for_removal)

            rank_by_se = dict(sorted(diff_dict.items(), key=lambda item: item[1]))
            worst_col =  list(rank_by_se.keys())[0] ## Remove this columns yields the best performance improve
            self.logging.append(rank_by_se)
            self.removal_dict[worst_col] = rank_by_se[worst_col]
            print(n, worst_col, rank_by_se[worst_col], sep='\t')
            cols_to_include_for_removal.remove(worst_col)

    def __parallel_runner(self, cols_to_include_for_removal:list[str]) -> dict[str, int]:

        num_jobs = -1
        diff_dict = dict(Parallel(n_jobs=num_jobs)(
                    delayed(self.compute_diffusion_diff)(cat_col_i, col_name, cols_to_include_for_removal)
                    for cat_col_i, col_name in enumerate(cols_to_include_for_removal)
                ))
        return diff_dict

    # Define a function that performs the computation for a single category
    def compute_diffusion_diff(self, cat_col_i:int, col_name:str , cols_to_include_for_removal:list[str]):
        col_to_try = cols_to_include_for_removal[:cat_col_i] + cols_to_include_for_removal[cat_col_i+1:]
        sn = simnet.similarity.SimilarityNetwork(
            self.traits_df,
            self.similarity_metric ,
            col_to_try)
        W = sn.fit_transform()

        diff = self.diffusion_diff_fn(W,
                                  self.dynamic_np)

        return col_name, diff

    def get_examiner(self):
        return StepwiseExaminer(self.logging)


class StepwiseExaminer:
    def __init__(self, log:list[dict[str, int]]):
        self.log = log
        self.orig_cols = list(log[0].keys())
        self.removal_dict = self.get_removal_dict()
        self.remove_order  = list(self.removal_dict.keys())       # the order of removal
    def get_removal_dict(self):
        """
        for each entry: key=remove this performs the best; value=the improved score
        """
        removal_dict:dict[str, float] = {}
        for iteration in self.log:
            rank_by_se = dict(sorted(iteration.items(), key=lambda item: item[1]))
            worst_col =  list(rank_by_se.keys())[0] ## Add this columns yields the best performance improve
            removal_dict[worst_col] = iteration[worst_col]
        return removal_dict

    def get_optimal_cols_set(self,orig_cols:list[str], n:int =1,):

        # removal_dict with min value
        ranked = list(
            dict(
            sorted(self.removal_dict.items(), key= lambda item: item[1])
             ).keys())[:n]

        population = []

        for col in ranked:                                  # col7, col4, col28, ...
            solution_set = [1]*len(orig_cols)          # [1 1 1 1 1 1 ]

            # when removed to col, who else is removed:
            iteration_when_removed = self.remove_order.index(col)
            who_else_is_removed = self.remove_order[:iteration_when_removed + 1]
            for col2 in who_else_is_removed:
                index_to_set_to_0 = orig_cols.index(col2)
                solution_set[index_to_set_to_0] = 0

            population.append(solution_set)

        return population



    def score_evolution(self, include_not_removed = False):

        display_order  = self.remove_order +  list(
                set(self.orig_cols) - set(self.remove_order) if include_not_removed else set()
            )
        score_lists = []
        for iteration in self.log:
            score_list = [iteration.get(col, 1) for col in display_order]
            score_lists.append(score_list)

        return score_lists, display_order

    def plot_score_evolution(self, title:str ="", ax: plt.Axes = None):

        if ax == None:
            fig, ax = plt.subplots(1,1)
        else:
            fig = ax.get_figure()

        ax.plot([131-n for n in range(len(self.removal_dict))], self.removal_dict.values())
        min_value_col, min_value = sorted(self.removal_dict.items(), key=lambda item: item[1])[0]
        min_value_x = 131 - list(self.removal_dict.values()).index(min_value)
        ax.hlines(y = min_value, xmin=0, xmax=131, linestyles='--', colors='grey')
        ax.annotate(f"{min_value_col}: {min_value}", (min_value_x, min_value+300))
        ax.scatter(x = min_value_x, y = min_value, c='r')
        ax.set_xlabel("Number of variables removed")
        ax.set_ylabel("Square Error")
        ax.set_title(title);
        return fig


    def plot_heatmap(self, top_n=10, include_not_removed = False, ax:plt.Axes = None, **kwargs):

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

    def compare_order_with(self, other:Self, ax:plt.Axes|None = None):

        if ax == None:
            fig, ax = plt.subplots(1,1)
        else:
            fig = ax.get_figure()

        A = self.score_evolution()[1]
        B = other.score_evolution()[1]
        # Plot elements of A on the left
        ax.scatter(np.zeros(len(A)), np.arange(len(A)), marker='o', color='blue', label='Left', s=2)

        # Plot elements of B on the right
        ax.scatter(np.ones(len(B)), np.arange(len(B)), marker='o', color='green', label='Right', s=2)

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
        ax.legend();

        return fig



def varibale_survival_analysis(dict_of_stepwise:dict[float, StepwiseExaminer], percentage:float = 0.5):
    num_of_var_to_extract:int = floor(
        len(list(dict_of_stepwise.values())[0].orig_cols) * percentage
        )

    orders_of_variables:dict[float, list[str]]= {k:
                                                 list(reversed(v.score_evolution(True)[1]))[:num_of_var_to_extract]
                                                 for k,v in dict_of_stepwise.items()}
    flatten_list_of_variables = reduce(lambda a,b: a+b ,orders_of_variables.values(), [])

    # counting
    return Counter(flatten_list_of_variables)
