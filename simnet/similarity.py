from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from abc import ABC, abstractmethod
from functools import cached_property
import networkx as nx
import numpy as np
import pandas as pd

class Similarity(ABC):
    """
    Base class for similarity (or dissimilarity) metric
    """
    @abstractmethod
    def find_similarity_matrix(self, df:np.ndarray, weight):
        """
        Returns the booleans similarity matrix, given a criteria or threshold
        provided in the construction stage.
        """
        ...

class CosineSimilarity(Similarity):
    """
    Conduct similarity on all-numeric numpy matrix(or array).
    Only take hot-encoded from. All values should be numeric
    """
    def __init__(self, threshold:float):
        self._th = threshold

    def find_similarity_matrix(self, df:np.ndarray, weight:np.ndarray|None = None):
        if weight is None:
            weight = np.ones(df.shape[1]) / df.shape[1]

        w_sqrt = np.sqrt(weight)
        if df.shape[1] != w_sqrt.shape[0]:
            raise ShapeMismatchError(f"Shape for each row in df is {df.shape[1]}, different from shape of weight vector {w_sqrt.shape[0]}")

        self.similarity_matrix = cosine_similarity(np.multiply(df, w_sqrt))
        n_row = df.shape[0]
        return (self.similarity_matrix > self._th) * (np.ones((n_row, n_row)) - np.identity(n_row))   # no self loop

class MatchNumberSimilarity(Similarity):
    """
    Count whether two nodes have matching traits > X
    All columns should be binary numbers. In other words, a boolean matrix
    """
    def __init__(self, threshold:int):
        self._th = threshold

    def find_similarity_matrix(self, df, weight):
        _df_sparse = csr_matrix(df)
        self.similarity_matrix = _df_sparse@_df_sparse.T
        return self.similarity_matrix >= self._th


class SimilarityNetwork:
    """
    Combines a dataframe with a similarity measure. Specifying the index columns
    and columns for matching, the class converts the data into fully numerical
    format for the `Similarity` subclass to function.
    """
    def __init__(self, df:pd.DataFrame, similarity_measure: Similarity, match_columns:list[str], index_column:str=None):
        if index_column is not None:
            self._df: pd.DataFrame = df.set_index(index_column)[match_columns] # type: ignore
            self.index_columns = index_column
        else:
            self._df = df[match_columns]
            self.index_columns = df.index
        self._df = pd.get_dummies(self._df)
        self.feature_n = len(self._df.columns)
        self.sm = similarity_measure
        self.nodes = self._df.index.to_list()

    def fit_transform(self, weight = None):
        """
        Computes the adjacency matrix. Return the matrix in a sparse matrix format.
        """
        self.adjacency_matrix = csr_matrix(
            self.sm.find_similarity_matrix(self._df.to_numpy(), weight)
        )
        return self.adjacency_matrix

    @cached_property
    def network(self) -> nx.Graph:
        """
        Creates a graph from the sparse matrix generated through `fit_transform`.
        Node names are relabeled to the assigned index name in construction.
        """
        _G  = nx.from_numpy_array(self.adjacency_matrix)
        return nx.relabel_nodes(_G, {i:n for i, n in enumerate(self.nodes)})

    @property
    def degree(self):
        return list(np.sum(self.adjacency_matrix.toarray(), axis = 0))
    def export(self, path:Path|str, file_name:str, node_df:pd.DataFrame|None=None):
        """
        param
        ------
        path: Path of export. Should be folder
        node_df: attribute for each node. Must have an columns that matches self.index_column for merging
        """
        rows = self.adjacency_matrix.tocoo().row
        cols = self.adjacency_matrix.tocoo().col
        edge_col = pd.DataFrame({'Source': rows, 'Target':cols})

        node_id_name_df = pd.DataFrame([{'id': i, self.index_columns:c} for i,c in enumerate(self.nodes)])
        if node_df is not None:
            nodes = node_id_name_df.merge(node_df, how = 'left', on = self.index_columns)
        else:
            nodes = node_id_name_df

        edge_col.to_csv(f"{path}/{file_name}_edge.csv", index = False)
        nodes.to_csv(f"{path}/{file_name}_node.csv", index = False)


class ShapeMismatchError(ValueError):
    pass