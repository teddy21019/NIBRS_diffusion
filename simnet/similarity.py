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
        return self.similarity_matrix > self._th

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
    def __init__(self, df:pd.DataFrame, similarity_measure: Similarity, index_column:str, match_columns:list[str]):
        self._df: pd.DataFrame = df.set_index(index_column)[match_columns] # type: ignore
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

class ShapeMismatchError(ValueError):
    pass