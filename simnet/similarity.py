from pathlib import Path
from tkinter import W
from typing import Callable
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from abc import ABC, abstractmethod
from functools import cached_property, partial
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
from numba import njit

from diffusion.simple_diff import get_diffusion_diff, get_diffusion_diff_dynamic

class Similarity(ABC):
    """
    Base class for similarity (or dissimilarity) metric
    """
    @abstractmethod
    def find_similarity_matrix(self, df:np.ndarray, weight) -> npt.NDArray[np.bool_]:
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

    def find_similarity_matrix(self, df: np.ndarray, weight) -> npt.NDArray[np.bool_]:
        self.adjacency_matrix, self.similarity_matrix =  self.__find_similarity_matrix(self._th, df, weight)
        return self.similarity_matrix

    @staticmethod
    @njit
    def __find_similarity_matrix(_th, df:np.ndarray, weight:np.ndarray|None = None) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.bool_]]:
        if weight is None:
            weight = np.ones(df.shape[1]) / df.shape[1]

        w_sqrt = np.sqrt(weight)
        # if df.shape[1] != w_sqrt.shape[0]:
        #     raise ShapeMismatchError(f"Shape for each row in df is {df.shape[1]}, different from shape of weight vector {w_sqrt.shape[0]}")

        A = np.multiply(df, w_sqrt)
        similarity: np.ndarray = np.dot(A, A.T)

        # inverse squared magnitude
        inv_square_mag = 1 /  np.diag(similarity)

        # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
        inv_square_mag[np.isinf(inv_square_mag)] = 0

        # inverse of the magnitude
        inv_mag:npt.NDArray[np.float_]  = np.sqrt(inv_square_mag)

        # cosine similarity (elementwise multiply by inverse magnitudes)
        cosine = similarity * inv_mag
        similarity_matrix = cosine.T * inv_mag

        n_row = df.shape[0]
        return similarity_matrix, (similarity_matrix > _th) * (np.ones((n_row, n_row)) - np.identity(n_row))   # no self loop


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

    def fit_transform(self, weight = None):
        """
        Computes the adjacency matrix. Return the matrix in a sparse matrix format.
        """
        self.adjacency_matrix = self.sm.find_similarity_matrix(self._df.to_numpy(), weight)
        return self.adjacency_matrix

    @cached_property
    def nodes(self) -> list[str]:
        return self._df.index.to_list()

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


class DiffusionDifferent:
    def __init__(self, df:pd.DataFrame, similarity_measure:Similarity, dynamic: pd.DataFrame, columns:list[str]):
        self.df = df
        self.similarity = similarity_measure
        self.dynamic_np = dynamic.to_numpy(dtype=np.float_)
        self.columns = columns

    def fit_diff(self, diff_diff_fn:Callable[[np.ndarray, np.ndarray], np.ndarray] = None):
        if diff_diff_fn is None:
            diff_diff_fn = partial(get_diffusion_diff_dynamic, th = 0.5)

        sn = SimilarityNetwork(
            df= self.df,
            similarity_measure=self.similarity,
            match_columns=self.columns
        )
        W = sn.fit_transform()

        return diff_diff_fn(W, self.dynamic_np)


    def fit(self, diff_diff_fn:Callable[[np.ndarray, np.ndarray], np.ndarray] = None):
        return int(np.sum(self.fit_diff(diff_diff_fn)))


class ShapeMismatchError(ValueError):
    pass