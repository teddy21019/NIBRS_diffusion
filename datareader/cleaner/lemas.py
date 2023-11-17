import pandas as pd
import numpy as np
from typing import TypeAlias, Callable

DF: TypeAlias = pd.DataFrame
PipeLine: TypeAlias = Callable[[DF], DF]

"""
    dtype tuning
"""
def categorize_strata(df:DF) -> DF:
    return df.astype({"STRATA": 'category'})

def negative_to_na(df: DF) -> DF:
    return df.replace(-8, np.nan).replace(-9, np.nan)

def sumup_num(df:DF, num)-> DF:
    return (df== num).sum()

def count_na_each_col(df:DF) -> DF:
    return df.isna().sum()

def merge_col_info_with_col_type(dtype_info:pd.Series, col_of_col:str='col')->PipeLine:
    dtype_df = dtype_info.rename("dtype")
    def merge_fn(df:DF) -> DF:
        return df.merge(dtype_df, left_on=col_of_col, right_index=True, how='left')

    return merge_fn

def count_89_and_empty(df:DF)->DF:
    count_8 = sumup_num(df, -8)
    count_9 = sumup_num(df, -9)
    empty_str = sumup_num(df, ' ')

    missing_on_89 = (pd.DataFrame({"neg_8": count_8, "neg_9":count_9, "empty_str":empty_str})
            .reset_index()
            .rename(columns={"index":"col"})
            .pipe(merge_col_info_with_col_type(df.dtypes)))
    return missing_on_89

def remove_col_with_many_89(df:DF, keep_8=True) -> DF:
    NROW = len(df)
    missing_on_89_df = count_89_and_empty(df)
    query_str =  "neg_9/@NROW<0.3" if keep_8 else "neg_8/@NROW<0.3 and neg_9/@NROW<0.3"
    keeping_cols = missing_on_89_df.query(query_str).col
    return df[keeping_cols]
"""
    Filter By Columns
"""
def remove_specify_name(df:DF)-> DF:
    cols_without_sp = [c for c in df.columns if not c.endswith('_SP')]
    return df[cols_without_sp]

def remove_flags(df:DF)-> DF:
    cols_without_flag = [c for c in df.columns if not (c.endswith('_FLAG') or c.startswith('FLAG'))]
    return df[cols_without_flag]

"""Plotting"""

def plot_89(df:DF):
    return df.query("neg_8 + neg_9>0").plot(x="neg_8", y="neg_9", kind='scatter')

"""
    By strata stat for each category col
"""

