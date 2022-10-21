import pandas as pd
import numpy as np


def get_mean_pivot_table(data: pd.DataFrame, row_idx:int, col_idx:int, date_col:int, value_col:int, start:int, end:int):
    """
    data: dataframe
    row_idx: number of column for the row index of the pivot table
    col_idx: number of column for the column index of the pivot table
    date_col: number of column for the the date data
    value_col: number of column for the values to be used for calculating the mean
    start, finish: used as filters
    """

    groupby_cols = data.columns
    # del groupby_cols[date_col]
    # del groupby_cols[value_col]
    data = data[(data[data.columns[date_col]]>=start)&(data[data.columns[date_col]]<=end)]
    data = data.drop(columns=data.columns[date_col])
    data = data.replace(0, np.nan).groupby([groupby_cols[row_idx], groupby_cols[col_idx]]).mean(groupby_cols[value_col]).reset_index()
    pivot_table = data.pivot(index=groupby_cols[row_idx], columns=groupby_cols[col_idx], values=groupby_cols[value_col]).fillna(0).reset_index()

    return pivot_table