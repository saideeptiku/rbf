import pandas as pd
from tabulate import tabulate


def print_df(df):
    print(tabulate(df, headers='keys', tablefmt='psql'))


def remove_columns(df, col_list):
    return df.drop(col_list, axis=1)


def select_data_filter(df, filter_dict):
    """
    select data with filters

    df: pandas data frame

    filter_dict: dictionary
    keys are column names
    values are filter value
    """
    df_part = df.copy()
    for key in filter_dict.keys():
        # filter out rows one by one
        df_part = df_part.loc[(df_part[key] == filter_dict[key])]

    return df_part


def get_intersection_on(df_x, df_y, on_col_labels_list):
    """
    take intersection of two
    with labels in list
    most use full in getting common data points from two data frames
    used for getting wifi data for common points
    """

    return pd.merge(df_x, df_y, how='inner', on=on_col_labels_list)


def spearman_footrule_distance(coordinate_a, coordinate_b):
    dist = 0

    for i in range(len(coordinate_a)):
        dist = dist + abs(coordinate_b[i] - coordinate_a[i])

    return dist
