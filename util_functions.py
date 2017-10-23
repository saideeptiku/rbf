import pandas as pd
from tabulate import tabulate
from scipy.spatial.distance import euclidean


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


def remove_wap_in_column_name(df, wap_str="WAP"):
    """
    remove wap in the string name
    :param wap_str:
    :param df:
    :return:
    """
    new_cols = []
    for col_name in df.columns:
        new_cols.append(col_name.split(wap_str)[-1])

    df.columns = new_cols

    return df


def remove_extra_columns(main_df, sub_df):
    """
    make columns in sub_df a subset of columns in main_df
    :param main_df:
    :param sub_df:
    :return:
    """
    pass


def set_df_index_begin(df, start_index=1):
    """
    set start index of data frame
    :param df:
    :param start_index:
    :return:
    """
    df.index += start_index

    return df


def spearman_footrule_distance(coordinate_a, coordinate_b):
    dist = 0

    for i in range(len(coordinate_a)):
        dist = dist + abs(coordinate_b[i] - coordinate_a[i])

    return dist


def euclideans(list_tuples1, list_tuples2):
    dist = []

    for (u1, v1), (u2, v2) in zip(list_tuples1, list_tuples2):
        dist.append(euclidean((u1, v1), (u2, v2,)))

    return dist


def average_euclidean(list_tuples1, list_tuples2):
    dist = euclideans(list_tuples1, list_tuples2)

    return sum(dist) / len(dist)


def convert_unit(number, unit):
    return number / unit


def pick_rows_on_interval(df, interval, begin=None):
    """
    pick only rows at an SAMPLE_INTERVAL
    :param df:
    :param begin:
    :param interval:
    :return:
    """

    # create list of indices to drop
    if not begin:
        begin = int(min(df.index))

    end = int(max(df.index))

    keep_inds = []

    for i in range(begin, end):

        ind = interval * i

        if ind < end:
            keep_inds.append(ind)
        else:
            break

    drop_inds = list(set(df.index) - set(keep_inds))

    # drop indexes
    print("dropping", str(len(drop_inds)), "of", str(len(df.index)))
    df = df.drop(df.index[drop_inds])

    # re-index
    df.index = range(0, len(df.index))

    return df


def pick_label_num_by_multiple(df, mul, pos_label_col):

    picks = []

    for i in range(1, int(max(df[pos_label_col].tolist()))):
        if i % mul == 0:
            picks.append(i)

    return df[df[pos_label_col].isin(picks)]


def avg_error_factored(actual_pos, found_pos, conversion_factor=20):
    """
    get avg. error in meters
    """

    avg_err = convert_unit(average_euclidean(actual_pos,
                                             found_pos),
                           conversion_factor)

    return avg_err
