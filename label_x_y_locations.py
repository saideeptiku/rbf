# util functions
import pandas as pd
import math
import numpy as np
from util_functions import print_df


class LabelManager:
    def __init__(self, max_labels):

        # create empty place holders
        self.label_pos_dict = {}
        for r in range(max_labels):
            self.label_pos_dict[r + 1] = []
            # label from 1 -> row length

    def get_label(self, x, y):
        for label in self.label_pos_dict.keys():

            # loc list has tuples of positions in x-y
            loc_list = self.label_pos_dict[label]

            if (x, y) in loc_list:
                return label

        return None

    def add_to_label(self, label, x, y):
        try:
            # loc list has tuples of positions in x-y
            loc_list = self.label_pos_dict[label]

            loc_list.append((x, y))

            self.label_pos_dict[label] = loc_list
        except KeyError:
            print("Label manager has no label:", label)
            print("this can happen if you have more locations than rows in data?")
            print("their is an error in your code...aborting!")
            exit()

    def get_empty_label(self):

        for label in self.label_pos_dict.keys():

            if not list(self.label_pos_dict[label]):
                return label

        print("all labels are taken!")
        print("this can happen if you have more locations than rows in data?")
        print("their is an error in your code...aborting!")
        exit()


def create_xy_labels(train_df, test_df, avg_var, x_col, y_col):
    """
    this file will read the data and apply labels to x-y positions for label data.
    Assumes that data was collected on the same path and at around same points
    their maybe be slights variations in x-y positions
    label x-y positions that are similar in training and testing with integers 1, 2, 3

    :param y_col: 
    :param x_col: 
    :param avg_var: 
    :param test_df: 
    :param train_df: 
    :return: 
    """

    # which has more rows test or train
    if int(test_df.shape[0]) > int(train_df.shape[0]):
        max_rows = int(test_df.shape[0])
    else:
        max_rows = int(train_df.shape[0])

    # init label manager
    lab_man = LabelManager(max_rows)

    # for each row in test get row in train that is in avg_var range
    for test_row in test_df[[x_col, y_col]].iterrows():
        (test_x, test_y) = tuple(test_row[1])

        for train_row in train_df[[x_col, y_col]].iterrows():
            (train_x, train_y) = tuple(train_row[1])

            if np.isnan([test_x, test_y, train_x, train_y]).any():
                # print("skipping:", [test_x, test_y, train_x, train_y])
                continue

            if get_euclidean(test_x, test_y, train_x, train_y) < avg_var:

                # check if any of the above has a label already assigned in the past
                label_test = lab_man.get_label(test_x, test_y)

                label_train = lab_man.get_label(train_x, train_y)

                # if both have a label, they should be same
                if label_test is not None and label_train is not None:
                    if label_test != label_train:
                        # this should not happen
                        print("too close positions have different labels.\n",
                              "consider reducing avg variation or\n",
                              "check your data at", test_row.train_row)
                    else:
                        pass

                # if test has label and train does not, label train
                elif label_test is not None and label_train is None:
                    # apply label_test L -> (train_x, train_y)
                    lab_man.add_to_label(label_test, train_x, train_y)
                    pass

                # if train has label and test does not, label test
                elif label_train is not None and label_test is None:
                    # apply label L_train -> (test_x, test_y)
                    lab_man.add_to_label(label_train, test_x, test_y)
                    pass

                # if both do not have labels, add both to new label
                elif label_test is None and label_train is None:
                    # test and train to a new label
                    empty_label = lab_man.get_empty_label()

                    lab_man.add_to_label(empty_label, test_x, test_y)
                    lab_man.add_to_label(empty_label, train_x, train_y)
                    pass

    return lab_man


def get_euclidean(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def add_position_label_df(label_manager, df, x_col, y_col, new_pos_col,
                          verbose=False, get_count=False):
    """
    add labels to df positions

    :param y_col: 
    :param x_col: 
    :param new_pos_col: 
    :param label_manager: 
    :param df: 
    :return: 
    """

    unlabeled_count = 0

    # add new column to df
    df[new_pos_col] = np.nan

    # go to each row and add position label
    for index, row in df.iterrows():
        x = row[x_col]
        y = row[y_col]

        if np.isnan([x, y]).any():
            continue

        label = label_manager.get_label(x, y)

        if label is None and get_count:
            unlabeled_count += 1
            if verbose:
                print("-" * 100)
                print("There seems to be a data point that was not labeled")
                print(df.name, "df at index:", index)
                print("consider increasing average variation")
                print("-" * 100)
            
            continue

        # unfortunately, iterrows() gives a copy of df and not df itself
        df.loc[index, new_pos_col] = label

    if get_count:
        return df, unlabeled_count

    return df


# public
def average_duplicate_labels(df, label_col_name):
    """
    average out wifi strengths of duplicate labels
    :param label_col_name:
    :param df:
    :return:
    """
    # get set of all labels and iter
    # print(set(df[label_col_name].tolist()))
    for label in list(set(df[label_col_name].tolist())):
        #   select rows with same label

        df_sel = df.loc[(df[label_col_name] == label)]

        # print_df(df_sel)

        # avg out values and not label
        if df_sel.shape[0] > 1:
            # remove all rows with label
            df = df.drop(df_sel.index)

            # add row with averaged values
            # print(max(df.index))
            df.loc[max(df.index) + 1] = df_sel.mean(axis=0)

    # re-index df
    df.index = range(0, len(df.index))

    # return df
    return df


# public
def label_similar_locations(train_df, test_df,
                            x_col='x', y_col='y',
                            label_block_size=5,
                            new_pos_col="position_label",
                            average_duplicate_labels_train=True,
                            average_duplicate_labels_test=True, 
                            verbose=False, get_count=False):
    """
    label each x-y location as a position from 1 to N.
    This will convert data collected at each point into \
    data collected from a region or sample space.    
    You will get only samples spaces that are an intersection of training and testing

    :param average_duplicate_labels_test:
    :param average_duplicate_labels_train:
    :param label_block_size: minimum distance between two sample spaces
                             The distance is not in meters.
                             Try and calculate the average distance acrros points in DB.

    :param train_df: Data frame that will be used for training
    :param test_df: Data frame that will be used for testing
    :param x_col: column name of x position in data frame
    :param y_col: column name of y position in data frame
    :param new_pos_col: column name for the new position label column
    :return: 
    """
    fail_label_count = 0

    if verbose:
        print("*" * 100)
        print("*", " " * 31, "label similar locations: report ", " " * 31, "*")
        print("*" * 100)

    # read data training
    train_df.name = "Training"

    # read validation training
    test_df.name = "Testing"

    # create labels
    lab_man = create_xy_labels(
        train_df, test_df, label_block_size, x_col, y_col)

    # label the df
    train_df_lbld, a = add_position_label_df(lab_man, train_df,
                                          x_col, y_col, 
                                          new_pos_col, get_count=True, verbose=verbose)

    test_df_lbld, b = add_position_label_df(lab_man, test_df,
                                         x_col, y_col, 
                                         new_pos_col, get_count=True, verbose=verbose)

    fail_label_count = a + b

    # remove unlabeled rows
    train_df_lbld = train_df_lbld[pd.notnull(train_df_lbld[new_pos_col])]
    test_df_lbld = test_df_lbld[pd.notnull(test_df_lbld[new_pos_col])]

    # make sure they both have at least one row for each label
    # get common labels
    common_labels = set.intersection(set(train_df_lbld[new_pos_col]),
                                     set(test_df_lbld[new_pos_col]))

    # keep rows that have position label in common_label
    train_df_lbld = train_df_lbld[train_df_lbld[new_pos_col].isin(
        common_labels)]
    test_df_lbld = test_df_lbld[test_df_lbld[new_pos_col].isin(common_labels)]

    if verbose:
        print("*" * 100)

    if average_duplicate_labels_test:
        test_df_lbld = average_duplicate_labels(test_df_lbld, new_pos_col)

    if average_duplicate_labels_train:
        train_df_lbld = average_duplicate_labels(train_df_lbld, new_pos_col)

    if get_count:
        return train_df_lbld, test_df_lbld, fail_label_count    

    return train_df_lbld, test_df_lbld


if __name__ == "__main__":
    training_df = pd.read_csv("data/bcinfill-s6-run1.csv")
    validation_df = pd.read_csv("data/bcinfill-s6-run2.csv")

    label_similar_locations(training_df, validation_df)
