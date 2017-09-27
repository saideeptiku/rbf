"""
This file has functions that allow for Rank Based Fingerprinting. IPIN 11/12
Created By: Machaj, Brida, Piche
"""
import pandas as pd
from util_functions import print_df
import scipy.spatial.distance as scidist
import operator
from util_functions import select_data_filter, spearman_footrule_distance
import numpy as np


class RBF:
    def __init__(self, offline_df, input_labels, position_labels, output_labels):
        """
        init class with a offline_data.
        Missing access points should be marked with NaN

        offline_df: pandas data frame
        each row in the data frame represents, 
        RSSI values and position at a reference point

        input_labels: list
        labels for columns that represent WAP unique names.
        unique names can be MAC ids.

        output_labels: string
        label that represent the position, values should be integer

        missing_val_marker: 
        marker for missing data
        for UCI DB, WAP with value 100 are missing
        so missing_val_marker was set to 100
        set to None if not required

        member variables:
        self.ranked_in_df: ranked input data frame 

        self.in_df : clean input data frame with only input and output labels

        self.inpu_labels: input labels

        self.pos_lbl: output labels
        """
        # offline db
        self.offline_df = offline_df

        # init variables like offline rank vectors
        self.input_labels = input_labels
        self.pos_lbl = position_labels
        self.output_labels = output_labels

        # create df with sorted rows
        # see build_sorted_df for what this looks like
        self.offline_sorted = RBF.build_sorted_df(offline_df[list(input_labels) + [position_labels]],
                                                  input_labels,
                                                  position_labels)

    def get_projected_position(self, sorted_vector, k, similarity_measure="spearmans_footrule"):
        """
        get the projected position

        sorted_vector: list of WAP names as a sorted vector
        This is a ranked vector representation of WiFi fingerprint at an unknown position

        k: integer
        These are the k reference points used to calculate the reference position

        similarity_measure: string
        option for similarity measure. These are distance formulas to be used.
        possible inputs - > spearman, spearmans_footrule, jaccard coefficient, hamming, canberra.
        """

        # dict to store distance from each label position in
        # the key is the position label
        distance_dict = {}

        # iterate over each index of radio map DB and get distance
        for index in list(self.offline_sorted.index):
            train_sv, train_ps = RBF.get_sorted_vector_at_index(index,
                                                                self.offline_sorted,
                                                                self.input_labels,
                                                                self.pos_lbl,
                                                                is_raw_df=False)

            coord_db_vector, coord_online_vector = RBF.__build_coord_vectors__(train_sv,
                                                                               sorted_vector)

            # get distance between online and db coordinates
            # print(coord_db_vector, coord_online_vector, train_ps)
            distance_dict[train_ps] = RBF.__get_distance__(coord_db_vector,
                                                           coord_online_vector,
                                                           distance_function=similarity_measure)

        # identify k reference points with the smallest distance between online and dab vector
        # keep only first k elements in the distance dict
        distance_dict = RBF.__keep_nearest_K_in_dict__(k, distance_dict)

        # get weight dict from distance dict
        weight_dict = RBF.__get_weight_dict_from(distance_dict)

        # calculate position using weighted average
        pos_x, pos_y = self.__get_weighted_average_position__(weight_dict)

        return pos_x, pos_y

    def __get_weighted_average_position__(self, weight_dict):
        # make list of tuple such that
        # (x, y, weight)
        weight_tuple = []
        for pos in weight_dict.keys():
            x, y = self.get_xy_from_position_label(pos, self.offline_df)
            weight = weight_dict[pos]
            weight_tuple.append((x, y, weight))

        # calculate weighted average for x, y
        num_x = 0
        num_y = 0
        total_weight = 0
        for x, y, weight in weight_tuple:
            num_x = num_x + (x * weight)
            num_y = num_y + (y * weight)
            total_weight += weight

        pos_x = num_x/total_weight
        pos_y = num_y/total_weight

        return pos_x, pos_y

    @staticmethod
    def __get_weight_dict_from(distance_dict):
        """
        weights are inverse of distance
        :param distance_dict: 
        :return: 
        """
        weights = {}
        for pos in distance_dict.keys():
            try:
                weights[pos] = (1 / float(distance_dict[pos]))
            except ZeroDivisionError:
                print("\nwarning: there is zero distance between two vectors.\n",
                      "           are you using the same data for train and test?")
                weights[pos] = (1 / float(0.000000000000001))

        return weights

    @staticmethod
    def __keep_nearest_K_in_dict__(k, distance_dict):
        """
        keep only the K elements that have the smallest distance
        :param k: 
        :param distance_dict: {position: distance}
        :return: distance_dict with first K elements
        """
        tmp = {}
        for pos, dist in sorted(distance_dict.items(), key=operator.itemgetter(1))[:k]:
            tmp[pos] = dist
        return tmp

    @staticmethod
    def __get_distance__(coordinate_a, coordinate_b, distance_function=""):
        """
        get distance between two coordinates using given distance function
        :param coordinate_a: 
        :param coordinate_b: 
        :param distance_function: spearman, spearmans_footrule, jaccard coefficient, hamming, canberra
        :return: 
        """

        # possible options for measures
        distance_options = ["spearman",  # 0
                            "spearmans_footrule",  # 1
                            "jaccard_coeff",  # 2
                            "hamming",  # 3
                            "canberra",  # 4
                            ]

        if distance_function is distance_options[0]:
            # spearman is square of euclidean
            return scidist.sqeuclidean(coordinate_a, coordinate_b)

        elif distance_function is distance_options[1]:
            # spearman foot rule is element-wise displacement between two ranked vectors
            return spearman_footrule_distance(coordinate_a, coordinate_b)

        elif distance_function is distance_options[2]:
            print("this distance option not available yet! \n", distance_options[2])
            # TODO:
            exit()

        elif distance_function is distance_options[3]:
            print("this distance option not available yet! \n", distance_options[3])
            # TODO:
            exit()

        elif distance_function is distance_options[4]:
            print("this distance option not available yet! \n", distance_options[4])
            # TODO:
            exit()

        else:
            print("invalid distance options. use one from")
            print(distance_options)
            exit()

    @staticmethod
    def __build_coord_vectors__(db_vector_sorted, online_vector_sorted):
        """
        convert sorted vectors into the form of coordinates
        this is the main crux of the algorithm described in the paper
        :param db_vector_sorted: 
        :param online_vector_sorted: 
        :return: 
        """

        # coordinate vectors for online sorted vector and this training DB vector
        cord_db_vector = [0 for _ in db_vector_sorted]
        cord_online_vector = list(range(1, len(online_vector_sorted) + 1))

        # iterate over elements in sorted_vector
        for i in range(len(online_vector_sorted)):
            online_ap = online_vector_sorted[i]

            # get position of online_AP in this vector of DB
            try:
                train_sv_i = db_vector_sorted.index(online_ap)
            except ValueError:
                # an AP in online phase not present in offline phase
                # ignore this
                continue

            # rank is one more than index
            rank = cord_online_vector[i]

            # put rank at index in cord_db_vector
            # populate train vector
            cord_db_vector[train_sv_i] = rank

        # check if both co-ord lengths are same
        # Note: In the case one or more AP are not found in the training DB,
        # The rank vector found in the radio map is padded with 0,
        # to achieve the same length.

        if len(cord_db_vector) > len(cord_online_vector):
            # add zeros to online vector
            len_diff = len(cord_db_vector) - len(cord_online_vector)
            zero_vec = [0 for _ in range(len_diff)]
            cord_online_vector = cord_online_vector + zero_vec

        elif len(cord_db_vector) < len(cord_online_vector):
            # add zeros to db vector
            len_diff = len(cord_online_vector) - len(cord_db_vector)
            zero_vec = [0 for _ in range(len_diff)]
            cord_db_vector = cord_db_vector + zero_vec

        # finally check if they are equal and exit if not equal
        if len(cord_db_vector) != len(cord_online_vector):
            exit("error in making coordinates from vectors!")

        return cord_db_vector, cord_online_vector

    @staticmethod
    def build_sorted_df(offline_df, input_labels, pos_lbl):
        """
        convert simple offline data frame into sorted data frame like so:
        
        Input DF: self.offline_df should be declared at init
                  self.output and self.input labels should be declared 
        WAP_1  WAP_2  WAP_3  pos_label
          5      4      2        1
          7      2      5        2
          2      1      10       3     
        
        :return: Sorted data frame that looks like so, columns are ranks, NA have last rank
          1      2      3      pos_label
        WAP_1  WAP_2  WAP_3        1
        WAP_1  WAP_3  WAP_2        2
        WAP_3  WAP_1  WAP_2        3     
        """

        # rank the data frame, so each value is now a rank
        # this will rank low to high
        # we do this because WiFI strength is in DB
        # So more negative value is stronger
        ranked_df = offline_df[input_labels].rank(axis=1, method='dense') \
            .join(offline_df[pos_lbl])

        # print_df(ranked_df)

        # create an empty data frame with columns names as ranks and has position label
        # columns are ranks and the data entries are column labels from ranked data frame
        sorted_df = pd.DataFrame(index=ranked_df.index,
                                 columns=list(range(1, len(input_labels) + 1)) + [pos_lbl])

        # print_df(sorted_df)

        # iterate over each row of ranked data
        for index in list(ranked_df.index):
            for ap_name in input_labels:
                ap_rank = ranked_df.get_value(index, ap_name)
                pos_num = ranked_df.get_value(index, pos_lbl)

                if pd.isnull(ap_rank):
                    continue

                # populate the sorted data frame
                sorted_df.loc[index, ap_rank] = ap_name
                sorted_df.loc[index, pos_lbl] = pos_num

        # print_df(sorted_df)
        return sorted_df

    @staticmethod
    def get_sorted_vector_at_index(index, df, input_labels, pos_lbl, is_raw_df=True):
        """
        return ranked WiFi vector that resides at an index on the frame
        index starts with 1
        :param is_raw_df: the df is just raw data if True
                          the data is in sorted format as given in build_sorted_df(), if True
        :param input_labels: 
        :param df: data frame
        :param index: 
        :param pos_lbl: position label
        :return: 
        ranked WiFi vector that resides at an index on the frame, 
        position label
        """
        if index == 0:
            exit("index should begin with 1")

        if index not in list(df.index):
            exit("index out of bounds! should be between 1 to " + str(list(df.index)[-1]))

        # keep only useful rows
        # this is required if data is not a sorted df
        if is_raw_df:
            try:
                df = df[list(input_labels) + [pos_lbl]]
            except KeyError:
                # columns in Train not in test. add nan columns to test
                miss_cols = set(input_labels) - set(df.columns)

                for col in miss_cols:
                    df[col] = np.nan

                # columns added let's try that again
                df = df[list(input_labels) + [pos_lbl]]

        # get required row as df
        row_index_df = df.ix[[index]]

        # df with only one row
        if is_raw_df:
            sorted_df = RBF.build_sorted_df(row_index_df, input_labels, pos_lbl)
            # print_df(sorted_df)
        else:
            sorted_df = df

        # get position value
        pos = sorted_df.loc[index, pos_lbl]

        # drop pos column
        sorted_df = sorted_df.drop(pos_lbl, axis=1)

        # get vector as list
        sorted_vector = list(sorted_df.loc[index])

        # remove NaNs and return
        return [x for x in sorted_vector if pd.notnull(x)], pos

    def get_xy_from_position_label(self, pos_label, df):
        """
        get the x, y position from the df for given position number
        :param pos_label: 
        :param df: 
        :return: 
        """
        fdf = select_data_filter(df, {self.pos_lbl: pos_label})

        for index, row in fdf.iterrows():
            x = row[self.output_labels[0]]
            y = row[self.output_labels[1]]

            return x, y

        return None
