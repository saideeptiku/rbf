"""
This file has functions that allow for Rank Based Fingerprinting. IPIN 11/12
Created By: Machaj, Brida, Piche
"""
import pandas as pd
from util_functions import print_df


class RBF:
    def __init__(self, offline_df, input_labels, output_label):
        """
        init class with a offline_data.
        Missing access points should be marked with NaN

        offline_df: pandas data frame
        each row in the dataframe represents, 
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

        # init variables like offline rank vectors
        self.input_labels = input_labels
        self.pos_lbl = output_label

        # create df with sorted rows
        self.offline_sorted = RBF.build_sorted_df(offline_df[list(input_labels) + [output_label]],
                                                  input_labels,
                                                  output_label)

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
    def get_sorted_vector_at_index(index, df, input_labels, pos_lbl):
        """
        return ranked WiFi vector that resides at an index on the frame
        index starts with 1
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

        df = df[list(input_labels) + [pos_lbl]]

        row_index_df = df.ix[[index]]

        # df with only one row
        sorted_df = RBF.build_sorted_df(row_index_df, input_labels, pos_lbl)

        pos = int(sorted_df.loc[1, pos_lbl])

        sorted_vector = list(sorted_df.loc[1])

        # remove NaNs and return
        return [int(x) for x in sorted_vector if pd.notnull(x)], pos

    def get_projected_position(self, sorted_vector, K, similarity_measure="spearmans_footrule"):
        """
        get the projected position

        ranked_vector: list of ints
        This is a ranked vector representation of WiFi fingerprint at an unknown position

        K: integer
        These are the K reference points used to calculate the reference position

        similarity_measure: string
        option for similarity measure. These are distance formulas to be used.
        possible inputs - > spearman, spearmans_footrule, jaccard coefficient, hamming, canberra.
        """
        # possible options for measures
        measures = ["spearman", "spearmans_footrule", "jaccard_coeff", "hamming", "canberra"]

        # TODO: check if given input is in the options

        # TODO: check if  K > 0 and is int

        # TODO: get distances from similarity measures

        # TODO: calculate projected position using value of K
        print(self.input_labels)
        pass

    def test(self):
        print()
