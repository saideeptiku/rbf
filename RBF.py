"""
This file has functions that allow for Rank Based Fingerprinting. IPIN 11/12
Created By: Machaj, Brida, Piche
"""
import pandas as pd


class RBF:
    def __init__(self, offline_df, input_labels, output_labels, missing_val_marker=float('NaN')):
        """
        init class with a offline_data.

        offline data: pandas data frame
        each row in the dataframe represents, 
        RSSI values and position at a reference point

        wap_labels: list
        labels for columns that represent WAP unique names.
        unique names can be MAC ids.

        pos_labels: list
        labels that represent the position
        can be upto 3

        missing_val_marker: 
        marker for missing data
        for UCI DB, WAP with value 100 are missing
        so missing_val_marker was set to 100
        set to None if not required

        member variables:
        self.ranked_in_df: ranked input dataframe 

        self.in_df : clean input dataframe with only input and output labels

        self.in_labels: input labels

        self.out_labels: output labels
        """

        # drop useless information from raw data
        offline_df = offline_df[list(input_labels) + list(output_labels)]

        # process missing val marker
        # replace missing data with NaN
        if missing_val_marker and missing_val_marker is not float('NaN'):
            for col_name in input_labels:
                offline_df = offline_df.replace({col_name: {missing_val_marker: float('NaN')}})

        # convert raw data into sorted MAC addresses or unique names
        # create an identical ranked table with location and rank matrix
        self.ranked_in_df = offline_df[input_labels].rank(axis=1, method='dense').join(offline_df[output_labels])

        # init variables like offline rank vectors
        self.in_df = offline_df

        self.in_labels = input_labels

        self.out_labels = output_labels

    def build_rank_vectors_online(self, online_data):
        """
        create rank vectors for the online data

        online_data: list 1-D
        RSSI values for different WAP.
        The order of the RSSI values should match the RSSI labels.
        """

        # TODO: only process common points or ref points

        # TODO: create rank vectors for offline phase
        pass

    def get_projected_position(K, similarity_measure=""):
        """
        get the projected position

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
        pass

    def test(self):
        print()