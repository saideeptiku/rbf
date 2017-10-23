"""
plotting functions for this project
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


def grouped_places_boxplot_devices(data_dict):
    """
    dict{place}{dev_train}{dev_test} => list
    """
    place_groups = list(data_dict.keys())

    train_devices = list(data_dict[place_groups[0]].keys())

    test_devices = list(data_dict[place_groups[0]][train_devices[0]].keys())

    print(place_groups, train_devices, test_devices)

    colors = ['red', 'green', 'blue']
