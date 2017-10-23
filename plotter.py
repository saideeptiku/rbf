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

    c = ['red', 'green', 'blue']

    # make data sets
    A = [
        data_dict[place_groups[0]][train_devices[0]][test_devices[0]],
        data_dict[place_groups[1]][train_devices[0]][test_devices[0]],
        data_dict[place_groups[2]][train_devices[0]][test_devices[0]]
    ]

    B = [
        data_dict[place_groups[0]][train_devices[0]][test_devices[1]],
        data_dict[place_groups[1]][train_devices[0]][test_devices[1]],
        data_dict[place_groups[2]][train_devices[0]][test_devices[1]]
    ]

    C = [
        data_dict[place_groups[0]][train_devices[0]][test_devices[2]],
        data_dict[place_groups[1]][train_devices[0]][test_devices[2]],
        data_dict[place_groups[2]][train_devices[0]][test_devices[2]]
    ]

    data = [A, B, C]
    sp = 0
    for i, d in enumerate(data):
        plt.boxplot(d, positions=[1+sp, 3.5+sp, 6+sp], notch=True, patch_artist=True,
                    boxprops=dict(facecolor=c[i], color=c[i]),
                    capprops=dict(color=c[i]),
                    whiskerprops=dict(color=c[i]),
                    flierprops=dict(color=c[i], markeredgecolor=c[i]),
                    medianprops=dict(color=c[i]),
                    )
        sp += .60

    plt.xlim(0.5, 8)
    plt.xticks([1.6, 4, 6.6], ['clark_a', 'library', 'mechanical'])

    patches = []
    for i, dev in enumerate(test_devices):
        patch = mpatches.Patch(color=c[i], label=dev)
        patches.append(patch)

    plt.legend(handles=patches)

    plt.ylabel("error(m)")

    plt.title("RBF: Train: "+str(train_devices[0]))

    plt.savefig("test.PNG")
    plt.show()

