"""
plotting functions for this project
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from time import sleep


def grouped_places_boxplot_devices(data_dict, train_device=None,
                                   test_devices=[], places=[],
                                   outfile=None, outdir="plots/NSF/"):
    """
    dict{place}{dev_train}{dev_test} => list
    """
    if len(places) == 0:
        place_groups = sorted(list(data_dict.keys()))
    else:
        place_groups = places

    if not train_device:
        train_device = list(data_dict[place_groups[0]].keys())[0]

    if not outfile:
        outfile = outdir + train_device + ".PNG"

    if not test_devices:
        test_devices = list(data_dict[place_groups[0]][train_device].keys())

    # print(train_device)

    c = ['red', 'green', 'blue', 'yellow', 'purple']

    # make data sets
    A = [
        data_dict[place_groups[0]][train_device][test_devices[0]],
        data_dict[place_groups[1]][train_device][test_devices[0]],
        data_dict[place_groups[2]][train_device][test_devices[0]]
    ]

    B = [
        data_dict[place_groups[0]][train_device][test_devices[1]],
        data_dict[place_groups[1]][train_device][test_devices[1]],
        data_dict[place_groups[2]][train_device][test_devices[1]]
    ]

    C = [
        data_dict[place_groups[0]][train_device][test_devices[2]],
        data_dict[place_groups[1]][train_device][test_devices[2]],
        data_dict[place_groups[2]][train_device][test_devices[2]]
    ]

    # print(A, B, C)

    data = [A, B, C]
    sp = 0
    for i, d in enumerate(data):
        plt.boxplot(d, positions=[1 + sp, 3.5 + sp, 6 + sp], notch=True, patch_artist=True,
                    boxprops=dict(facecolor=c[i], color=c[i]),
                    capprops=dict(color=c[i]),
                    whiskerprops=dict(color=c[i]),
                    flierprops=dict(color=c[i], markeredgecolor=c[i]),
                    medianprops=dict(color=c[i]),
                    )
        sp += .60

    plt.xlim(0.5, 8)
    plt.xticks([1.6, 4.1, 6.6], place_groups)

    patches = []
    for i, dev in enumerate(test_devices):

        device = dev
        if dev is train_device:
            device += " (Train)"
        else:
            device += " (Test)"

        patch = mpatches.Patch(color=c[i], label=device)
        patches.append(patch)

    plt.legend(handles=patches, loc=1, framealpha=0.5)

    plt.ylabel("error(m)")

    plt.title("Device Heterogenity (RBF) " + str(train_device))

    # plt.savefig(outfile)
    plt.savefig(outdir+train_device+".PNG")

    # plt.clf()
    # plt.close()
    plt.show()
