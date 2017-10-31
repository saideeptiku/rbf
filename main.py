# import modules
import pandas as pd
from label_x_y_locations import label_similar_locations
from RBF import RBF
import util_functions as util_f
import matplotlib.pyplot as plt
from CSUDB.data_fetcher import Device, Place, get_paths, read_meta, read_csv
from label_block_size import LABEL_BLOCK_SIZE
from collections import defaultdict
from plotter import grouped_places_boxplot_devices
from threading import Thread
import csv

# ######### Constants ##########


px_to_m = {
    'bc_infill': 20,
    'clark_A': 18,
    'lib': 20,
    'lib_2m': 20,
    'mech_f1': 17
}


PX_TO_M = 20

K_MIN = 1
K_MAX = 10

# KNN K value
K = 2

SAMPLE_INTERVAL = 5

# title stuff
TRAIN_DEVICE = "O2"
TEST_DEVICE = "LG"
PATH_NAME = "BC-INFILL"


###############################

# TODO: make this plot on map function.
def plot_map():
    # read data training
    # training_df = pd.read_csv("data/CSUDB/CSV/oneplus3/bc_infill/bc_infill_run0.csv")
    training_df = pd.read_csv(
        "CSUDB/CSV/oneplus2/bc_infill/bc_infill_run0.csv")
    # read validation training
    validation_df = pd.read_csv("CSUDB/CSV/LG/bc_infill/bc_infill_run1.csv")

    # data needs to preprocessed for RBF algorithm
    # For RBF apply labels to locations
    train_df, test_df = label_similar_locations(training_df,
                                                validation_df,
                                                label_block_size=LABEL_BLOCK_SIZE)

    # filter data pick every fourth element
    # print(training_df.shape)
    train_df = util_f.pick_label_num_by_multiple(
        train_df, SAMPLE_INTERVAL, "position_label")
    test_df = util_f.pick_label_num_by_multiple(
        test_df, SAMPLE_INTERVAL, "position_label")
    # print(training_df.shape)

    # fix column names
    train_df = util_f.remove_wap_in_column_name(train_df)
    test_df = util_f.remove_wap_in_column_name(test_df)

    # set index to begin with 1
    train_df = util_f.set_df_index_begin(train_df)
    test_df = util_f.set_df_index_begin(test_df)

    # make input feature columns list
    input_cols = train_df.columns[1:-4]
    # make output feature column list
    pos_label = train_df.columns[-1]

    # drop cols in test that are not present in train
    # print_df(train_df)

    # create a model for ranked offline data
    rbf_model = RBF(train_df, input_cols, pos_label, ['x', 'y'])

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    im = plt.imread("CSUDB/Maps/bc_infill.PNG")
    implot = plt.imshow(im, origin='upper', extent=[-15, 860, 250, 630])

    # data collection lists
    actual_pos = []
    found_pos = []

    # get position at each training position
    for index in [x for x in list(test_df.index) if x % 1 == 0]:
        print(index, sep="", end=", ", flush=True)

        # get the sorted vector and actual position from test data frame
        sorted_vector, actual_pos_label = RBF.get_sorted_vector_at_index(index,
                                                                         test_df,
                                                                         input_cols,
                                                                         pos_label)

        # actual x and y position
        try:
            real_x, real_y = rbf_model.get_xy_from_position_label(
                actual_pos_label, test_df)
        except:
            # print("label", actual_pos_label, test_df)
            continue

        actual_pos.append((real_x, real_y))
        real_pos_plt = ax1.scatter(
            real_x, real_y, c='b', marker='o', alpha=0.6, label="real position")

        # projected position
        pos_x, pos_y = rbf_model.get_projected_position(sorted_vector,
                                                        k=K,
                                                        similarity_measure='spearmans_footrule')
        found_pos.append((pos_x, pos_y))

        proj_pos_plt = ax1.scatter(
            pos_x, pos_y, c='r', marker='*', alpha=0.6, label="projected position")

    # , bbox_to_anchor=(0.2, 0.97))
    plt.legend(handles=[real_pos_plt, proj_pos_plt])

    # make title
    avg_err = str("%.2f" % util_f.convert_unit(
        util_f.average_euclidean(actual_pos, found_pos), PX_TO_M)) + "m"

    print("{:3} {:5}".format(K, avg_err))

    title = TRAIN_DEVICE + " + " + TEST_DEVICE + \
        " + K: " + str(K) + "\n" + "AVG ERROR: " + avg_err

    plt.title(title)

    plt.savefig("plots/" + "/" + title.replace("\n",
                                               "_").replace(" ", "_") + ".tiff")


def do_RBF(place, train_dev, train_run, test_dev, test_run,
           samples_per_rp=1, label_block_size=5, k=3):
    """
    Run RBF
    """
    # read data training
    train_paths = get_paths(place, train_dev, train_run, meta=True)
    test_paths = get_paths(place, test_dev, test_run, meta=True)    

    if  not test_paths or not train_paths:
        print("no data for", place, train_dev)
        return [], []

    train_meta_dict = read_meta(train_paths[1])


    training_df = read_csv(train_paths[0], ["WAP*"], ['x', 'y'], replace_na=-100,
                           force_samples_per_rp=samples_per_rp, rename_cols=train_meta_dict)

    # read data testing

    test_meta_dict = read_meta(test_paths[1])

    validation_df = read_csv(test_paths[0], ["WAP*"], ['x', 'y'], replace_na=-100,
                             force_samples_per_rp=samples_per_rp, rename_cols=test_meta_dict)

    # reverse mac_dict
    mac_dict = dict((v, k) for k, v in train_meta_dict.items())

    # keep common columns
    required_cols = list(set(training_df.columns).intersection(
        set(validation_df.columns)))
    training_df = training_df[required_cols]
    validation_df = validation_df[required_cols]

    # rename columns to WAPS
    training_df = training_df.rename(columns=mac_dict)
    validation_df = validation_df.rename(columns=mac_dict)

    # data needs to preprocessed for RBF algorithm
    # For RBF apply labels to locations
    train_df, test_df = label_similar_locations(training_df,
                                                validation_df,
                                                label_block_size=label_block_size)

    # fix column names
    train_df = util_f.remove_wap_in_column_name(train_df)
    test_df = util_f.remove_wap_in_column_name(test_df)

    # set index to begin with 1
    train_df = util_f.set_df_index_begin(train_df)
    test_df = util_f.set_df_index_begin(test_df)

    # make input feature columns list
    input_cols = list(set(train_df.columns) - set(['x', 'y', 'position_label']))
    # make output feature column list
    pos_label = 'position_label'

    # create a model for ranked offline data
    rbf_model = RBF(train_df, input_cols, pos_label, ['x', 'y'])

    # data collection lists
    actual_pos = []
    found_pos = []

    # get position at each training position
    for index in list(test_df.index):
        print(index, sep="", end=", ", flush=True)

        # get the sorted vector and actual position from test data frame
        sorted_vector, actual_pos_label = RBF.get_sorted_vector_at_index(index,
                                                                         test_df,
                                                                         input_cols,
                                                                         pos_label)

        # actual x and y position
        try:
            real_x, real_y = rbf_model.get_xy_from_position_label(
                actual_pos_label, test_df)
        except:
            # print("label", actual_pos_label, test_df)
            continue

        actual_pos.append((real_x, real_y))

        # projected position
        pos_x, pos_y = rbf_model.get_projected_position(sorted_vector,
                                                        k=k,
                                                        similarity_measure='spearmans_footrule')
        found_pos.append((pos_x, pos_y))

    return actual_pos, found_pos


def do_RBF_all(place_list=None, compare_self=True):
    """
    run RBF on all devices,
    :returns: dict{place}{dev_train}{dev_test} => list
    """
    errors = defaultdict(lambda: defaultdict(dict))

    if not place_list:
        print("place list not given!")
        place_list = Place.list_all
    else:
        print(place_list)

    for p in place_list:
        for dev_train in Device.list_all:
            for dev_test in Device.list_all:

                if dev_train == dev_test and not compare_self:
                    continue

                print("\n", p, dev_train, dev_test)

                real, guess = do_RBF(p, dev_train, 0, dev_test, 1,
                                     label_block_size=LABEL_BLOCK_SIZE[p])

                # convert each distance in meters
                error_in_m = [e / px_to_m[p] for e in util_f.euclideans(real, guess)]

                if error_in_m:
                    print("\n\n", p, dev_train, dev_test, "==>", sum(error_in_m) / len(error_in_m))

                errors[p][dev_train][dev_test] = error_in_m

    return errors


def write_to_file(filename, ddd_dict):
    cols = 'place, device_train, device_test'
    for i in range(100):
        cols += ', err' + str(i)

    f = open(filename, 'w')
    f.write(cols + "\n")
    for p in Place.list_all:
        for d1 in Device.list_all:
            for d2 in Device.list_all:

                # if d1 == d2:
                #     continue

                f.write(p + ", " + d1 + ", " + d2 + ", ")
                try:
                    str_dat = [str(x) for x in ddd_dict[p][d1][d2]]
                except:
                    print(p, d1, d2)
                    str_dat = []

                f.write(", ".join(str_dat) + "\n")
    f.close()


def read_from_file(filename):
    errors = defaultdict(lambda: defaultdict(dict))

    lines = []
    with open(filename) as f:
        lines = f.readlines()

    for line in lines[1:]:

        line_list = line.replace(" ", "").strip("\n").split(",")
        p = line_list[0]
        d1 = line_list[1]
        d2 = line_list[2]

        data_list = list(line_list[3:])

        if data_list:
            data = [float(x) for x in data_list[3:]]
        else:
            data = []
            print()

        errors[p][d1][d2] = data

    return errors

if __name__ == "__main__":

    # global errors
    # do_RBF_all()
    # num_threads = 2

    # t = Thread(target=do_RBF_all, args=[[Place.list_all[1]]] )
    # t.start()
    #
    # t1 = Thread(target=do_RBF_all, args=[[Place.list_all[2]]] )
    # t1.start()
    #
    # t3 = Thread(target=do_RBF_all, args=[[Place.list_all[3]]] )
    # t3.start()
    #
    # t.join()
    # t1.join()
    # t3.join()
    # do_RBF(Place.mech_f1, Device.oneplus3, 0, Device.samsung_s6, 1)
    # er = do_RBF_all(place_list=[Place.bc_infill, Place.mech_f1, Place.clark_a])
    
    # write_to_file('results/results.csv', er)


    # # write_to_file('results.csv', errors)
    # errors = read_from_file('results.csv')



    # #
    # # place_groups = list(errors.keys())
    # # # print(place_groups)
    # #
    # # train_devices = list(errors[place_groups[0]].keys())
    # # #
    # # test_devices = list(errors[place_groups[0]][train_devices[0]].keys())
    # # print(place_groups, train_devices, test_devices)
    # grouped_places_boxplot_devices(errors)

    # for dev in Device.list_all[1:]:
    #     grouped_places_boxplot_devices(read_from_file('results/results.csv'), train_device=dev)
    # print("\n\n")
    # print(errors)
    grouped_places_boxplot_devices(read_from_file('results/results.csv'),
                                   train_device=Device.oneplus2,
                                   test_devices=[Device.samsung_s6, Device.lg, Device.oneplus2],
                                   places=[Place.bc_infill, Place.clark_a, Place.mech_f1])

