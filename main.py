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

# ######### Constants ##########

px_to_m = {
    'bc_infill': 20,
    'clark_A': 18,
    'lib': 20,
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
           samples_per_rp=5, label_block_size=5, k=3):
    """
    Run RBF
    """
    # read data training
    train_paths = get_paths(place, train_dev, train_run, meta=True)
    train_meta_dict = read_meta(train_paths[1])

    training_df = read_csv(train_paths[0], ["WAP*"], ['x', 'y'], replace_na=-100,
                           force_samples_per_rp=samples_per_rp, rename_cols=train_meta_dict)

    # read data testing
    test_paths = get_paths(place, test_dev, test_run, meta=True)
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
    input_cols = list(set(train_df.columns) -
                      set(['x', 'y', 'position_label']))
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


def do_RBF_all():
    """
    run RBF on all devices,
    :returns: dict{place}{dev_train}{dev_test} => list
    """
    errors = defaultdict(lambda: defaultdict(dict))

    # ignore lg and bc_infill
    for p in Place.list_all[1:3]:
        for dev_train in Device.list_all[1:3]:
            for dev_test in Device.list_all[1:3]:

                if dev_train is dev_test:
                    continue

                real, guess = do_RBF(p, dev_train, 0, dev_test, 0,
                                     label_block_size=LABEL_BLOCK_SIZE[p])

                # convert each distance in meters
                error_in_m = [e / px_to_m[p] for e in util_f.euclideans(real, guess)]

                errors[p][dev_train][dev_test] = error_in_m

                return errors


if __name__ == "__main__":

    e = do_RBF_all()
    grouped_places_boxplot_devices(e)
    # print("\n\n")
    # print(errors)
