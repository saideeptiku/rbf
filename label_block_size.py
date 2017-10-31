"""
functions for analysis of the block size for a label.
Use dict for values.
"""
# import modules
import pandas as pd
from label_x_y_locations import label_similar_locations
from RBF import RBF
import util_functions as util_f
import matplotlib.pyplot as plt
from CSUDB.data_fetcher import Device, Place, get_paths, read_meta, read_csv

######## RESULTS OF ANALYSIS ##########

# size of block in px
LABEL_BLOCK_SIZE = {
    'bc_infill': 5,
    'clark_A': 6,
    'lib': 15,
    'lib_2m': 15,
    'mech_f1': 12
}

######################################



def block_size_exp(place, train_dev, train_run, test_dev, test_run,
                   samples_per_rp=5, label_block_size=5, verbose=False):
    """
    Returns number of rows that were not labeled.
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
    _, _, fail_count = label_similar_locations(training_df,
                                               validation_df,
                                               label_block_size=label_block_size,
                                               get_count=True, verbose=verbose)

    return fail_count


def test_block_size():
    """
    test the block sizes for different places.
    """
    fails = []
    for p in Place.list_all[1:]:
        for device1 in Device.list_all[1:]:
            for device2 in Device.list_all[1:]:
                try:
                    f = block_size_exp(p, device1, 0,
                                       device2, 0,
                                       label_block_size=LABEL_BLOCK_SIZE[p], verbose=False)

                    print(p, device1, device2, "SUCCESS - ",
                          f, end=' -\n', sep=' ', flush=True)

                except:
                    fails.append((p, device1, device2))
                    print(p, device1, device2, "FAIL",
                          end='\n', sep=' ', flush=True)
                    continue
    if fails:
        print("\nFAILS: \n", fails)
    else:
        print("\nSUCCESS.")


def block_exp_all(min_fails=3):
    """
    try different block sizes for each device pair and place. prints dict.
    """

    def all_elements_are(lst, elem):
        t = []
        for _ in range(len(lst)):
            t.append(elem)

        return t == lst

    def try_update_dict(dict_bs, key, some_bs, context=None):
        context = None
        if context:
            print("-", context, '-', dict_bs[key], some_bs, end='')

        if dict_bs[key] < some_bs:
            dict_bs[key] = some_bs
            return True
        else:
            print("*", end='')
            return False

    bs_dict = {}
    for p in Place.list_all[1:]:
        bs_dict[p] = 1

        print(p, end='\n')

        for device1 in Device.list_all:
            for device2 in Device.list_all:

                # same device skip
                if device1 == device2:
                    continue

                print("\t\t", device1, "+", device2,
                      end=':  ', flush=True, sep='')

                prev_list = [99999, ]
                for bs in range(1, 30):

                    try:
                        fails = block_size_exp(p, device1, 0,
                                               device2, 0,
                                               label_block_size=bs, verbose=False)

                    except:
                        print("^", end='', sep='')
                        continue

                    # if fails is zero, stop now
                    if fails == 0 and try_update_dict(bs_dict, p, bs, context="fails are zero"):
                        print("fails are zero", bs_dict[p])
                        break

                    # if fails is zero but cannot update dict, just break
                    elif fails == 0 and not try_update_dict(bs_dict, p, bs, context="F0NU"):
                        print("fails are zero, no update", bs_dict[p])
                        break

                    # if fails have reduced or remained same, make an entry and write a note
                    elif (prev_list[-1] > fails or prev_list[-1] == fails) and fails != 0:

                        # print(" [", prev_list[-1], "-->", fails,
                        #       ",", bs, end='] ', flush=True, sep="")

                        print(fails, "-->", flush=True, end=' ')

                        # store it in a list
                        prev_list.append(fails)

                    # if it worsens. pick previous fail
                    elif prev_list[-1] < fails and try_update_dict(bs_dict, p, bs - 1, context='worse') and fails != 0:
                        print("worse", bs - 1)
                        break

                    # uncaptured case
                    else:
                        print(prev_list[-1], fails, bs_dict[p], bs)
                        exit("uncaptured case")

                    # check last N elements were same, and there are atleast N elements in the list
                    if len(prev_list) >= min_fails and all_elements_are(prev_list[-min_fails:], fails):

                        # if is greater than bs for any previously chosen device
                        if try_update_dict(bs_dict, p, bs - min_fails + 1, context='rep'):
                            print("repeated elements", bs_dict[p])
                            break
                        else:
                            print("repeated elements, No updates", bs_dict[p])
                            break

        print("...done.", p, "=>", bs_dict[p])

    print("\n\n BLOCK_SIZE_DICT=", bs_dict)


if __name__ == "__main__":

    # print(get_map(PLACES[0]))
    # print(get_paths(PLACES[0], DEVICES[0], 0))
    # exit()
    # global K
    # # print("RBF ", TRAIN_DEVICE + " + " + TEST_DEVICE)
    # # print("{:3} {:5}".format("K", "AVG. ERROR"))
    # for i in range(K_MIN, K_MAX):
    #     K = i
    #     main()
    # print(block_size_exp(Place.clark_a, Device.oneplus3, 0, Device.oneplus2, 0,
    #                      label_block_size=6, verbose=False))

    # block_exp_all(min_fails=6)
    test_block_size()

    # p = [135, 15, 16, 17]

    # print(p[-3:])
