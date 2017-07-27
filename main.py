# import modules
import pandas as pd
from label_x_y_locations import label_similar_locations
from RBF import RBF
from util_functions import print_df
import matplotlib.pyplot as plt


# def main():
# read data training
training_df = pd.read_csv("data/bcinfill-s6-run1.csv")

# read validation training
validation_df = pd.read_csv("data/bcinfill-s6-run2.csv")

# data needs to preprocessed for RBF algorithm
# For RBF apply labels to locations
train_df, test_df = label_similar_locations(training_df,
                                            validation_df)

# make input feature columns list
input_cols = train_df.columns[1:-4]
# make output feature column list
pos_label = train_df.columns[-1]


#
# def run_RBF(train_df, test_df, input_cols, pos_label)

# create a model for ranked offline data
rbf_model = RBF(train_df, input_cols, pos_label, ['x', 'y'])

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

# get position at each training position
for index in list(test_df.index):
    # get the sorted vector and actual position from test data frame
    sorted_vector, actual_pos_label = RBF.get_sorted_vector_at_index(index,
                                                                     test_df,
                                                                     input_cols,
                                                                     pos_label)
    # actual x and y position
    real_x, real_y = rbf_model.get_xy_from_position_label(actual_pos_label, test_df)
    ax1.scatter(real_x, real_y, c='b', marker='o', alpha=0.4)

    # projected position
    pos_x, pos_y = rbf_model.get_projected_position(sorted_vector,
                                                    k=2,
                                                    similarity_measure='spearman')

    ax1.scatter(pos_x, pos_y, c='r', marker='*', alpha=0.4)

    print(index, sep="", end=", ", flush=True)

plt.show()
