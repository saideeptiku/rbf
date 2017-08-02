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

print_df(train_df)

#
# def run_RBF(train_df, test_df, input_cols, pos_label)

# create a model for ranked offline data
rbf_model = RBF(train_df, input_cols, pos_label, ['x', 'y'])

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

im = plt.imread("data/bcinfill_map.jpg")
implot = plt.imshow(im, origin='upper', extent=[-15, 860, 250, 630])

# get position at each training position
for index in [x for x in list(test_df.index) if x % 1 == 0]:
    # get the sorted vector and actual position from test data frame
    sorted_vector, actual_pos_label = RBF.get_sorted_vector_at_index(index,
                                                                     test_df,
                                                                     input_cols,
                                                                     pos_label)
    # actual x and y position
    real_x, real_y = rbf_model.get_xy_from_position_label(actual_pos_label, test_df)
    real_pos_plt = ax1.scatter(real_x, real_y, c='b', marker='o', alpha=0.6, label="real position")

    # projected position
    pos_x, pos_y = rbf_model.get_projected_position(sorted_vector,
                                                    k=1,
                                                    similarity_measure='spearmans_footrule')

    proj_pos_plt = ax1.scatter(pos_x, pos_y, c='r', marker='*', alpha=0.6, label="projected position")

    print(index, sep="", end=", ", flush=True)

plt.legend(handles=[real_pos_plt, proj_pos_plt], bbox_to_anchor=(0.63, 0.97))
plt.show()
