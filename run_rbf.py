# import modules
import pandas as pd
from label_x_y_locations import label_similar_locations
from RBF import RBF
from util_functions import print_df

# read data training
training_df = pd.read_csv("data/bcinfill-s6-run1.csv")

# read validation training
validation_df = pd.read_csv("data/bcinfill-s6-run2.csv")

# data needs to preprocessed for RBF algorithm
# For RBF apply labels to locations
train_df, test_df = label_similar_locations(training_df,
                                            validation_df)

# print_df(train_df)
# exit()

# make input feature columns list
input_cols = train_df.columns[1:-4]
# make output feature column list
pos_label = train_df.columns[-1]

# print("number of columns total:", len(train_df.columns))
#
# print("input cols:", input_cols)
# print("input col length:", len(input_cols))
#
# print("output cols:", pos_label)
# print("output col length:", len(pos_label))

# create a model for ranked offline data
rbf_model = RBF(train_df, input_cols, pos_label, ['x', 'y'])

# get position at each training position
for index in list(test_df.index):

    # get the sorted vector and actual position from test data frame
    sorted_vector, actual_pos_label = RBF.get_sorted_vector_at_index(index,
                                                                     test_df,
                                                                     input_cols,
                                                                     pos_label)

    projected_position = rbf_model.get_projected_position(sorted_vector,
                                                          k=5,
                                                          similarity_measure='spearman')

    print(index, sep="", end=", ", flush=True)




