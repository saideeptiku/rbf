# import modules
import pandas as pd
from label_x_y_locations import label_similar_locations
from util_functions import print_df

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
output_cols = train_df.columns[-1]

print("number of columns total:", len(train_df.columns))

print("input cols:", input_cols)
print("input col length:", len(input_cols))

print("output cols:", output_cols)
print("output col length:", len(output_cols))

