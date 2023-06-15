from ts2vec import TS2Vec
import datautils
import sys
sys.path.append("/home/zhangjunru/1104")
from code_ours.TSC_data_loader import TSC_multivariate_data_loader
import numpy as np
# Load the ECG200 dataset from UCR archive
#train_data, train_labels, test_data, test_labels = datautils.load_UEA('UWaveGestureLibrary')

# (Both train_data and test_data have a shape of n_instances x n_timestamps x n_features)
path = "/home/zhangjunru/0919UCR/datasets2/datasets/Multivariate2018_ts/Multivariate_ts/"
fname = "LSST"
train_data, train_labels, test_data, test_labels = TSC_multivariate_data_loader(path, fname) #instance * len * channels
train_data = np.array(np.transpose(train_data, (0, 2, 1)))
test_data = np.array(np.transpose(test_data, (0, 2, 1)))
input_dims = train_data.shape[2]
# Train a TS2Vec model
model = TS2Vec(
    input_dims=input_dims,
    device=0,
    output_dims=320
)
loss_log = model.fit(
    train_data,
    verbose=True
)

# Compute timestamp-level representations for test set
test_repr = model.encode(test_data)  # n_instances x n_timestamps x output_dims

# Compute instance-level representations for test set
test_repr = model.encode(test_data, encoding_window='full_series')  # n_instances x output_dims

# Sliding inference for test set
test_repr = model.encode(
    test_data,
    casual=True,
    sliding_length=1,
    sliding_padding=50
)  # n_instances x n_timestamps x output_dims
# (The timestamp t's representation vector is computed using the observations located in [t-50, t])