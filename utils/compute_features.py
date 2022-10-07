# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: blacklancer
## Modified from: https://github.com/hshustc/CVPR19_Incremental_Learning
## Copyright (c) 2022
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" The functions that compute the features """
from utils.misc import *


def compute_features(tg_feature_model, evalloader, num_samples, num_features, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    tg_feature_model.eval()

    features = np.zeros([num_samples, num_features])
    start_idx = 0
    with torch.no_grad():
        for inputs, targets in evalloader:
            inputs = inputs.to(device)
            # .cpu().numpy()
            features[start_idx:start_idx + inputs.shape[0], :] = np.squeeze(tg_feature_model(inputs).cpu().numpy())
            start_idx = start_idx + inputs.shape[0]
    assert (start_idx == num_samples)
    return features
