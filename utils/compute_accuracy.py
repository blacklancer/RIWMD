##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Modified from: https://github.com/hshustc/CVPR19_Incremental_Learning
## Max Planck Institute for Informatics
## yaoyao.liu@mpi-inf.mpg.de
## Copyright (c) 2021
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" The functions that compute the accuracies """

import torch.nn.functional as F
from scipy.spatial.distance import cdist
from utils.misc import *


def map_labels(order_list, Y_set):
    map_Y = []
    for idx in Y_set:
        map_Y.append(order_list.index(idx))
    map_Y = np.array(map_Y)
    return map_Y


def compute_accuracy(tg_model, tg_feature_model, class_means, evalloader, scale=None, print_info=True, device=None):

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 传递进来两个模型
    tg_model.eval()
    tg_feature_model.eval()

    correct_cnn, correct_proto, correct_ncm = 0, 0, 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # total统计现在为止的数据量，一个总数
            total += targets.size(0)

            # compute score for cnn
            outputs = tg_model(inputs)
            outputs = F.softmax(outputs, dim=1)
            if scale is not None:
                assert (scale.shape[0] == 1)
                assert (outputs.shape[1] == scale.shape[1])
                outputs = outputs / scale.repeat(outputs.shape[0], 1).type(torch.FloatTensor).to(device)
            _, predicted = outputs.max(1)
            correct_cnn += predicted.eq(targets).sum().item()

            # 下面两个计算使用的特征都是一样的，重复使用就好
            outputs_feature = (np.squeeze(tg_feature_model(inputs))).cpu().numpy()
            # Compute score for proto
            sqd_proto = cdist(class_means[:, :, 0].T, outputs_feature, 'sqeuclidean')
            score_proto = torch.from_numpy((-sqd_proto).T).to(device)
            _, predicted_proto = score_proto.max(1)
            correct_proto += predicted_proto.eq(targets).sum().item()

            # Compute score for NCM
            sqd_ncm = cdist(class_means[:, :, 1].T, outputs_feature, 'sqeuclidean')
            score_ncm = torch.from_numpy((-sqd_ncm).T).to(device)
            _, predicted_ncm = score_ncm.max(1)
            correct_ncm += predicted_ncm.eq(targets).sum().item()

    if print_info:
        print("  top 1 accuracy FC             :\t\t{:.2f} %".format(100. * correct_cnn / total))
        print("  top 1 accuracy Proto          :\t\t{:.2f} %".format(100. * correct_proto / total))
        # print("  top 1 accuracy NCM            :\t\t{:.2f} %".format(100. * correct_ncm / total))

    cnn_acc = 100. * correct_cnn / total
    proto_acc = 100. * correct_proto / total
    ncm_acc = 100. * correct_ncm / total

    return [cnn_acc, proto_acc, ncm_acc]
