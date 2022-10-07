# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: blacklancer
## Modified from: https://github.com/hshustc/CVPR19_Incremental_Learning
## Copyright (c) 2022
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
import torch.nn as nn
import torch.nn.functional as F

# 记录传递过程中进入余弦分类器的特征
cur_features_c = []
ref_features_c = []
def get_cur_features_c(self, inputs, outputs):
    global cur_features_c
    cur_features_c = inputs[0]

def get_ref_features_c(self, inputs, outputs):
    global ref_features_c
    ref_features_c = inputs[0]


def incremental_train_and_eval(args, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, trainloader, testloader,
                      iteration, start_iteration, lamda, fix_bn=False, weight_per_class=None, device=None):

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    handle_cur_features_c = tg_model.fc.register_forward_hook(get_cur_features_c)
    if iteration > start_iteration:
        ref_model.eval()
        handle_ref_features_c = ref_model.fc.register_forward_hook(get_ref_features_c)

    for epoch in range(args.epochs):
        #train
        tg_model.train()
        if fix_bn:
            for m in tg_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        train_loss, train_loss1, train_loss2 = 0, 0, 0
        correct = 0
        total = 0
        tg_lr_scheduler.step()
        print('\nEpoch: %d, LR: ' % epoch, end='')
        print(tg_lr_scheduler.get_lr())

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            tg_optimizer.zero_grad()
            outputs = tg_model(inputs)

            if iteration == start_iteration:
                loss1 = 0
                loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            else:
                ref_output = ref_model(inputs)
                loss1 = nn.CosineEmbeddingLoss()(cur_features_c, ref_features_c.detach(),
                                                 torch.ones(inputs.shape[0]).to(device)) * lamda
                loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            loss = loss1 + loss2
            loss.backward()
            tg_optimizer.step()

            train_loss += loss.item()
            if iteration > start_iteration:
                train_loss1 += loss1.item()
            else:
                train_loss1 += loss1
            train_loss2 += loss2.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('Train set: {}, Train Loss1: {:.4f}, Train Loss2: {:.4f}Train Loss: {:.4f} Acc: {:.4f}'.format(
            len(trainloader), train_loss1/(batch_idx+1), train_loss2/(batch_idx+1),
            train_loss/(batch_idx+1), 100.*correct/total))

        #eval
        tg_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = tg_model(inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print('Test set: {} Test Loss: {:.4f} Acc: {:.4f}'.format(\
            len(testloader), test_loss/(batch_idx+1), 100.*correct/total))

    print("Removing register_forward_hook")
    handle_cur_features_c.remove()
    if iteration > start_iteration:
        handle_ref_features_c.remove()

    return tg_model
