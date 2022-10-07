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

# 记录传递过程中layer1的feature
cur_features_1 = []
ref_features_1 = []
def get_ref_features_1(self, inputs, outputs):
    global ref_features_1
    ref_features_1 = outputs

def get_cur_features_1(self, inputs, outputs):
    global cur_features_1
    cur_features_1 = outputs

# 记录传递过程中layer2的feature
cur_features_2 = []
ref_features_2 = []
def get_ref_features_2(self, inputs, outputs):
    global ref_features_2
    ref_features_2 = outputs

def get_cur_features_2(self, inputs, outputs):
    global cur_features_2
    cur_features_2 = outputs

# 记录传递过程中layer3的feature
cur_features_3 = []
ref_features_3 = []
def get_ref_features_3(self, inputs, outputs):
    global ref_features_3
    # ref_features_3 = inputs[0]
    ref_features_3 = outputs

def get_cur_features_3(self, inputs, outputs):
    global cur_features_3
    # cur_features_3 = inputs[0]
    cur_features_3 = outputs

# 记录传递过程中layer3的feature
cur_features_4 = []
ref_features_4 = []
def get_ref_features_4(self, inputs, outputs):
    global ref_features_4
    ref_features_4 = outputs

def get_cur_features_4(self, inputs, outputs):
    global cur_features_4
    cur_features_4 = outputs


def MD(list_attentions_a, list_attentions_b, device=None, normalize=True, memory_flags=None, only_old=False):

    assert len(list_attentions_a) == len(list_attentions_b)
    # print(len(list_attentions_a))

    a = list_attentions_a
    b = list_attentions_b
    # shape of (b, n, w, h)
    assert a.shape == b.shape, (a.shape, b.shape)

    if only_old:
        a = a[memory_flags]
        b = b[memory_flags]

    a = torch.pow(a, 2)
    b = torch.pow(b, 2)

    a_h = a.sum(dim=3).view(a.shape[0], -1)  # shape of (b, c * w)
    b_h = b.sum(dim=3).view(b.shape[0], -1)
    a_w = a.sum(dim=2).view(a.shape[0], -1)  # shape of (b, c * h)
    b_w = b.sum(dim=2).view(b.shape[0], -1)
    a = torch.cat([a_h, a_w], dim=-1)
    b = torch.cat([b_h, b_w], dim=-1)

    if normalize:
        a = F.normalize(a, dim=1, p=2)
        b = F.normalize(b, dim=1, p=2)
    a = a.to(device)
    b = b.to(device)
    loss = nn.CosineEmbeddingLoss()(a, b.detach(), torch.ones(a.shape[0]).to(device))
    # loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
    return loss


def incremental_train_and_eval_MD(args, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, trainloader, testloader,
                      iteration, start_iteration, lamda, fix_bn=False, weight_per_class=None, device=None):

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    handle_cur_features_1 = tg_model.layer1.register_forward_hook(get_cur_features_1)
    handle_cur_features_2 = tg_model.layer2.register_forward_hook(get_cur_features_2)
    handle_cur_features_3 = tg_model.layer3.register_forward_hook(get_cur_features_3)
    handle_cur_features_4 = tg_model.layer4.register_forward_hook(get_cur_features_4)
    if iteration > start_iteration:
        ref_model.eval()
        handle_ref_features_1 = ref_model.layer1.register_forward_hook(get_ref_features_1)
        handle_ref_features_2 = ref_model.layer2.register_forward_hook(get_ref_features_2)
        handle_ref_features_3 = ref_model.layer3.register_forward_hook(get_ref_features_3)
        handle_ref_features_4 = ref_model.layer4.register_forward_hook(get_ref_features_4)

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
        print("batch_num:" + str(len(trainloader)))

        for batch_idx, (inputs, targets) in enumerate(trainloader):

            inputs, targets = inputs.to(device), targets.to(device)
            targets = torch.tensor(targets, dtype=torch.long)

            tg_optimizer.zero_grad()
            outputs = tg_model(inputs)

            if iteration == start_iteration:
                loss1 = 0
                loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            else:
                ref_output = ref_model(inputs)
                loss11 = MD(cur_features_1, ref_features_1, device)
                loss12 = MD(cur_features_2, ref_features_2, device)
                loss13 = MD(cur_features_3, ref_features_3, device)
                loss14 = MD(cur_features_4, ref_features_4, device)
                loss1 = (1 * loss11 + 1 * loss12 + 1 * loss13 + 1 * loss14) * lamda
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
        print('Train set: {}, Train Loss1: {:.4f}, Train Loss2: {:.4f}, Train Loss: {:.4f} Acc: {:.4f}'.format(
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
                targets = torch.tensor(targets, dtype=torch.long)
                outputs = tg_model(inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print('Test set: {} Test Loss: {:.4f} Acc: {:.4f}'.format(\
            len(testloader), test_loss/(batch_idx+1), 100.*correct/total))

    print("Removing register_forward_hook")
    handle_cur_features_1.remove()
    handle_cur_features_2.remove()
    handle_cur_features_3.remove()
    handle_cur_features_4.remove()
    if iteration > start_iteration:
        handle_ref_features_1.remove()
        handle_ref_features_2.remove()
        handle_ref_features_3.remove()
        handle_ref_features_4.remove()

    return tg_model
