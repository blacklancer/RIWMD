#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: blacklancer
## Modified from: https://github.com/yaoyao-liu/class-incremental-learning
## Copyright (c) 2022
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Class-incremental learning trainers. """
import torch
from tensorboardX import SummaryWriter
import numpy as np
import os
import os.path as osp
from trainers.base_trainer import BaseTrainer
from trainers.incremental_train_and_eval import incremental_train_and_eval
from trainers.incremental_train_and_eval_MD import incremental_train_and_eval_MD
import warnings
warnings.filterwarnings('ignore')

try:
    import cPickle as pickle
except:
    import pickle



class Trainer(BaseTrainer):
    def train(self):
        """The class that contains the code for the class-incremental system.
        This trianer is based on the base_trainer.py in the same folder.
        If you hope to find the source code of the functions used in this trainers,
        you may find them in base_trainer.py.
        """

        # Set tensorboard recorder
        # self.train_writer = SummaryWriter(comment=self.save_path)

        # Initial the array to store the accuracies for each phase
        # 中间的3分别是FC，NEM，NCM的准确度
        top1_acc_list_cumul = np.zeros((int(self.args.num_classes / self.args.nb_cl), 3, self.args.nb_runs))
        top1_acc_list_ori = np.zeros((int(self.args.num_classes / self.args.nb_cl), 3, self.args.nb_runs))

        # Load the training and test samples from the dataset
        X_train_total, Y_train_total, X_valid_total, Y_valid_total = self.set_dataset()
        # print(X_train_total.shape)
        # print(Y_train_total.shape)
        # print(X_valid_total.shape)
        # print(Y_valid_total.shape)

        # Initialize the class order
        order, order_list = self.init_class_order()
        np.random.seed(None)

        # Set empty lists for the data
        # _cumuls means all phases and _ori means 0-th phase
        X_train_cumuls = []
        Y_train_cumuls = []
        X_valid_cumuls = []
        Y_valid_cumuls = []
        X_protoset_cumuls = []
        Y_protoset_cumuls = []

        # Initialize the prototypes
        alpha_dr_herding, prototypes = self.init_prototypes(self.dictionary_size, order, X_train_total, Y_train_total)

        # Set the starting iteration
        # We start training the class-incremental learning system from 50 classes to provide a good initial encoder
        # and start_iter means 0-th phase, the following phase is added by iteration
        start_iter = int((self.args.num_classes-self.args.nb_cl_fg) / self.args.nb_cl) - 1

        # Set the models and some parameter to None
        # These models and parameters will be assigned in the following phases
        tg_model = None
        ref_model = None

        for iteration in range(start_iter, int(self.args.num_classes / self.args.nb_cl)):
            print("iteration=", iteration)

            # Initialize models for the current phase
            tg_model, ref_model, last_iter, lamda_mult, cur_lamda = self.init_current_phase_model(
                iteration, start_iter, tg_model)

            # Initialize datasets for the current phase
            if iteration == start_iter:
                indices_train, X_train_cumul, Y_train_cumul, X_valid_cumul, Y_valid_cumul, \
                X_train_cumuls, Y_train_cumuls, X_valid_cumuls, Y_valid_cumuls, \
                X_train, map_Y_train, map_Y_valid_cumul, X_protoset_cumuls, Y_protoset_cumuls, X_valid_ori, Y_valid_ori = \
                    self.init_current_phase_dataset(iteration, start_iter, last_iter, order, order_list, 
                                                    X_train_total, Y_train_total, X_valid_total, Y_valid_total, \
                                                    X_train_cumuls, Y_train_cumuls, X_valid_cumuls, Y_valid_cumuls,
                                                    X_protoset_cumuls, Y_protoset_cumuls)
            else:
                indices_train, X_valid_cumul, X_train_cumul, Y_valid_cumul, Y_train_cumul, \
                X_train_cumuls, Y_train_cumuls, X_valid_cumuls, Y_valid_cumuls, \
                X_train, map_Y_train, map_Y_valid_cumul, X_protoset_cumuls, Y_protoset_cumuls, X_protoset, Y_protoset = \
                    self.init_current_phase_dataset(iteration, start_iter, last_iter, order, order_list,
                                                    X_train_total, Y_train_total, X_valid_total, Y_valid_total,
                                                    X_train_cumuls, Y_train_cumuls, X_valid_cumuls, Y_valid_cumuls,
                                                    X_protoset_cumuls, Y_protoset_cumuls)

            # judge iteration equals start_iter or not
            is_start_iteration = (iteration == start_iter)

            # Update training and test dataloader
            # 这里就根本没有用到X_train_cumul，这个是用来测试上限的
            trainloader, testloader = self.update_train_and_valid_loader(
                X_train, map_Y_train, X_valid_cumul, map_Y_valid_cumul)

            # Set the names for the checkpoints
            # iter_{} means 0-th or i-th phase
            ckp_name = osp.join(self.save_path, 'cp_iter_{}_model.pth'.format(iteration))
            print('Check point name: ', ckp_name)

            # Start training from the checkpoints
            if self.args.resume and os.path.exists(ckp_name):
                print("###############################")
                print("Loading models from checkpoint")
                tg_model = torch.load(ckp_name)
                print("###############################")
            # Start training (if we don't resume the models from the checkppoints)
            else:
                # Set the optimizer
                tg_optimizer, tg_lr_scheduler = self.set_optimizer(iteration, start_iter, tg_model)

                tg_model = tg_model.to(self.device)
                if iteration > start_iter:
                    ref_model = ref_model.to(self.device)

                # train and eval progress
                if self.args.re_imprint_weights and self.args.multiple_distillation:
                    print("incremental_train_and_eval_RIW_MD")
                    # ReImprint weights
                    proto = self.set_proto_set(iteration, last_iter, X_protoset_cumuls, prototypes)
                    if iteration > start_iter and self.args.re_imprint_weights:
                        tg_model = self.imprint_weights(tg_model, iteration, X_train, map_Y_train, self.dictionary_size,
                                                        proto)
                    tg_model = incremental_train_and_eval_MD(
                        self.args, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, trainloader, testloader,
                        iteration, start_iter, cur_lamda)
                elif self.args.re_imprint_weights:
                    print("incremental_train_and_eval_RIM")
                    # ReImprint weights
                    proto = self.set_proto_set(iteration, last_iter, X_protoset_cumuls, Y_protoset_cumuls, prototypes)
                    if iteration > start_iter and self.args.imprint_weights:
                        tg_model = self.imprint_weights(tg_model, iteration, X_train, map_Y_train, self.dictionary_size,
                                                        proto)
                    tg_model = incremental_train_and_eval(
                        self.args, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, trainloader, testloader,
                        iteration, start_iter, cur_lamda)
                elif self.args.multiple_distillation:
                    print("incremental_train_and_eval_MD")
                    tg_model = incremental_train_and_eval_MD(
                        self.args, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, trainloader, testloader,
                        iteration, start_iter, cur_lamda)
                else:
                    print("incremental_train_and_eval")
                    tg_model = incremental_train_and_eval(
                        self.args, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, trainloader, testloader,
                        iteration, start_iter, cur_lamda)

            # save the model from current iteration
            if is_start_iteration:
                torch.save(tg_model, ckp_name)

            print('\nSelect the exemplars')
            # Select the exemplars according to the current model after train
            X_protoset_cumuls, Y_protoset_cumuls, class_means, alpha_dr_herding = self.set_exemplar_set(
                tg_model, iteration, last_iter, order, alpha_dr_herding, prototypes)

            # Compute the accuracies for current phase
            top1_acc_list_ori, top1_acc_list_cumul = \
                self.compute_acc(class_means, order, order_list, tg_model,
                                 X_valid_ori, Y_valid_ori, X_valid_cumul, Y_valid_cumul,
                                 iteration, top1_acc_list_ori, top1_acc_list_cumul)

            # 这还真他妈的叫做average accuracy，那种按权重的他妈的叫做weighted accuracy
            # Compute the average accuracy
            num_of_testing = iteration - start_iter + 1
            avg_cumul_acc_fc = np.sum(top1_acc_list_cumul[start_iter:, 0]) / num_of_testing
            avg_cumul_acc_proto = np.sum(top1_acc_list_cumul[start_iter:, 1]) / num_of_testing
            avg_cumul_acc_ncm = np.sum(top1_acc_list_cumul[start_iter:, 2]) / num_of_testing
            print('Computing average accuracy...')
            print("  Average accuracy (FC)         :\t\t{:.2f} %".format(avg_cumul_acc_fc))
            print("  Average accuracy (Proto)      :\t\t{:.2f} %".format(avg_cumul_acc_proto))
            # print("  Average accuracy (NCM)        :\t\t{:.2f} %".format(avg_cumul_acc_ncm))
            # Write the results to the tensorboard
            # self.train_writer.add_scalar('avg_acc/fc', float(avg_cumul_acc_fc), iteration)
            # self.train_writer.add_scalar('avg_acc/proto', float(avg_cumul_acc_icarl), iteration)

        # Save the results and close the tensorboard writer
        torch.save(top1_acc_list_ori, osp.join(self.save_path, 'acc_list_ori.pth'))
        torch.save(top1_acc_list_cumul, osp.join(self.save_path, 'acc_list_cumul.pth'))
        # self.train_writer.close()
