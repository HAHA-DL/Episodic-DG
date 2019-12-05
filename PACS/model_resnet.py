import os

dirpath = os.pardir
import sys

sys.path.append(dirpath)
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.optim import lr_scheduler

import resnet_epi_fcr
import resnet_vanilla
from common.data_reader import BatchImageGenerator
from common.utils import *


class ModelAggregate:
    def __init__(self, flags):

        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        self.setup(flags)
        self.setup_path(flags)
        self.configure(flags)

    def setup(self, flags):
        torch.backends.cudnn.deterministic = flags.deterministic
        print('torch.backends.cudnn.deterministic:', torch.backends.cudnn.deterministic)
        fix_all_seed(flags.seed)

        self.network = resnet_vanilla.resnet18(pretrained=False, num_classes=flags.num_classes)
        self.network = self.network.cuda()

        print(self.network)
        print('flags:', flags)
        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        flags_log = os.path.join(flags.logs, 'flags_log.txt')
        write_log(flags, flags_log)

        self.load_state_dict(flags, self.network)

    def setup_path(self, flags):

        root_folder = flags.data_root
        train_data = ['art_painting_train.hdf5',
                      'cartoon_train.hdf5',
                      'photo_train.hdf5',
                      'sketch_train.hdf5']

        val_data = ['art_painting_val.hdf5',
                    'cartoon_val.hdf5',
                    'photo_val.hdf5',
                    'sketch_val.hdf5']

        test_data = ['art_painting_test.hdf5',
                     'cartoon_test.hdf5',
                     'photo_test.hdf5',
                     'sketch_test.hdf5']

        self.train_paths = []
        for data in train_data:
            path = os.path.join(root_folder, data)
            self.train_paths.append(path)

        self.val_paths = []
        for data in val_data:
            path = os.path.join(root_folder, data)
            self.val_paths.append(path)

        unseen_index = flags.unseen_index

        self.unseen_data_path = os.path.join(root_folder, test_data[unseen_index])
        self.train_paths.remove(self.train_paths[unseen_index])
        self.val_paths.remove(self.val_paths[unseen_index])

        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        flags_log = os.path.join(flags.logs, 'path_log.txt')
        write_log(str(self.train_paths), flags_log)
        write_log(str(self.val_paths), flags_log)
        write_log(str(self.unseen_data_path), flags_log)

        self.batImageGenTrains = []
        for train_path in self.train_paths:
            batImageGenTrain = BatchImageGenerator(flags=flags, file_path=train_path, stage='train',
                                                   b_unfold_label=False)
            self.batImageGenTrains.append(batImageGenTrain)

        self.batImageGenVals = []
        for val_path in self.val_paths:
            batImageGenVal = BatchImageGenerator(flags=flags, file_path=val_path, stage='val',
                                                 b_unfold_label=False)
            self.batImageGenVals.append(batImageGenVal)

        self.batImageGenTest = BatchImageGenerator(flags=flags, file_path=self.unseen_data_path, stage='test',
                                                   b_unfold_label=False)

    def load_state_dict(self, flags, nn):

        if flags.state_dict:

            try:
                tmp = torch.load(flags.state_dict)
                if 'state' in tmp.keys():
                    pretrained_dict = tmp['state']
                else:
                    pretrained_dict = tmp
            except:
                pretrained_dict = model_zoo.load_url(flags.state_dict)

            model_dict = nn.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict and v.size() == model_dict[k].size()}

            print('model dict keys:', len(model_dict.keys()), 'pretrained keys:', len(pretrained_dict.keys()))
            print('model dict keys:', model_dict.keys(), 'pretrained keys:', pretrained_dict.keys())
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            nn.load_state_dict(model_dict)

    def configure(self, flags):

        for name, para in self.network.named_parameters():
            print(name, para.size())

        self.optimizer = sgd(parameters=self.network.parameters(),
                             lr=flags.lr,
                             weight_decay=flags.weight_decay,
                             momentum=flags.momentum)

        self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=flags.step_size, gamma=0.1)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def train(self, flags):
        self.network.train()
        self.network.bn_eval()
        self.best_accuracy_val = -1

        for ite in range(flags.loops_train):

            self.scheduler.step(epoch=ite)

            # get the inputs and labels from the data reader
            total_loss = 0.0
            for index in range(len(self.batImageGenTrains)):
                images_train, labels_train = self.batImageGenTrains[index].get_images_labels_batch()

                inputs, labels = torch.from_numpy(
                    np.array(images_train, dtype=np.float32)), torch.from_numpy(
                    np.array(labels_train, dtype=np.float32))

                # wrap the inputs and labels in Variable
                inputs, labels = Variable(inputs, requires_grad=False).cuda(), \
                                 Variable(labels, requires_grad=False).long().cuda()

                # forward with the adapted parameters
                outputs, _ = self.network(x=inputs)

                # loss
                loss = self.loss_fn(outputs, labels)

                total_loss += loss

            # init the grad to zeros first
            self.optimizer.zero_grad()

            # backward your network
            total_loss.backward()

            # optimize the parameters
            self.optimizer.step()

            if ite < 500 or ite % 500 == 0:
                print(
                    'ite:', ite, 'total loss:', total_loss.cpu().item(), 'lr:',
                    self.scheduler.get_lr()[0])

            flags_log = os.path.join(flags.logs, 'loss_log.txt')
            write_log(
                str(total_loss.item()),
                flags_log)

            if ite % flags.test_every == 0 and ite is not 0:
                self.test_workflow(self.batImageGenVals, flags, ite)

    def test_workflow(self, batImageGenVals, flags, ite):

        accuracies = []
        for count, batImageGenVal in enumerate(batImageGenVals):
            accuracy_val = self.test(batImageGenTest=batImageGenVal, flags=flags, ite=ite,
                                     log_dir=flags.logs, log_prefix='val_index_{}'.format(count))

            accuracies.append(accuracy_val)

        mean_acc = np.mean(accuracies)

        if mean_acc > self.best_accuracy_val:
            self.best_accuracy_val = mean_acc

            acc_test = self.test(batImageGenTest=self.batImageGenTest, flags=flags, ite=ite,
                                 log_dir=flags.logs, log_prefix='dg_test')

            f = open(os.path.join(flags.logs, 'Best_val.txt'), mode='a')
            f.write(
                'ite:{}, best val accuracy:{}, test accuracy:{}\n'.format(ite, self.best_accuracy_val,
                                                                          acc_test))
            f.close()

            if not os.path.exists(flags.model_path):
                os.makedirs(flags.model_path)

            outfile = os.path.join(flags.model_path, 'best_model.tar')
            torch.save({'ite': ite, 'state': self.network.state_dict()}, outfile)

    def bn_process(self, flags):
        if flags.bn_eval == 1:
            self.network.bn_eval()

    def test(self, flags, ite, log_prefix, log_dir='logs/', batImageGenTest=None):

        # switch on the network test mode
        self.network.eval()

        if batImageGenTest is None:
            batImageGenTest = BatchImageGenerator(flags=flags, file_path='', stage='test', b_unfold_label=True)

        images_test = batImageGenTest.images
        labels_test = batImageGenTest.labels

        threshold = 50
        if len(images_test) > threshold:

            n_slices_test = int(len(images_test) / threshold)
            indices_test = []
            for per_slice in range(n_slices_test - 1):
                indices_test.append(int(len(images_test) * (per_slice + 1) / n_slices_test))
            test_image_splits = np.split(images_test, indices_or_sections=indices_test)

            # Verify the splits are correct
            test_image_splits_2_whole = np.concatenate(test_image_splits)
            assert np.all(images_test == test_image_splits_2_whole)

            # split the test data into splits and test them one by one
            test_image_preds = []
            for test_image_split in test_image_splits:
                images_test = Variable(torch.from_numpy(np.array(test_image_split, dtype=np.float32))).cuda()
                tuples = self.network(images_test)

                predictions = tuples[-1]['Predictions']
                predictions = predictions.cpu().data.numpy()
                test_image_preds.append(predictions)

            # concatenate the test predictions first
            predictions = np.concatenate(test_image_preds)
        else:
            images_test = Variable(torch.from_numpy(np.array(images_test, dtype=np.float32))).cuda()
            tuples = self.network(images_test)

            predictions = tuples[-1]['Predictions']
            predictions = predictions.cpu().data.numpy()

        accuracy = compute_accuracy(predictions=predictions, labels=labels_test)
        print('----------accuracy test----------:', accuracy)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        f = open(os.path.join(log_dir, '{}.txt'.format(log_prefix)), mode='a')
        f.write('ite:{}, accuracy:{}\n'.format(ite, accuracy))
        f.close()

        # switch on the network train mode
        self.network.train()
        self.bn_process(flags)

        return accuracy


class ModelEpiFCR(ModelAggregate):
    def __init__(self, flags):
        ModelAggregate.__init__(self, flags)

    def configure(self, flags):

        for name, para in self.ds_nn.named_parameters():
            print(name, para.size())
        for name, para in self.agg_nn.named_parameters():
            print(name, para.size())

        self.optimizer_ds_nn = sgd(parameters=[{'params': self.ds_nn.parameters()}],
                                   lr=flags.lr,
                                   weight_decay=flags.weight_decay,
                                   momentum=flags.momentum)

        self.scheduler_ds_nn = lr_scheduler.StepLR(optimizer=self.optimizer_ds_nn, step_size=flags.step_size,
                                                   gamma=0.1)

        self.optimizer_agg = sgd(parameters=[{'params': self.agg_nn.feature.parameters()},
                                             {'params': self.agg_nn.classifier.parameters()}],
                                 lr=flags.lr,
                                 weight_decay=flags.weight_decay,
                                 momentum=flags.momentum)

        self.scheduler_agg = lr_scheduler.StepLR(optimizer=self.optimizer_agg,
                                                 step_size=flags.step_size, gamma=0.1)

        self.loss_fn = crossentropyloss().cuda()

    def setup(self, flags):
        torch.backends.cudnn.deterministic = flags.deterministic
        print('torch.backends.cudnn.deterministic:', torch.backends.cudnn.deterministic)
        fix_all_seed(flags.seed)

        self.ds_nn = resnet_epi_fcr.DomainSpecificNN(num_classes=flags.num_classes).cuda()
        self.agg_nn = resnet_epi_fcr.DomainAGG(num_classes=flags.num_classes).cuda()

        print(self.ds_nn)
        print(self.agg_nn)
        print('flags:', flags)

        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        flags_log = os.path.join(flags.logs, 'flags_log.txt')
        write_log(flags, flags_log)

        self.load_state_dict(flags, self.ds_nn.feature1)
        self.load_state_dict(flags, self.ds_nn.feature2)
        self.load_state_dict(flags, self.ds_nn.feature3)
        self.load_state_dict(flags, self.agg_nn.feature)

        self.configure(flags)

        self.loss_weight_epic = flags.loss_weight_epic
        self.loss_weight_epif = flags.loss_weight_epif
        self.loss_weight_epir = flags.loss_weight_epir

        self.best_accuracy_val = -1.0

    def setup_path(self, flags):

        root_folder = flags.data_root
        train_data = ['art_painting_train.hdf5',
                      'cartoon_train.hdf5',
                      'photo_train.hdf5',
                      'sketch_train.hdf5']

        val_data = ['art_painting_val.hdf5',
                    'cartoon_val.hdf5',
                    'photo_val.hdf5',
                    'sketch_val.hdf5']

        test_data = ['art_painting_test.hdf5',
                     'cartoon_test.hdf5',
                     'photo_test.hdf5',
                     'sketch_test.hdf5']

        self.train_paths = []
        for data in train_data:
            path = os.path.join(root_folder, data)
            self.train_paths.append(path)

        self.val_paths = []
        for data in val_data:
            path = os.path.join(root_folder, data)
            self.val_paths.append(path)

        unseen_index = flags.unseen_index

        self.unseen_data_path = os.path.join(root_folder, test_data[unseen_index])
        self.train_paths.remove(self.train_paths[unseen_index])
        self.val_paths.remove(self.val_paths[unseen_index])

        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        flags_log = os.path.join(flags.logs, 'path_log.txt')
        write_log(str(self.train_paths), flags_log)
        write_log(str(self.val_paths), flags_log)
        write_log(str(self.unseen_data_path), flags_log)

        self.batImageGenTrains = []
        for train_path in self.train_paths:
            batImageGenTrain = BatchImageGenerator(flags=flags, file_path=train_path, stage='train',
                                                   b_unfold_label=False)
            self.batImageGenTrains.append(batImageGenTrain)

        self.batImageGenTrainsDg = []
        for train_path in self.train_paths:
            batImageGenTrain = BatchImageGenerator(flags=flags, file_path=train_path, stage='train',
                                                   b_unfold_label=False)
            self.batImageGenTrainsDg.append(batImageGenTrain)

        self.batImageGenVals = []
        for val_path in self.val_paths:
            batImageGenVal = BatchImageGenerator(flags=flags, file_path=val_path, stage='val',
                                                 b_unfold_label=False)
            self.batImageGenVals.append(batImageGenVal)

        self.batImageGenTest = BatchImageGenerator(flags=flags, file_path=self.unseen_data_path, stage='test',
                                                   b_unfold_label=False)

        self.candidates = np.arange(0, len(self.batImageGenTrains))

    def bn_process(self, flags):
        if flags.bn_eval == 1:
            self.ds_nn.bn_eval()
            self.agg_nn.bn_eval()

    def train_agg_nn(self, ite, flags):

        candidates = list(self.candidates)
        index_val = np.random.choice(candidates, size=1)[0]
        candidates.remove(index_val)
        index_trn = np.random.choice(candidates, size=1)[0]
        assert index_trn != index_val

        self.scheduler_agg.step(ite)

        # get the inputs and labels from the data reader
        agg_xent_loss = 0.0
        epir_loss = 0.0
        epic_loss = 0.0
        epif_loss = 0.0
        for index in range(len(self.batImageGenTrainsDg)):
            images_train, labels_train = self.batImageGenTrainsDg[index].get_images_labels_batch()

            inputs, labels = torch.from_numpy(
                np.array(images_train, dtype=np.float32)), torch.from_numpy(
                np.array(labels_train, dtype=np.float32))

            # wrap the inputs and labels in Variable
            inputs, labels = Variable(inputs, requires_grad=False).cuda(), \
                             Variable(labels, requires_grad=False).long().cuda()

            # forward
            outputs_agg, outputs_rand, _ = self.agg_nn(x=inputs, agg_only=False)

            # loss
            agg_xent_loss += self.loss_fn(outputs_agg, labels)
            epir_loss += self.loss_fn(outputs_rand, labels) * self.loss_weight_epir

            if index == index_val:
                assert index != index_trn

                if ite >= flags.ite_train_epi_c:
                    net = self.ds_nn.features[index_trn](inputs)
                    outputs_val, _ = self.agg_nn.classifier(net)
                    epic_loss += self.loss_fn(outputs_val, labels) * self.loss_weight_epic

                if ite >= flags.ite_train_epi_f:
                    net = self.agg_nn.feature(inputs)
                    outputs_val, _ = self.ds_nn.classifiers[index_trn](net)
                    epif_loss += self.loss_fn(outputs_val, labels) * self.loss_weight_epif

        # init the grad to zeros first
        self.optimizer_agg.zero_grad()

        # backward your network
        (agg_xent_loss + epir_loss + epic_loss + epif_loss).backward()

        # optimize the parameters
        self.optimizer_agg.step()

        if ite % 500 == 0 or ite < flags.loops_warm + 500:
            print(
                'ite:', ite,
                'agg_xent_loss:', agg_xent_loss.item(),
                'epir_loss:', epir_loss.item(),
                'epic_loss:', epic_loss.item() if type(epic_loss) is not float else epic_loss,
                'epif_loss:', epif_loss.item() if type(epif_loss) is not float else epif_loss,
                'lr:', self.scheduler_agg.get_lr()[0])

        flags_log = os.path.join(flags.logs, 'agg_xent_loss.txt')
        write_log(str(agg_xent_loss.item()), flags_log)
        flags_log = os.path.join(flags.logs, 'epir_loss.txt')
        write_log(str(epir_loss.item()), flags_log)
        if ite >= flags.ite_train_epi_c:
            flags_log = os.path.join(flags.logs, 'epic_loss.txt')
            write_log(str(epic_loss.item()), flags_log)
        if ite >= flags.ite_train_epi_f:
            flags_log = os.path.join(flags.logs, 'epif_loss.txt')
            write_log(str(epif_loss.item()), flags_log)

    def train_agg_nn_warm(self, ite, flags):

        self.scheduler_agg.step(ite)

        # get the inputs and labels from the data reader
        agg_xent_loss = 0.0
        for index in range(len(self.batImageGenTrainsDg)):
            images_train, labels_train = self.batImageGenTrainsDg[index].get_images_labels_batch()

            inputs, labels = torch.from_numpy(
                np.array(images_train, dtype=np.float32)), torch.from_numpy(
                np.array(labels_train, dtype=np.float32))

            # wrap the inputs and labels in Variable
            inputs, labels = Variable(inputs, requires_grad=False).cuda(), \
                             Variable(labels, requires_grad=False).long().cuda()

            # forward
            outputs, _, _ = self.agg_nn(x=inputs)

            # loss
            agg_xent_loss += self.loss_fn(outputs, labels)

        # init the grad to zeros first
        self.optimizer_agg.zero_grad()

        # backward your network
        agg_xent_loss.backward()

        # optimize the parameters
        self.optimizer_agg.step()

        if ite % 500 == 0 or ite < flags.loops_warm + 500:
            print(
                'ite:', ite,
                'agg_xent_loss:', agg_xent_loss.item(),
                'lr:', self.scheduler_agg.get_lr()[0])

        flags_log = os.path.join(flags.logs, 'agg_xent_loss.txt')
        write_log(str(agg_xent_loss.item()), flags_log)

    def train_ds_nn(self, ite, flags):

        self.scheduler_ds_nn.step(ite)

        # get the inputs and labels from the data reader
        xent_loss = 0.0
        for index in range(len(self.batImageGenTrains)):
            images_train, labels_train = self.batImageGenTrains[index].get_images_labels_batch()

            inputs, labels = torch.from_numpy(
                np.array(images_train, dtype=np.float32)), torch.from_numpy(
                np.array(labels_train, dtype=np.float32))

            # wrap the inputs and labels in Variable
            inputs, labels = Variable(inputs, requires_grad=False).cuda(), \
                             Variable(labels, requires_grad=False).long().cuda()

            # forward
            outputs, _ = self.ds_nn(x=inputs, domain=index)

            # loss
            loss = self.loss_fn(outputs, labels)

            xent_loss += loss

        self.optimizer_ds_nn.zero_grad()

        # backward your network
        xent_loss.backward()

        # optimize the parameters
        self.optimizer_ds_nn.step()

        if ite % 500 == 0 or ite < 500:
            print(
                'ite:', ite,
                'xent_loss:', xent_loss.item(),
                'lr:', self.scheduler_ds_nn.get_lr()[0])

        flags_log = os.path.join(flags.logs, 'ds_nn_loss_log.txt')
        write_log(str(xent_loss.item()), flags_log)

    def test_workflow(self, batImageGenVals, flags, ite, prefix=''):

        accuracies = []
        for count, batImageGenVal in enumerate(batImageGenVals):
            accuracy_val = self.test(batImageGenTest=batImageGenVal, flags=flags, ite=ite,
                                     log_dir=flags.logs, log_prefix='val_index_{}'.format(count))

            accuracies.append(accuracy_val)
        mean_acc = np.mean(accuracies)

        if mean_acc > self.best_accuracy_val:
            self.best_accuracy_val = mean_acc

            acc_test = self.test(batImageGenTest=self.batImageGenTest, flags=flags, ite=ite,
                                 log_dir=flags.logs, log_prefix='dg_test')

            f = open(os.path.join(flags.logs, 'Best_val.txt'), mode='a')
            f.write(
                'ite:{}, best val accuracy:{}, test accuracy:{}\n'.format(ite, self.best_accuracy_val,
                                                                          acc_test))
            f.close()

            if not os.path.exists(flags.model_path):
                os.makedirs(flags.model_path)

            outfile = os.path.join(flags.model_path, '{}best_agg.tar'.format(prefix))
            torch.save({'ite': ite, 'state': self.agg_nn.state_dict()}, outfile)

    def test(self, flags, ite, log_prefix, log_dir='logs/', batImageGenTest=None):

        # switch on the network test mode
        self.agg_nn.eval()

        images_test = batImageGenTest.images
        labels_test = batImageGenTest.labels

        threshold = 50
        if len(images_test) > threshold:

            n_slices_test = int(len(images_test) / threshold)
            indices_test = []
            for per_slice in range(n_slices_test - 1):
                indices_test.append(int(len(images_test) * (per_slice + 1) / n_slices_test))
            test_image_splits = np.split(images_test, indices_or_sections=indices_test)

            # Verify the splits are correct
            test_image_splits_2_whole = np.concatenate(test_image_splits)
            assert np.all(images_test == test_image_splits_2_whole)

            # split the test data into splits and test them one by one
            test_image_preds = []
            for test_image_split in test_image_splits:
                images_test = Variable(torch.from_numpy(np.array(test_image_split, dtype=np.float32))).cuda()
                tuples = self.agg_nn(images_test, agg_only=True)

                predictions = tuples[-1]['Predictions']
                predictions = predictions.cpu().data.numpy()
                test_image_preds.append(predictions)

            # concatenate the test predictions first
            predictions = np.concatenate(test_image_preds)
        else:
            images_test = Variable(torch.from_numpy(np.array(images_test, dtype=np.float32))).cuda()
            tuples = self.agg_nn(images_test, agg_only=True)

            predictions = tuples[-1]['Predictions']
            predictions = predictions.cpu().data.numpy()

        accuracy = compute_accuracy(predictions=predictions, labels=labels_test)
        print('----------accuracy test----------:', accuracy)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        f = open(os.path.join(log_dir, '{}.txt'.format(log_prefix)), mode='a')
        f.write('ite:{}, accuracy:{}\n'.format(ite, accuracy))
        f.close()

        # switch on the network train mode
        self.agg_nn.train()
        self.bn_process(flags)

        return accuracy

    def train(self, flags):
        self.ds_nn.train()
        self.agg_nn.train()
        self.bn_process(flags)

        for ite in range(flags.loops_train):

            if ite <= flags.loops_warm:

                self.train_ds_nn(ite, flags)

                if flags.warm_up_agg == 1 and ite <= flags.loops_agg_warm:
                    self.train_agg_nn_warm(ite, flags)

                if ite % flags.test_every == 0 and ite is not 0 and flags.warm_up_agg != 1:
                    self.test_workflow(self.batImageGenVals, flags, ite, prefix='')

            else:

                self.train_ds_nn(ite, flags)
                self.train_agg_nn(ite, flags)

                if ite % flags.test_every == 0 and ite is not 0:
                    self.test_workflow(self.batImageGenVals, flags, ite, prefix='')
