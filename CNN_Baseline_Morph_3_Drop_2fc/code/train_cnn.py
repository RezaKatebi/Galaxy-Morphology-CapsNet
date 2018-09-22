# ======================================================================
# Author: Reza Katebi
# This code contains the Baseline CNN model
# ======================================================================

import numpy
import time
import torch
import torch.nn as NN
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from DataLoader import DataLoad

# Change the shape of the input tensor


class ViewOP(torch.nn.Module):
    def __init__(self, *shape):
        super(ViewOP, self).__init__()
        self.shape = shape

    def forward(self, input):
        # output = input.view(input.size(0), -1)
        # print(output.shape)
        return input.view(input.size(0), -1)  # (self.shape)


########### Convolutional neural network class ############
class ConvNet(object):
    def __init__(self):
        if torch.cuda.is_available():
            self.use_gpu = True
        else:
            self.use_gpu = False

    def ConvBlock(self, num, layers, model, inp_fl, outp_fl):
        for i in range(layers):
            if i == 0:
                model.add_module('Conv2d_{}_{}'.format(num, i),
                                 NN.Conv2d(inp_fl, outp_fl, kernel_size=3, stride=1, padding=(1, 1)))
                model.add_module('BN_{}_{}'.format(num, i), NN.BatchNorm2d(outp_fl))
                model.add_module('ReLU_{}_{}'.format(num, i), NN.ReLU())
            else:
                model.add_module('Conv2d_{}_{}'.format(num, i),
                                 NN.Conv2d(outp_fl, outp_fl, kernel_size=3, stride=1, padding=(1, 1)))
                model.add_module('BN_{}_{}'.format(num, i), NN.BatchNorm2d(outp_fl))
                model.add_module('ReLU_{}_{}'.format(num, i), NN.ReLU())
        model.add_module('Pooling_{}_{}'.format(num, i), NN.MaxPool2d(2))

    def CNNModel(self, class_num):

        model = NN.Sequential()
        # Conv blocks
        model.add_module('Conv2d_1', NN.Conv2d(3, 512, kernel_size=9, stride=1))
        model.add_module('ReLU_1', NN.ReLU())
        model.add_module('Pooling_1', NN.MaxPool2d(2))
        model.add_module('Conv2d_2', NN.Conv2d(512, 256, kernel_size=5, stride=1))
        model.add_module('ReLU_2', NN.ReLU())
        model.add_module('Pooling_1', NN.MaxPool2d(2))

        # Flatten
        model.add_module('Flatten', ViewOP())
        model.add_module('Linear_1', NN.Linear(200704, 1024))
        model.add_module('ReLU_L_1', NN.ReLU())
        model.add_module('Dropout_1', NN.Dropout(0.5))
        # Fully connected layer 2
        model.add_module('Linear_2', NN.Linear(1024, 1024))
        model.add_module('ReLU_L_2', NN.ReLU())
        model.add_module('Dropout_2', NN.Dropout(0.5))
        # Output layer
        model.add_module('Linear_3', NN.Linear(1024, class_num))
        model.add_module('LogSoftmax', NN.LogSoftmax(dim=1))

        return model

    def augmentation(self, x, max_shift=2):
        _, _, height, width = x.size()

        h_shift, w_shift = numpy.random.randint(-max_shift, max_shift + 1, size=2)
        source_height_slice = slice(max(0, h_shift), h_shift + height)
        source_width_slice = slice(max(0, w_shift), w_shift + width)
        target_height_slice = slice(max(0, -h_shift), -h_shift + height)
        target_width_slice = slice(max(0, -w_shift), -w_shift + width)

        shifted_image = torch.zeros(*x.size())
        shifted_image[:, :, source_height_slice, source_width_slice] = x[:,
                                                                         :, target_height_slice, target_width_slice]
        return shifted_image.float()

    def augmentation_rotate(self, x):
        x = x.data.numpy()
        a = numpy.random.randint(0, 4)
        rot_x = numpy.rot90(x, k=a, axes=(2, 3)).copy()
        return torch.tensor(rot_x).float()

    def train(self, model, optimizer, train_loader, ftype, itype):
        correct = 0
        losses = 0
        for _, train_batch in enumerate(train_loader, 0):
            batch_x = train_batch[0] / 255
            batch_y = train_batch[1]

            X = Variable(self.augmentation(batch_x).type(ftype), requires_grad=False)
            Y = Variable(batch_y.type(itype), requires_grad=False)
            pred_Y = model(X)

            loss_func = F.nll_loss(pred_Y, Y)
            losses += loss_func.item() * batch_x.size(0)
            optimizer.zero_grad()
            loss_func.backward()
            optimizer.step()

            _, idx = torch.max(pred_Y, dim=1)

            # Move tensor from GPU to CPU.
            if self.use_gpu:
                idx = idx.cpu()
                Y = Y.cpu()

            idx = idx.data.numpy()
            Y = Y.data.numpy()
            correct += numpy.sum(idx == Y)
        accuracy = correct / len(train_loader.dataset)
        return losses/len(train_loader.dataset), accuracy
    # Evaluate the trained model on test set.

    def evaluate_model(self, model, test_loader, ftype, itype):
        model.eval()
        correct = 0
        for _, test_batch in enumerate(test_loader, 0):
            batch_x = test_batch[0] / 255
            batch_y = test_batch[1]

            X = Variable(self.augmentation(batch_x).type(ftype), requires_grad=False)
            Y = Variable(batch_y.type(itype), requires_grad=False)
            pred_Y = model(X)
            _, idx = torch.max(pred_Y, dim=1)

            # Move tensor from GPU to CPU.
            if self.use_gpu:
                idx = idx.cpu()
                Y = Y.cpu()

            idx = idx.data.numpy()
            Y = Y.data.numpy()
            correct += numpy.sum(idx == Y)
        accuracy = correct / len(test_loader.dataset)

        return accuracy

    # Entry point for training and evaluation.
    def train_and_evaluate(self, FLAGS):
        class_num = 2
        num_epochs = FLAGS.num_epochs
        batch_size = FLAGS.batch_size
        learning_rate = FLAGS.learning_rate

        # Set random number generator seed.
        torch.manual_seed(1024)
        if self.use_gpu:
            torch.cuda.manual_seed_all(1024)
        numpy.random.seed(1024)

        model = eval("self.CNNModel")(class_num)

        if self.use_gpu:
            ftype = torch.cuda.FloatTensor  # float type
            itype = torch.cuda.LongTensor  # int type
            model.cuda()
        else:
            ftype = torch.FloatTensor  # float type
            itype = torch.LongTensor  # int type

        params = model.parameters()

        #-------------------- Create optimizer ---------------------
        optimizer = optim.Adam(params)
        #--------------------- DataLoader --------------------------
        # Loading the custum dataset using DataLoader
        Galaxy = DataLoad(batch_size)
        test_accs = []
        # Train the model.
        train_accs = []
        for i in range(num_epochs):
            print(21 * '*', 'epoch', i+1, 21 * '*')
            start_time = time.time()
            loss_train, accuracy = self.train(model, optimizer, Galaxy.train_loader, ftype, itype)
            train_accs.append(accuracy)
            end_time = time.time()
            print('the training took: %d(s)' % (end_time - start_time))
            print('training accuracy is {} and losss is {}'.format(accuracy, loss_train))

            test_acc = self.evaluate_model(model, Galaxy.test_loader, ftype, itype)

            print("Accuracy of the trained model on test set is", test_acc)
            test_accs.append(test_acc)

        numpy.save('../results/test_accs', test_accs)
        numpy.save('../results/train_accs', train_accs)

        return self.evaluate_model(model, Galaxy.test_loader, ftype, itype)
