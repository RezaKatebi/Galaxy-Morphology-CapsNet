import sys
sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

BATCH_SIZE = 5
NUM_CLASSES = 37
NUM_EPOCHS = 30
NUM_ROUTING_ITERATIONS = 3


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1,
                                                                    transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


def augmentation(x, max_shift=2):
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:,
                                                                     :, target_height_slice, target_width_slice]
    return shifted_image.float()


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(
                num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(*priors.size())).cuda()
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=32 * 28 * 28, in_channels=8,
                                           out_channels=16)

        self.Linear = nn.Linear(16 * NUM_CLASSES, NUM_CLASSES)

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)
        x = (x ** 2).sum(dim=-1) ** 0.5
        # x = self.Linear(x.view(x.size(0), -1))

        return x


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.mse = nn.MSELoss(size_average=True)

    def forward(self, labels, x):

        return self.mse(x, labels)


if __name__ == "__main__":
    from torch.autograd import Variable
    from torch.optim import Adam
    from torchnet.engine import Engine
    # from torchvision.datasets.mnist import MNIST
    from tqdm import tqdm
    import torchnet as tnt

    model = CapsuleNet()
    # model.load_state_dict(torch.load('epochs/epoch_327.pt'))
    model.cuda()

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = Adam(model.parameters())

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    outputs_ = []
    lables_ = []

    capsule_loss = CapsuleLoss()

    def get_iterator(mode):
        X = np.load('../data/train_downsample.npy')[:61500]
        y = np.load('../data/train_labels_regression.npy')[:61500]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        if mode:
            data = torch.from_numpy(X_train).float()
            labels = torch.from_numpy(y_train)
        else:
            data = torch.from_numpy(X_test).float()
            labels = torch.from_numpy(y_test)

        tensor_dataset = tnt.dataset.TensorDataset([data, labels])
        return tensor_dataset.parallel(batch_size=BATCH_SIZE, num_workers=1, shuffle=mode)

    def processor(sample):
        data, labels, training = sample

        data = augmentation(data.float() / 255.0)
        labels = labels.to(torch.float)

        data = Variable(data).cuda()
        labels = Variable(labels).cuda()

        if training:
            Output = model(data, labels)
        else:
            Output = model(data)

        loss = capsule_loss(labels, Output)

        return loss, Output

    def reset_meters():
        meter_loss.reset()

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        meter_loss.add(state['loss'].item())

    def on_start_epoch(state):
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(state):
        print('[Epoch %d] Training Loss: %.4f' % (
            state['epoch'], np.sqrt(meter_loss.value()[0])))

        train_losses.append(np.sqrt(meter_loss.value()[0]))
        reset_meters()

        engine.test(processor, get_iterator(False))

        print('[Epoch %d] Testing Loss: %.4f' % (
            state['epoch'], np.sqrt(meter_loss.value()[0])))

        torch.save(model.state_dict(), '../epochs/epoch_%d.pt' % state['epoch'])
        test_losses.append(np.sqrt(meter_loss.value()[0]))

    # def on_start(state):
    #     state['epoch'] = 327
    #
    # engine.hooks['on_start'] = on_start
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    train_losses = []
    test_losses = []
    engine.train(processor, get_iterator(True), maxepoch=NUM_EPOCHS, optimizer=optimizer)

    np.save("../results/train_losses", train_losses)
    np.save("../results/test_losses", test_losses)
