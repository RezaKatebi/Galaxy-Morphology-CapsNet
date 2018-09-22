# ======================================================================
# Author: Reza Katebi
# This code runs the CNN
# run this code for the training
# ======================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from train_cnn import ConvNet


# Set parameters for CNNs.
parser = argparse.ArgumentParser('CNN Exercise.')
parser.add_argument('--learning_rate',
                    type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--num_epochs',
                    type=int,
                    default=200,
                    help='Number of epochs to run trainer.')
parser.add_argument('--beta',
                    type=float,
                    default=0.1,
                    help='decay rate of L2 regulization.')
parser.add_argument('--batch_size',
                    type=int, default=20,
                    help='Batch size. Must divide evenly into the dataset sizes.')
parser.add_argument('--input_data_dir',
                    type=str,
                    default='./data/mnist',
                    help='Directory to put the training data.')


FLAGS = None
FLAGS, unparsed = parser.parse_known_args()


cnn = ConvNet()
accuracy = cnn.train_and_evaluate(FLAGS)

# Output accuracy
print(20 * '*' + 'model' + 20 * '*')
print('accuracy is %f' % (accuracy))
print()
