import argparse
import torch


parser = argparse.ArgumentParser(prog='CapsNet', description='Example of CapsNet')

parser.add_argument('--epochs', type=int, default=5,
                    help='Number of epochs to train for (default: 5)')
parser.add_argument('--data_path', type=str, default='./data',
                    help='directory to find data (default: ./data)')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size (default: 128)')
parser.add_argument('--use_gpu', default=torch.cuda.is_available(), action='store_true',
                    help='to use gpu or not (default: true if gpu detected)')

parser.add_argument('--lr', type=float, default=0.01,
                    help='ADAM learning rate (default: 0.01)')

parser.add_argument('--log_interval', type=int, default=10,
                    help='number of batches between logging (default: 1)')

parser.add_argument('--visdom', default=False, action='store_true',
                    help='Whether or not to use visdom for plotting progrss (default: false')

parser.add_argument('--dataset', type=str, default='MNIST',
                    help='The dataset to train on, currently supported: MNIST, Fashion MNIST')

parser.add_argument('--load_checkpoint', type=str, default='',
                    help='path to load a previously trained model from (default: "" indicating not checkpointing)')

parser.add_argument('--checkpoint_interval', type=int, default=0,
                    help='how often to checkpoint the model (default: 0, dont checkpoint)')

parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                    help='dir to store checkpoints in (default: ./checkpoints)')

parser.add_argument('--gen_dir', type=str, default='./generated',
                    help='folder to store generated images in (default: ./generated)')

args = parser.parse_args()
