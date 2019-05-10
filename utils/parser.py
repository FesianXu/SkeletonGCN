#!/usr/bin/env python

import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser(description='Spatial Temporal Graph Convolution Network implemented based on DGL')

    parser.add_argument('--weights',
                        default=None,
                        help='''pretrained model's weights''')
    parser.add_argument('--work-dir',
                        default=None,
                        help='The folder for saving the models and configs')
    parser.add_argument('--config',
                        default=None,
                        help='path to the configuration file')

    parser.add_argument('--phase', 
                        default='train', 
                        help='must be train or test')

    parser.add_argument('--log-interval',
                        type=int,
                        default=100,
                        help='the interval for printing messages (#iteration)')
    parser.add_argument('--save-interval',
                        type=int,
                        default=10,
                        help='the interval for storing models (#iteration)')
    parser.add_argument('--eval-interval',
                        type=int,
                        default=5,
                        help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print-log-cmd',
                        type=str2bool,
                        default=True,
                        help='print logging or not in the bash envs')
    parser.add_argument('--print-log-tbx',
                        type=str2bool,
                        default=True,
                        help='print logging or not in the tensorboardX envs')
    

    parser.add_argument('--model-args',
                        type=dict,
                        default=dict(),
                        help='the arguments of the model')

    parser.add_argument('--model', default=None, help='the model will be used')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=64,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # optim
    parser.add_argument('--base-lr', type=float,
                        default=0.01, help='initial learning rate')
    parser.add_argument('--step',
                        type=int,
                        default=[20, 40, 60],
                        nargs='+',
                        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--device',
                        type=int,
                        default=0,
                        nargs='+',
                        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool,
                        default=False, help='use nesterov or not')
    parser.add_argument('--batch-size', type=int,
                        default=256, help='training batch size')
    parser.add_argument('--test-batch-size', type=int,
                        default=256, help='test batch size')
    parser.add_argument('--start-epoch',
                        type=int,
                        default=0,
                        help='start training from which epoch')
    parser.add_argument('--num-epoch',
                        type=int,
                        default=80,
                        help='stop training in which epoch')
    parser.add_argument('--weight-decay',
                        type=float,
                        default=0.0005,
                        help='weight decay for optimizer')

    parser.add_argument('--print-log',
                        type=str2bool,
                        default=True,
                        help='print logging or not')
    parser.add_argument('--force_run',
                        default=False,
                        help='run the inference anyway')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()
    print(p)
