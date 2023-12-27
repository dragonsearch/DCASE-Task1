import argparse


def parse(parser):

    # Model parameters
    parser.add_argument('--model_class', type=str, default='resnet18', help='model class name')
    parser.add_argument('--model_file', type=str, default='resnet18', help='model class .py file, must be in same folder as main.py')
    parser.add_argument('--loss', type=str, default='CrossEntropyLoss', help='loss function to use')

    # General parameters
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use')
    

    # Training parameters
    parser.add_argument('--train', type=bool, default=True, help='To train the model or not')
    #parser.add_argument('--train_path', type= str, default= None, help='path to the data folder')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--start_epoch', type=int, default=1, help='epoch to start training from')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')

    # Evaluation parameters

    parser.add_argument('--eval', type=bool, default=True, help='To evaluate the model or not')
    parser.add_argument('--exp_name', type=str, default='resnet18', help='name of the experiment')
    parser.add_argument('--test_path', type=str, default='resnet18', help='test path')

    # Metrics parameter
    parser.add_argument('--metrics', type=str, default="{'MulticlassAccuracy': [10,1,'macro']}", help="torchmetrics to use example: {'MulticlassAccuracy': [25,1,\'macro\']} ")

    # Json parameters
    parser.add_argument('--json', type=str, default='', help='json file path')
    args = parser.parse_args()
    args = parse_json(args, parser)
    return args

import json

def parse_json(args, parser):
    if args.json != '':
        with open(args.json, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)
    return args

