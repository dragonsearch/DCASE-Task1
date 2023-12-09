import torchmetrics
import torchsummary
import torch
from torchmetrics.classification import (MulticlassF1Score, MulticlassPrecision, 
                                            MulticlassRecall, MulticlassPrecisionRecallCurve,
                                            MulticlassROC, MulticlassConfusionMatrix, MulticlassAccuracy)
from training import Training
import evaluate
import argparse
import numpy as np
import parse

#REMOVE LATER TESTING PURPOSES
from torchvision import datasets
from torchvision.transforms import ToTensor

# Parse arguments
parser = argparse.ArgumentParser(description='Runs the model with the specified parameters')
args = parse.parse(parser)
# Parse metrics dictionary this uses eval() to convert the string to a dictionary
# which is not particularly safe but is of no concern given the context
args.metrics = eval(args.metrics)

#
# General config
#

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device(args.device)

# Load data using the dataloader
# The data is on /data/TAU-urban-acoustic-scenes-2022-mobile-development/audio folder 
# and the labels are on /data/TAU-urban-acoustic-scenes-2022-mobile-development/meta.csv
# The data is already split into train, and test sets.


# REMOVE LATER TESTING PURPOSES
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_loader = torch.utils.data.DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)


# Import the model (from model_path)

imp = __import__(args.model_file[:args.model_file.index(".")])

# Create the model
model = getattr(imp, args.model_class)().to(device)

# Print the model summary
torchsummary.summary(model, (1, 28,28),64)

# Create the optimizer
optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)

# Create the loss function
criterion = getattr(torch.nn, args.loss)()

# Create the metrics

metrics = {metric : getattr(torchmetrics.classification, metric)(*args.metrics[metric]) for metric in args.metrics}

# Create the training loop

if args.train:
    modelhyperparams = args.__dict__
    params = {
        'model': model,
        'name': args.exp_name,
        'train_loader': train_loader,
        'val_loader': test_loader,
        'criterion': criterion,
        'optimizer': optimizer,
        'end_epoch': args.start_epoch + args.n_epochs,
        'metrics': metrics
    }
    modelhyperparams.update(params)
    trainer = Training(model, modelhyperparams)
    dict_loss, dict_metrics = trainer.train()


# TODO: Eval + save predictions, after dataset is correctly implemented
#if args.eval:
    #evaluator = evaluate.evaluation(model, test_loader, metrics, args.exp_name)
    #evaluator.eval()

