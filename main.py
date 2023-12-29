import torchmetrics
import torchsummary
import torch
import torchaudio
from torchmetrics.classification import (MulticlassF1Score, MulticlassPrecision, 
                                            MulticlassRecall, MulticlassPrecisionRecallCurve,
                                            MulticlassROC, MulticlassConfusionMatrix, MulticlassAccuracy)
from Trainer import Trainer
import Evaluator
import argparse
import numpy as np
import parse

from dataset import AudioDataset, AudioDatasetEval
from model import BasicCNNNetwork

#REMOVE LATER TESTING PURPOSES
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

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
print (f"Using device: {device}")

# Load data using the dataloader
# The data is on /data/TAU-urban-acoustic-scenes-2022-mobile-development/audio folder 
# and the labels are on /data/TAU-urban-acoustic-scenes-2022-mobile-development/meta.csv
# The data is already split into train, and test sets.
#TODO: Implement the dataset class

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = 22050,
        n_fft=2048,
        hop_length=512,
        n_mels=64
    )

audiodataset = AudioDataset(
    'data/TAU-urban-acoustic-scenes-2022-mobile-development/meta.csv', 
    'data/TAU-urban-acoustic-scenes-2022-mobile-development/audio', 
    mel_spectrogram, 22050,
    'cuda'
    )

audio_evaluation_dataset = AudioDatasetEval(
    'data/TAU-urban-acoustic-scenes-2023-mobile-evaluation/evaluation_setup/fold1_test.csv', 
    'data/TAU-urban-acoustic-scenes-2023-mobile-evaluation/audio/', 
    mel_spectrogram, 22050,
    'cuda'
    )

# Val Train split
train = int(0.8 * len(audiodataset))
test = len(audiodataset) - train
train_data, test_data = torch.utils.data.random_split(audiodataset, [train, test])

eval_loader = torch.utils.data.DataLoader(audio_evaluation_dataset, batch_size=args.batch_size, shuffle=True)

train_loader = torch.utils.data.DataLoader(audiodataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)


# Import the model (from model_path)

imp = __import__(args.model_file[:args.model_file.index(".")])

# Create the model
model = getattr(imp, args.model_class)().to(device)

# Print the model summary
torchsummary.summary(model, (1, 64,44),32)

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
        'model': model.to(device),
        'name': args.exp_name,
        'train_loader': train_loader,
        'val_loader': test_loader,
        'criterion': criterion,
        'optimizer': optimizer,
        'end_epoch': args.start_epoch + args.n_epochs,
        'metrics': metrics
    }

    modelhyperparams.update(params)
    trainer = Trainer(modelhyperparams)
    dict_loss, dict_metrics = trainer.train()

if args.eval:
    evaluatorhyperparams = args.__dict__ 
    params = {
        'model': model.to(device),
        'eval_loader': eval_loader,
        'name': args.exp_name,
        'label_encoder': audiodataset.label_encoder,
    }
    evaluatorhyperparams.update(params)
    evaluator = Evaluator.Evaluator(evaluatorhyperparams)
    evaluator.eval()

# TODO: Eval + save predictions, after dataset is correctly implemented
#if args.eval:
    #evaluator = evaluate.evaluation(model, test_loader, metrics, args.exp_name)
    #evaluator.eval()

