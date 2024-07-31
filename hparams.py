import torchmetrics
import torch
from torchmetrics.classification import (MulticlassF1Score, MulticlassPrecision, 
                                            MulticlassRecall, MulticlassPrecisionRecallCurve,
                                            MulticlassROC, MulticlassConfusionMatrix, MulticlassAccuracy)
import importlib
from metrics.devAccuracy import DevAccuracy


def first_hparams(trial):
    trial_model_params = {
        'batch_size': trial.suggest_categorical('batch_size', [32,128,512]),
        'name': trial.suggest_categorical('exp_name', ["teacher_sgd_kl_nodropout_TFSEPNET_0.0b=32_orig_20_nomixup_lr[0.4, 0.0, 0.0, 0, 0.0, 0.0, 0.6]_"]) + str(trial.number), 
        "start_epoch": 1,
        "end_epoch": 400,
        'lr' : 1e-3,
        'mixup_alpha': trial.suggest_categorical('mixup_alpha', [0.0]),
        'mixup_prob': trial.suggest_categorical('mixup_prob', [0.0]),
        'optimizer': "AdamW",
        "loss": "CrossEntropyLoss",
        'metrics': {'MulticlassAccuracy': [10,1,'macro'], 'MulticlassConfusionMatrix': [10]},
        'device': "cuda",
        'model_file': 'model_classes.tfsepnet.py',
        "model_class": "TfSepNet",
        "early_stopping_patience": 200,
        "early_stopping_threshold": 0.01,
        "seed": 200,
        # Mel spectrogram parameters
        "sample_rate": 32000,
        "n_fft": 2048,
        "hop_length": 744,
        "n_mels": 128,
        "cache_transforms": False,
        # Create time-shifted spectrograms, only once.
        "create_timeshift": False,
        # Model parameters
        "dropout_rate": 0.4,
        "skip_cache_train": [0,1,2],
        "skip_cache_val": [0],
        "transform_probs": [0.4, 0.0, 0.0, 0, 0.0, 0.0, 0.6],
        #Knowledge distillation parameters
        "teacher_name": "TFSEPNET_less_relu_80_nomixup_lr[0.25, 0.25, 0.1, 0, 0.0, 0.2, 0.2]_0",
        "teacher_epoch": 199,
        "temperature": 5,
        "weight_teacher": 0.25,
        "teacher": True,
        # Print some class examples plots to tensorboard for ease of visualization
        "save_class_samples_tensorboard": False,

    }
    return trial_model_params


def get_model(params):
    """
    Imports the model from the model_file argument and returns it
    If you wish to use a different model, you can change the model_file argument
    or model_class argument
    """
    # Import the model (from model_path)
    model_file = params['model_file']
    # Import lib
    imp = importlib.import_module(model_file.replace('.py',''))
    model_class = params["model_class"]
    # Create the model
    device = params["device"]
    model = getattr(imp, model_class)
    import inspect
    # Extract the model parameters from the params dictionary
    model_params = {m_param: v for m_param, v 
                    in params.items() 
                        if m_param in [init_param.name 
                            for init_param 
                                in inspect.signature(model.__init__).parameters.values()]}
    model = model(**model_params)
    model.to(device)
    return model

def get_optimizer(model, params):
    optimizer_str = params['optimizer']
    lr_str = params['lr']
    optimizer = getattr(torch.optim, optimizer_str)(model.parameters(), lr=lr_str)
    return optimizer

def get_criterion(params):
    loss = params['loss']
    criterion = getattr(torch.nn, loss)()
    return criterion

def get_metrics(params):
    metrics_str = params['metrics']
    metrics = {metric : getattr(torchmetrics.classification, metric)(*metrics_str[metric]) for metric in metrics_str}
    return metrics

def get_scheduler(optimizer, params):
    # Return cosine annealing warm  scheduler
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=1e-6, last_epoch=-1)