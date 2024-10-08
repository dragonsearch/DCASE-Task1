import torch
import numpy as np
import pickle


def save_ckpt(model, optimizer, scheduler, PATH, params={}):
    """
    Args:
        model (torch.nn.Module): Model to be saved
        optimizer (torch.optim.Optimizer): Optimizer to be saved
        PATH (str): Path where the model will be saved
        params (dict): Dictionary with the parameters to be saved     
    """
    #Join 2 dicts using | (python 3.9) (for python 3.5+ {**x, **y})
    save_dict = {'model_state_dict':model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() 
                if scheduler is not None else None,
                }# | params
    torch.save(save_dict, PATH)
    print(f'Saved model (ckpt) into: {PATH}')

def load_ckpt(model,optimizer, scheduler, PATH):
    """
    Args:
        model (torch.nn.Module): Model to be loaded
        optimizer (torch.optim.Optimizer): Optimizer to be loaded
        mode (str): Mode of the model (train or eval)
        PATH (str): Path where the model is saved
    """
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    print(f'Loaded model (ckpt) from: {PATH}')
    return model, optimizer

def dict_to_txt(dictionary, PATH):
    """
    Converts a dictionary to a string and saves it to newly created a txt file
    """
    with open(PATH, 'w') as f:
        print(dictionary, file=f)
    print(f'Saved dictionary into: {PATH} as a txt file')



def count_and_print_least_common_classes(arr,idx2class):
    # Calculate counts of each class
    unique_classes, counts = np.unique(arr, return_counts=True)
    # Sort the classes based on counts in ascending order
    sorted_indices = np.argsort(counts)
    # Get the least 10 common classes
    least_common_classes = unique_classes[sorted_indices[:30]]
    least_counts = counts[sorted_indices[:30]]
    # Apply the idx function to change the class names
    least_common_classes = np.array(list(map(lambda x: idx2class[x], least_common_classes)))
    # Print the least 10 common classes
    print("Least 10 common classes after applying idx function:")
    i = 0
    for class_name in least_common_classes:
        print("Class: ", class_name, " Count: ", least_counts[i])
        i+=1
    return least_common_classes, least_counts



def wrong_class(y_pred, labels, samples):
    wrong_predictions = []
    predicted_indices = torch.argmax(y_pred, dim=1)
    comparison = predicted_indices != labels
    different_elements = torch.nonzero(comparison).squeeze()
    if different_elements.shape[0] != 0:
        print("found wrong elements")
        for element in different_elements:
            wrong_predictions.append({'samples': samples[element], 'label': labels[element], 'prediction': y_pred[element]})


def save_obj(obj, name):
    """
    Saves a pickle object to disk. It adds .pkl extension
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print(f"Saved {name} to disk")


def load_obj(name):
    """
    Loads a pickle object from disk. It assumes .pkl extension
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)