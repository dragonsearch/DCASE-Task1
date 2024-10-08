import torch as torch
import numpy as np
import pandas as pd
import pathlib
from tools.utils import load_ckpt
from tools.utils import save_obj, load_ckpt


class Evaluator():
    """
    Class for evaluating the model on the test set.
    This class assumes that the model is already trained and that we don't the truth labels.
    Output is a dictionary with the probabilities of each class for each sample in the format:
    {filename: [0.2, ..., n_classes], filename: [0.2, ..., n_classes], ...}

    Requires that the test_loader has returns the filename of each sample on the getitem method
    in the following format: (sample, label, filename)

    TODO: Testing
    """
    def __init__(self, params) -> None:
        """
        Args: 
            model: trained model
            test_loader: test loader
            name: name of the experiment
        
        Parameters:
            device: device used for training
            n_total_steps_test: number of batches in the test loader
            predictions: dictionary with the predictions in the format 
                {filename: [probA, probB ..., n_classes], ...}
        """
        self.model = params["model"]
        self.load_model(params["eval_file"])
        self.eval_loader = params["eval_loader"]
        self.label_encoder = params["label_encoder"]
        self.name = params["name"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_total_steps_test = len(self.eval_loader)
        self.predictions = {} 

    def save_preds(self):
        pathlib.Path("./models/" + self.name + "/preds/obj").mkdir(parents=True, exist_ok=True)
        save_obj(self.predictions, "./models/" + self.name +"/preds/obj/preds_dict" + "_" + str(self.name))

    def save_csv(self):
        columns = ['filename', 'scene_label', 'airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']
        # Put the max probability in the scene_label column
        for filename, probs in self.predictions.items():
            decoded_max = self.label_encoder.inverse_transform([np.argmax(probs)])[0]
            self.predictions[filename] = [filename, decoded_max] + list(probs)
        df = pd.DataFrame.from_dict(self.predictions, orient='index', columns=columns)
        df.to_csv("./models/" + self.name + "/preds/" + str(self.name) + ".csv", float_format='%.2f', index=False)

    def load_model(self, path):
        dummy_optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model, _ = load_ckpt(self.model, dummy_optimizer, "models/" + path)

    def eval_batches(self):
        for i, (samples, labels, filenames, identifiers) in enumerate(self.eval_loader):
            samples = samples.to(self.device)
            y_pred = self.model(samples)
            self.predictions.update({filename: y_pred[j].cpu().detach().numpy() for j, filename in enumerate(filenames)})
            if i % 100 == 0:
                print(f'Step [{i}/{self.n_total_steps_test}]')

    def eval(self):
        with torch.no_grad():
            self.model.eval()
            self.eval_batches()
            self.save_preds()
            self.save_csv()
            self.model.train()
        print("Evaluation done")




