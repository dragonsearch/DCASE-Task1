from torchmetrics import Metric
import torch
from torch import Tensor
from torchmetrics.classification import MulticlassConfusionMatrix
import numpy as np
class CityAccuracy(Metric):
    def __init__(self, num_cities: int,**kwargs):
        super().__init__(**kwargs)
        self.num_cities = num_cities
        self.add_state("correct", default=torch.zeros((num_cities), dtype=torch.int64))
        self.add_state("total", default=torch.zeros((num_cities), dtype=torch.int64))
        self.add_state("accuracy", default=torch.zeros((num_cities), dtype=torch.float32))
    def update(self, preds, target, cities) -> None:
        """ Cities have to be encoded as integers"""
        #Cities tells us the index of the city to be incremented
        # The comparison preds == target gets us the index which should be checked for incrementing in cities

        # We get the index of the city to be incremented
        #Get max from preds
        preds = torch.argmax(preds, dim=1)
        # Print all cities

        city_correct_index = cities[preds == target].to(self.correct.device)
        # Increment those cities on the matrix, and increment total of each city
        #print('City correct index', city_correct_index)
        city_correct_count = torch.bincount(city_correct_index, minlength=self.num_cities)
        #print('City correct count', city_correct_count)
        # Sum the city count to each index
        self.correct += city_correct_count
        # Sum the total count to each index
        self.total += torch.bincount(cities, minlength=self.num_cities)
        #print('Correct', self.correct)
        #print('Total', self.total)
    def compute(self) -> Tensor:
        self.accuracy = self.correct.float() / self.total.float()
        #print(self.accuracy)
        self.accuracy[torch.isnan(self.accuracy)] = 0
        return self.accuracy