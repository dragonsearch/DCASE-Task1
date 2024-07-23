from torchmetrics import Metric
import torch
from torch import Tensor


class DevAccuracy(Metric):
    def __init__(self, num_devices: int,**kwargs):
        super().__init__(**kwargs)
        self.num_devices = num_devices
        self.add_state("correct", default=torch.zeros((num_devices), dtype=torch.int64))
        self.add_state("total", default=torch.zeros((num_devices), dtype=torch.int64))
        self.add_state("accuracy", default=torch.zeros((num_devices), dtype=torch.float32))
    def update(self, preds, target, devices) -> None:
        """ Devices have to be encoded as integers"""
        #Devices tells us the index of the device to be incremented
        # The comparison preds == target gets us the index which should be checked for incrementing in devices

        # We get the index of the device to be incremented
        #Get max from preds
        preds = torch.argmax(preds, dim=1)
        # Print all devices

        device_correct_index = devices[preds == target].to(self.correct.device)
        # Increment those devices on the matrix, and increment total of each device
        #print('Device correct index', device_correct_index)
        device_correct_count = torch.bincount(device_correct_index, minlength=self.num_devices)
        #print('Device correct count', device_correct_count)
        # Sum the device count to each index
        self.correct += device_correct_count
        # Sum the total count to each index
        self.total += torch.bincount(devices, minlength=self.num_devices)
        #print('Correct', self.correct)
        #print('Total', self.total)
    def compute(self) -> Tensor:
        self.accuracy = self.correct.float() / self.total.float()
        #print(self.accuracy)
        self.accuracy[torch.isnan(self.accuracy)] = 0
        return self.accuracy
"""
if __name__ == '__main__':
    # Test the device accuracy
    preds = torch.tensor([0, 1, 1, 1, 0, 1])
    target = torch.tensor([1, 1, 1, 1, 1, 1])
    devices = torch.tensor([1, 1, 1, 2, 2, 1])
    acc = DevAccuracy(3)
    acc.update(preds, target, devices)
    print('Accuracy', acc.compute())
"""