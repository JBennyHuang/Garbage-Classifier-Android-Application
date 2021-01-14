import torch as pt
import numpy as np
import PIL
import os

from pipeline import Dataset, image_transforms
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    device = pt.device('cuda:0' if pt.cuda.is_available() else 'cpu')

    BATCH_SIZE = 32

    dataset = Dataset(BATCH_SIZE)

    model = pt.load(f'model.pt')
    model = model.eval()

    accuracies = []
    for test_x, test_y in dataset.next_test():

        test_x = test_x.to(device)
        test_y = test_y.to(device)

        prediction = model(test_x)

        accuracy = accuracy_score(pt.argmax(prediction, dim=-1).cpu().numpy(), test_y.cpu().numpy())
        accuracies.append(accuracy)

    print(np.mean(accuracies))