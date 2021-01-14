import numpy as np
import torch as pt
import torchvision as tv

from shutil import copyfile

from pipeline import Dataset
from model import Model

if __name__ == '__main__':

    device = pt.device('cuda:0' if pt.cuda.is_available() else 'cpu')

    BATCH_SIZE = 64

    dataset = Dataset(BATCH_SIZE)

    model = Model(6)

    model = model.to(device)

    LR = 0.00005
    EPOCH = 50

    optimizer = pt.optim.Adam(model.parameters(), LR)
    criterion = pt.nn.CrossEntropyLoss()

    best_validation_loss = None

    for epoch in range(EPOCH):

        training_losses = []
        for train_x, train_y in dataset.next_train():

            train_x = train_x.to(device)
            train_y = train_y.to(device)

            prediction = model(train_x)

            loss = criterion(prediction, train_y)
            loss.backward()

            training_losses.append(loss.item())

            optimizer.step()

        training_loss = np.mean(training_losses)

        print(f'EPOCH {epoch} TRAINING LOSS: {training_loss}')

        validation_losses = []
        for valid_x, valid_y in dataset.next_valid():

            valid_x = valid_x.to(device)
            valid_y = valid_y.to(device)

            prediction = model(valid_x)

            loss = criterion(prediction, valid_y)

            validation_losses.append(loss.item())

        validation_loss = np.mean(validation_losses)

        if best_validation_loss is None:
            best_validation_loss = validation_loss
        
        if best_validation_loss > validation_loss:
            best_validation_loss = validation_loss
            pt.save(model, f'models/model_{validation_loss}.pt')

        print(f'EPOCH {epoch } VALIDATION LOSS: {validation_loss}')

    copyfile(f'models/model_{best_validation_loss}.pt', f'model.pt')
