import torch as pt
import torchvision as tv

class Model(pt.nn.Module):
    def __init__(self, n_classes):
        super(Model, self).__init__()
        self.model = tv.models.resnet50(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = pt.nn.Sequential(
            pt.nn.Linear(self.model.fc.in_features, 256),
            pt.nn.ReLU(),
            pt.nn.Linear(256, 256),
            pt.nn.ReLU(),
            pt.nn.Linear(256, n_classes),
            pt.nn.Sigmoid()
        ) 

    def forward(self, x):
        return self.model(x)

