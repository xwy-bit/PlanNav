import torch.nn as NN
from torchvision.models import resnet18,ResNet18_Weights,resnet34,ResNet34_Weights,resnet50,ResNet50_Weights
import torch
class PreProcess:
    def __init__(self,name):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if name == 'resnet34':
            self.weights = ResNet34_Weights.DEFAULT
            self.model = resnet34(weights = self.weights).cuda()
            self.transform = self.weights.transforms()
            
            # extract resnet34 feature
            self.model.fc = NN.Linear(512,512).cuda()
            NN.init.eye_(self.model.fc.weight)
            NN.init.zeros_(self.model.fc.bias)
            self.model.eval()
        elif name == 'resnet50':
            self.weights = ResNet50_Weights.DEFAULT
            self.model = resnet50(weights = self.weights).cuda()
            self.transform = self.weights.transforms()
            
            # extract resnet50 feature
            self.model.fc = NN.Linear(2048,2048).cuda()
            NN.init.eye_(self.model.fc.weight)
            NN.init.zeros_(self.model.fc.bias)
            self.model.eval()
        elif name == 'resnet18':
            self.weights = ResNet18_Weights.DEFAULT
            self.model = resnet18(weights = self.weights).cuda()
            self.transform = self.weights.transforms()
            
            # extract resnet18 feature
            self.model.fc = NN.Linear(512,512).cuda()
            NN.init.eye_(self.model.fc.weight)
            NN.init.zeros_(self.model.fc.bias)
            self.model.eval()
        else:
            raise TypeError('No Such Model')
    def trans(self,image):
        return self.transform(image).cuda()
    def go(self,images):
        transformed_images = self.trans(images).cuda()
        return self.model(transformed_images)
