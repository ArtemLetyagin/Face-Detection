import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch

class Comparator(nn.Module):
    def __init__(self):
        super(Comparator, self).__init__()
        self.transformation = transforms.Compose([transforms.Resize((50,50)),
                                    transforms.ToTensor()])
        vgg16 = models.vgg16(pretrained=True)
        
        self.vgg16 = nn.Sequential(*list(vgg16.features.children()))
        
        self.fc1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
        
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
        
            nn.Linear(256, 2)
        )      
        
    def forward_once(self, x):
        output = self.vgg16(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

    def similarity(self, image1, image2):
        image1 = self.transformation(image1).view(1,3,50,50)
        image2 = self.transformation(image2).view(1,3,50,50)
        output1 = self.forward_once(image1)
        output2 = self.forward_once(image2)
        return F.pairwise_distance(output1, output2).item()

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label)*torch.pow(euclidean_distance, 2)+label*torch.pow(torch.clamp(self.margin-euclidean_distance, min=0.0), 2))
        
        return loss_contrastive