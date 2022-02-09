#import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
#from sklearn.metrics import confusion_matrix

# transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# datasets
trainset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=True,
    transform=transform)
testset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=False,
    transform=transform)

# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                        shuffle=True, num_workers=2)


testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                        shuffle=False, num_workers=2)

# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# helper function to show an image
# (used in the `plot_classes_preds` function below)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
if __name__ == '__main__':
                

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    num_epoch=15   

    running_loss = 0.0
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        print('epoch', epoch+1)   
       

        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
     
          
        
    print('Finished Training')
    
    
    def evaluation(dataloader):
     total, correct,target_true = 0, 0, 0
     #keeping the network in evaluation mode 
     
     nb_classes = 10

     confusion_matrix = torch.zeros(nb_classes, nb_classes)
     
     net.eval()
     y_pred_list = []
     for data in dataloader:
         inputs, labels = data
         #moving the inputs and labels to gpu
         outputs = net(inputs)
         _, pred = torch.max(outputs.data, 1)
         total += labels.size(0)
         correct += (pred == labels).sum().item()
         
        # predicted_true += torch.sum(predicted_classes).float()
        # correct_true += torch.sum(
        #predicted_classes == target_classes * predicted_classes == 0).float()
        
         for t, p in zip(labels.view(-1), pred.view(-1)):
               confusion_matrix[t.long(), p.long()] += 1
         
     confusion_matrix_df = pd.DataFrame(confusion_matrix.numpy())
     
     tp=confusion_matrix_df.iloc[0,0]
     fp=confusion_matrix_df.iloc[0,:].sum()-tp
     fn=confusion_matrix_df.iloc[:,0].sum()-tp
     tn=confusion_matrix_df.sum().sum()-tp-fp-fn
     acuracy=(tp+tn)/confusion_matrix_df.sum().sum()
     precision=tp/(tp+fp)
     recall=tp/(tp+fn)
     f1_score=(2*precision*recall)/(precision+recall)
     
     print('acuracy ',acuracy)
     print('precision ',precision)
     print('Recall ',recall)
     print('F1_Score ',f1_score)
     print(confusion_matrix_df) 
        
     #return 100 * correct / total
 
    evaluation(testloader)
    print('Finished Teste')