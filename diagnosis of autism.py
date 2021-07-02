import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data.dataset import Dataset
import cv2
from torch import Tensor
from torch.utils.data import DataLoader
"""
Loads the train/test set. 
Every image in the dataset is 28x28 pixels and the labels are numbered from 0-9
for A-J respectively.
Set root to point to the Train/Test folders.
"""

# Creating a sub class of torch.utils.data.dataset.Dataset
class notMNIST(Dataset):

    # The init method is called when this class will be instantiated.
    def __init__(self, root):
        Images, Y = [], []
        folders = os.listdir(root)

        for folder in folders:
            folder_path = root+'/'+folder
            
            for ims in os.listdir(folder_path):
                img_path = folder_path +'/' +ims
                try:
            
                    Images.append(np.array(cv2.imread(img_path)))
                    
                    if(folder[0] == 'a'):
                        Y.append(0)  # Folders are A-J so labels will be 0-9
                    else:
                        Y.append(1)
                    
                except:
                    # Some images in the dataset are damaged
                    #print()
                    pass
                    #print("File {}/{} is broken".format(folder, ims))
        data = [(x, y) for x, y in zip(Images, Y)]
        self.data = data

    # The number of items in the dataset
    def __len__(self):
        return len(self.data)

    # The Dataloader is a generator that repeatedly calls the getitem method.
    # getitem is supposed to return (X, Y) for the specified index.
    def __getitem__(self, index):
        img = self.data[index][0]
        #print(img.shape)
        img = cv2.resize(img,(128,128))
        #print(img.shape)

        # 8 bit images. Scale between [0,1]. This helps speed up our training
        img = img / 255.0
        # Input for Conv2D should be Channels x Height x Width
        img_tensor = Tensor(img).view(3, 128, 128).float()
        #img_tensor = Tensor(img).float()
        label = self.data[index][1]
        return (img_tensor, label)

class Model(nn.Module):
    
    def __init__(self,model):
        super(Model, self).__init__()
        self.model = model
        self.num_feature = model.classifier[1].in_features
        self.out_feature = model.classifier[1].out_features
        
        self.fc1 = nn.Linear(self.out_feature,128)
        self.fc2 = nn.Linear(128,2)
    
    def forward(self,x):
        x = self.model(x)
        #x = x.view(-1,1280)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.softmax(self.fc2(x))
        return x
    

# Instantiating the notMNIST dataset class we created
train_dataset = notMNIST('C:/Users/Bharathraj C L/Downloads/447704_1090147_bundle_archive/train')
print("Loaded data")

# Creating a dataloader
train_loader = DataLoader(dataset=train_dataset,batch_size = 64 ,shuffle=True)

# Instantiating the model, loss function and optimizer
model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
#model = models.MobileNetV2()
net = Model(model)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

    
    

loss_history = []


def train(epoch):
    epoch_loss = 0
    n_batches = len(train_dataset) // 80
    print('trining data')

    for step, data in enumerate(train_loader, 0):
        train_x, train_y = data
        #print(train_x.shape)
        y_hat = net(train_x)
        #print(y_hat)
        train_y = torch.Tensor(np.array(train_y))
        #print(train_y)

        # CrossEntropyLoss requires arg2 to be torch.LongTensor
        loss = criterion(y_hat, train_y.long())
        epoch_loss += loss.item()
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()
        optimizer.step()
        # There are len(dataset)/BATCH_SIZE batches.
        # We print the epoch loss when we reach the last batch.
        if step % n_batches == 0 and step != 0:
            epoch_loss = epoch_loss / n_batches
            loss_history.append(epoch_loss)
            print("Epoch {}, loss {}".format(epoch, epoch_loss))
            epoch_loss = 0

def predict_class(model,image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img,(64,64))
    img = img / 255.0
    img_tensor = Tensor(img).view(1,3, 64, 64).float()
    
    result = model(img_tensor)
    return result


'''
for epoch in range(5):
    print(epoch)
    for i, data in enumerate(train_loader):
        inputs, labels = data
        #inputs = Variable(inputs).cuda()
        #labels = Variable(labels).cuda()
        
        # forward + backward + optimize
        
        # zeroes the gradient buffers of all parameters
        optimizer.zero_grad()
        #forward pass
        outputs = model(inputs)
        # calculate the loss
        print(inputs.shape)
        print(outputs.shape)
        print(labels.shape)
        loss = criterion(outputs, labels)
        # backpropagation
        loss.backward()
        # Does the update after calculating the gradients
        optimizer.step()
        break
    print(loss)
    break
        #if (i+1) % 5 == 0: # print every 100 mini-batches
        #    print(epoch, i+1, loss.data[0])



for epoch in range(100):
    print(epoch)
    for i, data in enumerate(train_loader):
        inputs, labels = data
        #inputs = Variable(inputs).cuda()
        #labels = Variable(labels).cuda()
        
        # forward + backward + optimize
        
        # zeroes the gradient buffers of all parameters
        optimizer.zero_grad()
        #forward pass
        outputs = net(inputs)
        # calculate the loss
        #print(inputs.shape)
        #print(outputs.shape)
        loss = criterion(outputs, labels)
        # backpropagation
        loss.backward()
        # Does the update after calculating the gradients
        optimizer.step()
        break
    print(loss)
    break
        #if (i+1) % 5 == 0: # print every 100 mini-batches
        #    print(epoch, i+1, loss.data[0])
'''

for i in range(0,50):
    train(i)
    

    
image_path = 'C:/Users/Bharathraj C L/Downloads/447704_1090147_bundle_archive/test/autistic/056.jpg'
print(predict_class(net,image_path))
