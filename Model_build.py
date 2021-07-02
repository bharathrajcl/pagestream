# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 12:10:15 2020

@author: Bharathraj C L
"""

import os
from argparse import Namespace
from collections import Counter
import json
import re
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook,notebook

import glob
import pdf2image
from pdf2image import convert_from_path, convert_from_bytes

'''
pdf_path = 'C:/Users/Bharathraj C L/Desktop/dataset/New folder/*.pdf'
tr = pdf_path[51:-4]
pdf_list = glob.glob(pdf_path)
image_path = 'C:/Users/Bharathraj C L/Desktop/dataset/image1/'
cy = []
for c,i in enumerate(pdf_list):
    path = i[51:-4]
    image_list = convert_from_path(i)
    if(len(image_list) >= 5):
        for count,j in enumerate(image_list):
            #print(j)
            j.save(image_path+'/'+path+'-'+str(count+1)+'.jpg', 'JPEG')
            if(count >= 4):
                break
    #cy.append(count)
    print(c)
    
def find_relation(k,l):
    k = int(k)
    l = int(l)
    if(k > l):
        return k-l
    if(k < l):
        return k+l
    
rt = im_list[0][47:]
im_list = glob.glob(image_path+'*.jpg')
unique_list = list(set([x[47:60] for x in im_list]))

data = []
for j in unique_list:
    data_x = []
    temp_data = []
    for i in im_list:
        if(i[47:60] == j):
            temp_data.append(i)
    act_data = temp_data.copy()
    for k in act_data:
        for l in temp_data:
            if(k[61:62] != l[61:62]):
                data.append([k[47:],l[47:],int(l[61:62])])
                

po =  np.random.randint(100,size=(3,20,20))
ty = torch.from_numpy(po).view(1,3,20,20)
print(ty.shape)
convnet1 = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels=12, kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(2))
convnet1(ty.float()).shape

import pandas as pd
df = pd.DataFrame(data,columns = ['image1','image2','relation'])
df.to_csv('dataset.csv',index= False)

'''
 
class TextClassifier(nn.Module):
    
    def __init__(self, embedding_size, num_embeddings, num_channels, 
                 hidden_dim, num_classes, dropout_p, 
                 pretrained_embeddings=None, padding_idx=0):
        """
        Args:
            embedding_size (int): size of the embedding vectors
            num_embeddings (int): number of embedding vectors
            filter_width (int): width of the convolutional kernels
            num_channels (int): number of convolutional kernels per layer
            hidden_dim (int): the size of the hidden dimension
            num_classes (int): the number of classes in classification
            dropout_p (float): a dropout parameter 
            pretrained_embeddings (numpy.array): previously trained word embeddings
                default is None. If provided, 
            padding_idx (int): an index representing a null position
        """
        super(TextClassifier, self).__init__()

        if pretrained_embeddings is None:

            self.emb = nn.Embedding(embedding_dim=embedding_size,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx)
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.emb = nn.Embedding(embedding_dim=embedding_size,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx,
                                    _weight=pretrained_embeddings)
        
        self.bigru = nn.GRU(input_size= 600, hidden_size=400, num_layers=200, batch_first=False, bidirectional=True)
            
        self.convnet1 = nn.Sequential(nn.Conv1d(in_channels=embedding_size, out_channels=num_channels, kernel_size=3,stride=2),
            nn.ELU(),
            nn.MaxPool1d(2))
        self.convnet2 = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size, 
                   out_channels=num_channels, kernel_size=4,stride=3),
            nn.ELU(),
            nn.MaxPool1d(2))
        self.convent3 = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size, 
                   out_channels=num_channels, kernel_size=5,stride=4),
            nn.ELU(),
            nn.MaxPool1d(2))
        
      
    

        self._dropout_p = dropout_p
        self.fc1 = nn.Linear(num_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_in, apply_softmax=False):
        """The forward pass of the classifier
        
        Args:
            x_in (torch.Tensor): an input data tensor. 
                x_in.shape should be (batch, dataset._max_seq_length)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, num_classes)
        """
        
        # embed and permute so features are channels
        
        
        
        x_embedded = self.emb(x_in) #.permute(0, 2, 1)
        x_embedded = x_embedded.view(1,300,600)
        #print(x_embedded.shape,'1')
        x_embedded,_ = self.bigru(x_embedded)
        print(x_embedded.shape,'2')
        x_embedded = x_embedded.view(1,300, 600)
        #print(x_embedded.shape,'1')
        feature1 = self.convnet1(x_embedded).view(200,149)
        #print(features.shape,'2')
        
        #print(remaining_size,'extra')
        
        feature2 = self.convnet2(x_embedded).view(200,99)
        
        #print(remaining_size,'extra')
        
        feature3 = self.convent3(x_embedded).view(200,74)
        feature_data = torch.cat((feature1,feature2,feature3), dim= 1).to('cpu')
        #print(feature1.shape,feature2.shape,feature3.shape,feature_data.shape)
        feature_data = feature_data.view(1,200,322)
        # average and remove the extra dimension
        remaining_size = feature_data.size(dim=2)
        features = F.avg_pool1d(feature_data, remaining_size).squeeze(dim=2)
        #print(features.shape,'3')
        text_feature = F.dropout(features, p=self._dropout_p)
        #print(text_feature.shape,'4','text')
        # mlp classifier
        intermediate_vector = F.relu(F.dropout(self.fc1(text_feature), p=self._dropout_p))
        #print(intermediate_vector.shape,'5')
        #intermediate_vector = intermediate_vector
        prediction_vector = self.fc2(intermediate_vector)
        #print(prediction_vector.shape,'6')
        text_result = F.softmax(prediction_vector, dim=1)
        #print(text_result.shape,'7')

        return text_result,intermediate_vector




class ImageClassifier(nn.Module):
    
    def __init__(self,model_image):
        super(ImageClassifier, self).__init__()
        self.model = model_image
        #self.model2 = model_image
        #self.model3 = model_image
        self.convnet1 = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels=12, kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(3))
        self.convnet2 = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels=12, kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.AvgPool2d(3))
        self.convnet3 = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels=12, kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(3))
        self.convnet4 = nn.Sequential(
            nn.Conv2d(in_channels= 12, out_channels=6, kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(3),
            nn.Conv2d(in_channels= 6, out_channels=3, kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(3),
            nn.Conv2d(in_channels= 3, out_channels=2, kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Dropout())
            #nn.MaxPool2d(3))
        #self.model4 = 
        #self.num_feature = model_image.classifier[1].in_features
        #self.out_feature = model_image.classifier[1].out_features
        
        self.fc1 = nn.Linear(800,400)
        self.fc2 = nn.Linear(400,2)
    
    def forward(self,x):
        #print(x[0].shape,x[1].shape,x[2].shape)
        x1 = self.convnet1(x[0].view(1,3,150,600)).view(12,49,199)
        x2 = self.convnet2(x[1].view(1,3,300,600)).view(12,99,199)
        x3 = self.convnet3(x[2].view(1,3,150,600)).view(12,49,199)
        #print(x1.shape,x2.shape,x3.shape)
        x = torch.cat((x1,x2,x3),dim=1).view(1,12,197,199).to('cpu')
        #x = torch.cat((x1,x2,x3),dim=1).view(1,3,394,398).to('cpu')
        x = self.convnet4(x)
        #print(x.shape)
        #x = self.model(x)
        #x = x.view(3,100,10)
        #x = self.convnet(x)
        #print(x.shape)
        x = x.view(-1,800)
        #print(x.shape,'2')
        image_feature = F.relu(self.fc1(x))
        #print(image_feature.shape,'3')
        x = F.dropout(image_feature)
        image_result = F.softmax(self.fc2(x))
        #print(image_result.shape,'4','image')
        return image_result,image_feature


class TextImageClassifier(nn.Module):
    def __init__(self, embedding_size, num_embeddings, num_channels, 
                 hidden_dim, num_classes, dropout_p, pretrained_model,
                 pretrained_embeddings=None, padding_idx=0):
       
        super(TextImageClassifier, self).__init__()

        if pretrained_embeddings is None:

            self.emb_text = nn.Embedding(embedding_dim=embedding_size,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx)
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.emb_text = nn.Embedding(embedding_dim=embedding_size,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx,
                                    _weight=pretrained_embeddings)
        
        self.bigru_text = nn.GRU(input_size= 600, hidden_size=400, num_layers=200, batch_first=False, bidirectional=True)
            
        self.convnet1_text = nn.Sequential(nn.Conv1d(in_channels=embedding_size, out_channels=num_channels, kernel_size=3,stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.convnet2_text = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size, 
                   out_channels=num_channels, kernel_size=4,stride=3),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.convent3_text = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size, 
                   out_channels=num_channels, kernel_size=5,stride=4),
            nn.ReLU(),
            nn.MaxPool1d(2))
        
      
    

        self._dropout_p = dropout_p
        self.fc1_text = nn.Linear(num_channels, hidden_dim)
        self.fc2_text = nn.Linear(hidden_dim, num_classes)
        
        self.model = pretrained_model
        #self.model2 = model_image
        #self.model3 = model_image
        self.convnet1_image = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels=12, kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(3))
        self.convnet2_image= nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels=12, kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.AvgPool2d(3))
        self.convnet3_image = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels=12, kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(3))
        self.convnet4_image = nn.Sequential(
            nn.Conv2d(in_channels= 12, out_channels=6, kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(3),
            nn.Conv2d(in_channels= 6, out_channels=3, kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(3),
            nn.Conv2d(in_channels= 3, out_channels=2, kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Dropout())
            #nn.MaxPool2d(3))
        #self.model4 = 
        #self.num_feature = model_image.classifier[1].in_features
        #self.out_feature = model_image.classifier[1].out_features
        
        self.fc1_image = nn.Linear(800,400)
        self.fc2_image = nn.Linear(400,2)
        
        self.fc1_mix = nn.Linear(600,400)
        self.fc2_mix = nn.Linear(400,2)
    
    def forward(self, x_text, x_image):
        """The forward pass of the classifier
        
        Args:
            x_in (torch.Tensor): an input data tensor. 
                x_in.shape should be (batch, dataset._max_seq_length)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, num_classes)
        """
        
        # embed and permute so features are channels
        x_embedded = self.emb_text(x_text) #.permute(0, 2, 1)
        x_embedded = x_embedded.view(1,300,600)
        x_embedded,_ = self.bigru_text(x_embedded)
        #print(x_embedded.shape)
        #x_embedded = x_embedded.view(1,300, 600)
        feature1 = self.convnet1_text(x_embedded).view(200,199)
        feature2 = self.convnet2_text(x_embedded).view(200,133)
        feature3 = self.convent3_text(x_embedded).view(200,99)
        #print(feature1.shape,feature2.shape,feature3.shape)
        feature_data = torch.cat((feature1,feature2,feature3), dim= 1).to('cpu')
        feature_data = feature_data.view(1,200,431)
        remaining_size = feature_data.size(dim=2)
        features = F.avg_pool1d(feature_data, remaining_size).squeeze(dim=2)
        text_feature = F.dropout(features, p=self._dropout_p)
        #print(text_feature.shape)
        intermediate_vector = F.relu(F.dropout(self.fc1_text(text_feature), p=self._dropout_p))
        
        x1 = self.convnet1_image(x_image[0].view(1,3,150,600)).view(12,49,199)
        x2 = self.convnet2_image(x_image[1].view(1,3,300,600)).view(12,99,199)
        x3 = self.convnet3_image(x_image[2].view(1,3,150,600)).view(12,49,199)
        x = torch.cat((x1,x2,x3),dim=1).view(1,12,197,199).to('cpu')
        x = self.convnet4_image(x)
        x = x.view(-1,800)
        image_feature = F.relu(self.fc1_image(x))
        mix_data = torch.cat((intermediate_vector, image_feature), dim = 1).to('cpu')
        mix_data = mix_data.view(1,600)
        mix_feature = F.relu(self.fc1_mix(mix_data))
        mix_feature = F.dropout(mix_feature)
        mix_result = F.softmax(self.fc2_mix(mix_feature))
        
        return mix_result



class TextImageClassifier(nn.Module):
    def __init__(self, embedding_size, num_embeddings, num_channels, 
                 hidden_dim, num_classes, dropout_p, pretrained_model,
                 pretrained_embeddings=None, padding_idx=0):
       
        super(TextImageClassifier, self).__init__()

        if pretrained_embeddings is None:

            self.emb_text = nn.Embedding(embedding_dim=embedding_size,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx)
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.emb_text = nn.Embedding(embedding_dim=embedding_size,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx,
                                    _weight=pretrained_embeddings)
        
        self.bigru_text = nn.GRU(input_size= 600, hidden_size=400, num_layers=200, batch_first=False, bidirectional=True)
            
        self.convnet1_text = nn.Sequential(nn.Conv1d(in_channels=embedding_size, out_channels=num_channels, kernel_size=3,stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.convnet2_text = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size, 
                   out_channels=num_channels, kernel_size=4,stride=3),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.convent3_text = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size, 
                   out_channels=num_channels, kernel_size=5,stride=4),
            nn.ReLU(),
            nn.MaxPool1d(2))
        
      
    

        self._dropout_p = dropout_p
        self.fc1_text = nn.Linear(num_channels, hidden_dim)
        self.fc2_text = nn.Linear(hidden_dim, num_classes)
        
        self.model = pretrained_model
        #self.model2 = model_image
        #self.model3 = model_image
        self.convnet1_image = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels=12, kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(3))
        self.convnet2_image= nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels=12, kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.AvgPool2d(3))
        self.convnet3_image = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels=12, kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(3))
        self.convnet4_image = nn.Sequential(
            nn.Conv2d(in_channels= 12, out_channels=6, kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(3),
            nn.Conv2d(in_channels= 6, out_channels=3, kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(3),
            nn.Conv2d(in_channels= 3, out_channels=2, kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Dropout())
            #nn.MaxPool2d(3))
        #self.model4 = 
        #self.num_feature = model_image.classifier[1].in_features
        #self.out_feature = model_image.classifier[1].out_features
        
        self.fc1_image = nn.Linear(800,400)
        self.fc2_image = nn.Linear(400,2)
        
        self.fc1_mix = nn.Linear(600,400)
        self.fc2_mix = nn.Linear(400,2)
    
    def forward(self, x_text, x_image):
        """The forward pass of the classifier
        
        Args:
            x_in (torch.Tensor): an input data tensor. 
                x_in.shape should be (batch, dataset._max_seq_length)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, num_classes)
        """
        
        # embed and permute so features are channels
        x_embedded = self.emb_text(x_text) #.permute(0, 2, 1)
        x_embedded = x_embedded.view(1,300,600)
        x_embedded,_ = self.bigru_text(x_embedded)
        #print(x_embedded.shape)
        #x_embedded = x_embedded.view(1,300, 600)
        feature1 = self.convnet1_text(x_embedded).view(200,199)
        feature2 = self.convnet2_text(x_embedded).view(200,133)
        feature3 = self.convent3_text(x_embedded).view(200,99)
        #print(feature1.shape,feature2.shape,feature3.shape)
        feature_data = torch.cat((feature1,feature2,feature3), dim= 1).to('cpu')
        feature_data = feature_data.view(1,200,431)
        remaining_size = feature_data.size(dim=2)
        features = F.avg_pool1d(feature_data, remaining_size).squeeze(dim=2)
        text_feature = F.dropout(features, p=self._dropout_p)
        #print(text_feature.shape)
        intermediate_vector = F.relu(F.dropout(self.fc1_text(text_feature), p=self._dropout_p))
        
        x1 = self.convnet1_image(x_image[0].view(1,3,150,600)).view(12,49,199)
        x2 = self.convnet2_image(x_image[1].view(1,3,300,600)).view(12,99,199)
        x3 = self.convnet3_image(x_image[2].view(1,3,150,600)).view(12,49,199)
        x = torch.cat((x1,x2,x3),dim=1).view(1,12,197,199).to('cpu')
        x = self.convnet4_image(x)
        x = x.view(-1,800)
        image_feature = F.relu(self.fc1_image(x))
        mix_data = torch.cat((intermediate_vector, image_feature), dim = 1).to('cpu')
        mix_data = mix_data.view(1,600)
        mix_feature = F.relu(self.fc1_mix(mix_data))
        mix_feature = F.dropout(mix_feature)
        mix_result = F.softmax(self.fc2_mix(mix_feature))
        
        return mix_result





class TextImageClassifier_image1(nn.Module):
    def __init__(self, embedding_size, num_embeddings, num_channels, 
                 hidden_dim, num_classes, dropout_p, pretrained_model,
                 pretrained_embeddings=None, padding_idx=0):
       
        super(TextImageClassifier_image1, self).__init__()

        if pretrained_embeddings is None:

            self.emb_text = nn.Embedding(embedding_dim=embedding_size,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx)
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.emb_text = nn.Embedding(embedding_dim=embedding_size,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx,
                                    _weight=pretrained_embeddings)
        
        self.bigru_text = nn.GRU(input_size= 600, hidden_size=400, num_layers=200, batch_first=False, bidirectional=True)
            
        self.convnet1_text = nn.Sequential(nn.Conv1d(in_channels=embedding_size, out_channels=num_channels, kernel_size=3,stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.convnet2_text = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size, 
                   out_channels=num_channels, kernel_size=4,stride=3),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.convent3_text = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size, 
                   out_channels=num_channels, kernel_size=5,stride=4),
            nn.ReLU(),
            nn.MaxPool1d(2))
        
      
    

        self._dropout_p = dropout_p
        self.fc1_text = nn.Linear(num_channels, hidden_dim)
        self.fc2_text = nn.Linear(hidden_dim, num_classes)
        
        self.model = pretrained_model
        #self.model2 = model_image
        #self.model3 = model_image
        self.convnet1_image = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels=12, kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(3))
        self.convnet2_image= nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels=12, kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.AvgPool2d(3))
        self.convnet3_image = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels=12, kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(3))
        self.convnet4_image = nn.Sequential(
            nn.Conv2d(in_channels= 12, out_channels=6, kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(3),
            nn.Conv2d(in_channels= 6, out_channels=3, kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(3),
            nn.Conv2d(in_channels= 3, out_channels=2, kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Dropout())
            #nn.MaxPool2d(3))
        #self.model4 = 
        #self.num_feature = model_image.classifier[1].in_features
        #self.out_feature = model_image.classifier[1].out_features
        
        self.fc1_image = nn.Linear(800,400)
        self.fc2_image = nn.Linear(400,2)
        
        self.fc1_mix = nn.Linear(600,400)
        self.fc2_mix = nn.Linear(400,2)
    
    def forward(self, x_text, x_image):
        """The forward pass of the classifier
        
        Args:
            x_in (torch.Tensor): an input data tensor. 
                x_in.shape should be (batch, dataset._max_seq_length)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, num_classes)
        """
        
        # embed and permute so features are channels
        x_embedded = self.emb_text(x_text) #.permute(0, 2, 1)
        x_embedded = x_embedded.view(1,300,600)
        x_embedded,_ = self.bigru_text(x_embedded)
        #print(x_embedded.shape)
        #x_embedded = x_embedded.view(1,300, 600)
        feature1 = self.convnet1_text(x_embedded).view(200,199)
        feature2 = self.convnet2_text(x_embedded).view(200,133)
        feature3 = self.convent3_text(x_embedded).view(200,99)
        #print(feature1.shape,feature2.shape,feature3.shape)
        feature_data = torch.cat((feature1,feature2,feature3), dim= 1).to('cpu')
        feature_data = feature_data.view(1,200,431)
        remaining_size = feature_data.size(dim=2)
        features = F.avg_pool1d(feature_data, remaining_size).squeeze(dim=2)
        text_feature = F.dropout(features, p=self._dropout_p)
        #print(text_feature.shape)
        intermediate_vector = F.relu(F.dropout(self.fc1_text(text_feature), p=self._dropout_p))
        
        x1 = self.convnet1_image(x_image[0].view(1,3,150,600)).view(12,49,199)
        x2 = self.convnet2_image(x_image[1].view(1,3,300,600)).view(12,99,199)
        x3 = self.convnet3_image(x_image[2].view(1,3,150,600)).view(12,49,199)
        x = torch.cat((x1,x2,x3),dim=1).view(1,12,197,199).to('cpu')
        x = self.convnet4_image(x)
        x = x.view(-1,800)
        image_feature = F.relu(self.fc1_image(x))
        mix_data = torch.cat((intermediate_vector, image_feature), dim = 1).to('cpu')
        mix_data = mix_data.view(1,600)
        #mix_feature = F.relu(self.fc1_mix(mix_data))
        #mix_feature = F.dropout(mix_feature)
        #mix_result = F.softmax(self.fc2_mix(mix_feature))
        
        return mix_data
    

class TextImageClassifier_image2(nn.Module):
    def __init__(self, embedding_size, num_embeddings, num_channels, 
                 hidden_dim, num_classes, dropout_p, pretrained_model,
                 pretrained_embeddings=None, padding_idx=0):
       
        super(TextImageClassifier_image2, self).__init__()

        if pretrained_embeddings is None:

            self.emb_text = nn.Embedding(embedding_dim=embedding_size,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx)
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.emb_text = nn.Embedding(embedding_dim=embedding_size,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx,
                                    _weight=pretrained_embeddings)
        
        self.bigru_text = nn.GRU(input_size= 600, hidden_size=400, num_layers=200, batch_first=False, bidirectional=True)
            
        self.convnet1_text = nn.Sequential(nn.Conv1d(in_channels=embedding_size, out_channels=num_channels, kernel_size=3,stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.convnet2_text = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size, 
                   out_channels=num_channels, kernel_size=4,stride=3),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.convent3_text = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size, 
                   out_channels=num_channels, kernel_size=5,stride=4),
            nn.ReLU(),
            nn.MaxPool1d(2))
        
      
    

        self._dropout_p = dropout_p
        self.fc1_text = nn.Linear(num_channels, hidden_dim)
        self.fc2_text = nn.Linear(hidden_dim, num_classes)
        
        self.model = pretrained_model
        #self.model2 = model_image
        #self.model3 = model_image
        self.convnet1_image = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels=12, kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(3))
        self.convnet2_image= nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels=12, kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.AvgPool2d(3))
        self.convnet3_image = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels=12, kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(3))
        self.convnet4_image = nn.Sequential(
            nn.Conv2d(in_channels= 12, out_channels=6, kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(3),
            nn.Conv2d(in_channels= 6, out_channels=3, kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool2d(3),
            nn.Conv2d(in_channels= 3, out_channels=2, kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Dropout())
            #nn.MaxPool2d(3))
        #self.model4 = 
        #self.num_feature = model_image.classifier[1].in_features
        #self.out_feature = model_image.classifier[1].out_features
        
        self.fc1_image = nn.Linear(800,400)
        self.fc2_image = nn.Linear(400,2)
        
        self.fc1_mix = nn.Linear(600,400)
        self.fc2_mix = nn.Linear(400,2)
    
    def forward(self, x_text, x_image):
        """The forward pass of the classifier
        
        Args:
            x_in (torch.Tensor): an input data tensor. 
                x_in.shape should be (batch, dataset._max_seq_length)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, num_classes)
        """
        
        # embed and permute so features are channels
        x_embedded = self.emb_text(x_text) #.permute(0, 2, 1)
        x_embedded = x_embedded.view(1,300,600)
        x_embedded,_ = self.bigru_text(x_embedded)
        #print(x_embedded.shape)
        #x_embedded = x_embedded.view(1,300, 600)
        feature1 = self.convnet1_text(x_embedded).view(200,199)
        feature2 = self.convnet2_text(x_embedded).view(200,133)
        feature3 = self.convent3_text(x_embedded).view(200,99)
        #print(feature1.shape,feature2.shape,feature3.shape)
        feature_data = torch.cat((feature1,feature2,feature3), dim= 1).to('cpu')
        feature_data = feature_data.view(1,200,431)
        remaining_size = feature_data.size(dim=2)
        features = F.avg_pool1d(feature_data, remaining_size).squeeze(dim=2)
        text_feature = F.dropout(features, p=self._dropout_p)
        #print(text_feature.shape)
        intermediate_vector = F.relu(F.dropout(self.fc1_text(text_feature), p=self._dropout_p))
        
        x1 = self.convnet1_image(x_image[0].view(1,3,150,600)).view(12,49,199)
        x2 = self.convnet2_image(x_image[1].view(1,3,300,600)).view(12,99,199)
        x3 = self.convnet3_image(x_image[2].view(1,3,150,600)).view(12,49,199)
        x = torch.cat((x1,x2,x3),dim=1).view(1,12,197,199).to('cpu')
        x = self.convnet4_image(x)
        x = x.view(-1,800)
        image_feature = F.relu(self.fc1_image(x))
        mix_data = torch.cat((intermediate_vector, image_feature), dim = 1).to('cpu')
        mix_data = mix_data.view(1,600)
        #mix_feature = F.relu(self.fc1_mix(mix_data))
        #mix_feature = F.dropout(mix_feature)
        #mix_result = F.softmax(self.fc2_mix(mix_feature))
        
        return mix_data



    
class MixClassifier_continue(nn.Module):
    def __init__(self, embedding_size, num_embeddings, num_channels, 
                 hidden_dim, num_classes, dropout_p, pretrained_model,
                 pretrained_embeddings=None, padding_idx=0):
        super(MixClassifier_continue, self).__init__()
        #self.model = model_image
        #self.num_feature = model_image.classifier[1].in_features
        #self.out_feature = model_image.classifier[1].out_features
        self.image1_model = TextImageClassifier_image1(embedding_size, num_embeddings, num_channels, hidden_dim, num_classes, dropout_p, pretrained_model,pretrained_embeddings=None, padding_idx=0)
        self.image2_model = TextImageClassifier_image2(embedding_size, num_embeddings, num_channels, hidden_dim, num_classes, dropout_p, pretrained_model,pretrained_embeddings=None, padding_idx=0)
        self.convnet_image = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels=12, kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.AvgPool2d(3))
        
        self.fc1 = nn.Linear(432,320)
        self.fc2 = nn.Linear(320,128)
        self.fc3 = nn.Linear(128,5)
    
    def forward(self,x_image1,x_text1,x_image2,x_text2):
        #x = self.model(x)
        image1_vector = self.image1_model(x_text1,x_image1)
        image2_vector = self.image2_model(x_text2,x_image2)
        #x = x.view(-1,400)
        x = torch.cat((image1_vector,image2_vector),dim = 0).view(1,3,20,20)
        #po =  np.random.randint(100,size=(1,432))
        #x = torch.from_numpy(po).float().view(1,432)
        x = self.convnet_image(x).view(1,432) #need to complete
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.dropout(x)
        mix_result = F.softmax(self.fc3(x))
        return mix_result


    

    