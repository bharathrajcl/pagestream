# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 13:43:57 2020

@author: Bharathraj C L
"""


import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import cv2
from torch import Tensor
from torch.utils.data import DataLoader
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from imagepreprocess import ImageDataset
from textpreprocess import TextDataset
import textpreprocess
from tqdm import notebook
model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11', pretrained=True)
from argparse import Namespace



args = Namespace(
    # Data and Path hyper parameters
    news_csv="dataset.csv",
    news_csv_act = "dataset_processed.csv",
    vectorizer_file="vectorizer.json",
    model_state_file="model.pth",
    save_dir="model_storage/ch5/document_classification",
    # Model hyper parameters
    glove_filepath='C:/Users/Bharathraj C L/Downloads/glove.6B/glove.6B.300d.txt', 
    use_glove=True,
    embedding_size=300, 
    hidden_dim= 200, 
    num_channels=200, 
    # Training hyper parameter
    seed=1337, 
    learning_rate=0.001, 
    dropout_p=0.1, 
    batch_size=1, 
    num_epochs=10, 
    early_stopping_criteria=5, 
    # Runtime option
    cuda=True, 
    catch_keyboard_interrupt=True, 
    reload_from_files=False,
    expand_filepaths_to_save_dir=True,
    device = 'cpu',
    image_root='C:/Users/Bharathraj C L/Downloads/447704_1090147_bundle_archive/image1'
) 
args.cuda = False if not torch.cuda.is_available() else True
args.device = torch.device("cuda" if args.cuda else "cpu")
args.device = torch.device("cuda" if args.cuda else "cpu")
if not torch.cuda.is_available():
    args.cuda = False

def create_vectorize_embedding(image_root,dataset_path):
    
    data = pd.read_csv(dataset_path)  
    custom_dataset = ImageDataset(image_root,data)
    custom_loader = DataLoader(dataset = custom_dataset ,shuffle=True)
    dataset_act = TextDataset.load_dataset_and_make_vectorizer(args.news_csv_act)
    dataset_act.save_vectorizer(args.vectorizer_file)
    vectorizer_act = dataset_act.get_vectorizer()
    #dataset_act._max_seq_length
    
    
    # Use GloVe or randomly initialized embeddings
    if args.use_glove:
        words = vectorizer_act.title_vocab._token_to_idx.keys()
        embeddings_act = textpreprocess.make_embedding_matrix(glove_filepath=args.glove_filepath, 
                                           words=words)
        print("Using pre-trained embeddings")
    else:
        print("Not using pre-trained embeddings")
        embeddings_act = None
    return vectorizer_act,embeddings_act,custom_loader
        

vectorizer_act,embeddings_act,custom_loader = create_vectorize_embedding(args.image_root,args.news_csv_act)



'''
ut = list(data.values)
op = []
for i in ut:
    op.append(i[0])
op = list(set(op))

text_path = 'C:/Users/Bharathraj C L/Downloads/447704_1090147_bundle_archive/image1/'
da = {}
import time
ti= 0
for c,i in enumerate(op):
    start = time.time()
    image = cv2.imread(text_path+i)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pytesseract.image_to_string(rgb)
    results = textpreprocess.clean_text(results)
    da[i] = results
    end = time.time()
    ti = end-start  + ti
    print(c,ti/(c+1))


at = []
for i in ut:
    at.append([i[0],da[i[0]],i[1],da[i[1]],i[2]])
dy = pd.DataFrame(at,columns = ['image1','text1','image2','text2','label'])
dy.to_csv('dataset_processed.csv',index = False)
    



#cv2.imread('C:/Users/Bharathraj C L/Downloads/447704_1090147_bundle_archive/image1/GS-1966-02-08-1.jpg')

print("Loaded data")

data = pd.DataFrame('dataset_processed.csv')
'''

# Creating a dataloader

#cus = []
#for i in custom_loader:
#    cus.append(i)

#data1 = pd.DataFrame(cus,columns=['image1','text1','image2','text2','target'])

    
temp = []
'''
if args.expand_filepaths_to_save_dir:
    args.vectorizer_file = os.path.join(args.save_dir,
                                        args.vectorizer_file)

    args.model_state_file = os.path.join(args.save_dir,
                                         args.model_state_file)
    
    print("Expanded filepaths: ")
    print("\t{}".format(args.vectorizer_file))
    print("\t{}".format(args.model_state_file))
'''  





from Model_build import MixClassifier_continue
#TextImageClassifier
classifier_mix = MixClassifier_continue(embedding_size=args.embedding_size, 
                            num_embeddings=len(vectorizer_act.title_vocab),
                            num_channels=args.num_channels,
                            hidden_dim=args.hidden_dim, 
                            num_classes= 5, #len(vectorizer.category_vocab), 
                            dropout_p=args.dropout_p,
                            pretrained_model = model,
                            pretrained_embeddings=embeddings_act,
                            padding_idx=5)

loss_mix = torch.nn.CrossEntropyLoss()
optimizer_mix = torch.optim.Adam(classifier_mix.parameters(),lr = 0.001)
scheduler_mix = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_mix,
                                           mode='min', factor=0.5,
                                           patience=1)


epoch_bar = notebook.tqdm(desc='training routine', 
                          total=args.num_epochs,
                          position=0)




def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def train(args,classifier_mix,custom_loader):
    running_acc = 0
    running_loss = 0
    classifier_mix.train()
    for epoch_index in range(args.num_epochs):
        for step,data in enumerate(custom_loader,0):
            image1,text1,image2,text2,label = data
            dataset1 = vectorizer_act.vectorize(text1[0])
            if(len(dataset1) < 600):
                temp_var = np.zeros(600-len(dataset1))
                dataset1 = np.concatenate((dataset1,temp_var),axis = 0)
            elif(len(dataset1) > 600):
                dataset1 = dataset1[:600]
                
            dataset2 = vectorizer_act.vectorize(text2[0])
            if(len(dataset2) < 600):
                temp_var = np.zeros(600-len(dataset2))
                dataset2 = np.concatenate((dataset2,temp_var),axis = 0)
            elif(len(dataset2) > 600):
                dataset2 = dataset2[:600]
            #print(type(dataset1))
            #print(type(dataset2))
            image_data = cv2.imread(image1[0])
            image_data= cv2.resize(image_data,(600,600))
            image_data = image_data / 255.0
            list_image1 = [Tensor(image_data[0:150,:]),Tensor(image_data[150:450,:]),Tensor(image_data[450:,:])]
            
            image_data = cv2.imread(image2[0])
            image_data= cv2.resize(image_data,(600,600))
            image_data = image_data / 255.0
            list_image2 = [Tensor(image_data[0:150,:]),Tensor(image_data[150:450,:]),Tensor(image_data[450:,:])]
            
            
            optimizer_mix.zero_grad()
            
            result= classifier_mix(list_image1,torch.from_numpy(dataset1).to(args.device).long(),list_image2,torch.from_numpy(dataset2).to(args.device).long())
            #print(result,'1')
            label = torch.from_numpy(np.array(int(label[0])-1)).long().view(1)
            #print(label.shape,'2')
            loss_data = loss_mix(result,label)
            loss_t = loss_data.item()
            running_loss += (loss_t - running_loss) / (step + 1)
            loss_data.backward()
            optimizer_mix.step()
            acc = compute_accuracy(result, label)
            running_acc += (acc - running_acc) / (step + 1)
            print('steps :',step,'loss text:',loss_data.item())#,'loss image:',loss_image_data.item(),'loss mix:',loss_mix_data.item())
            
        print('epoch:',epoch_index,'loss:',running_loss)
        if(epoch_index%2 == 0):
            torch.save(model, args.model_state_file)
    
    return model
        
        
    
