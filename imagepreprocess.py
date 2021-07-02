# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 12:10:15 2020

@author: Bharathraj C L
"""




from torch.utils.data.dataset import Dataset
import cv2
import pytesseract
import textpreprocess
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import time
class ImageDataset(Dataset):

    # The init method is called when this class will be instantiated.
    def __init__(self, root,data):
        Images1,Images2,Text1,Text2, Y = [],[],[],[],[]
        folders = list(data.values)
        #folders = os.listdir(root)
        

        for count,folder in enumerate(folders):
            start = time.time()
            image1_path = root+'/'+folder[0]
            image2_path = root+'/'+folder[2]
            Images1.append(image1_path)
            Images2.append(image2_path)
            Text1.append(str(folder[1]))
            Text2.append(str(folder[3]))
            Y.append(str(folder[4]))
            '''
            image = cv2.imread(image1_path)
            #print(image1_path)
            #print(image)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pytesseract.image_to_string(rgb)
            results = textpreprocess.clean_text(results)
            Text1.append(results)
            image = cv2.imread(image2_path)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pytesseract.image_to_string(rgb)
            results = textpreprocess.clean_text(results)
            Text2.append(results)
            '''
            end = time.time()
            print(folder[4],end-start)
        data = [(x1_image,x1_text,x2_image,x2_text,y) for x1_image,x1_text,x2_image,x2_text,y in zip(Images1,Text1,Images2,Text2,Y)]
        self.data = data

    # The number of items in the dataset
    def __len__(self):
        return len(self.data)

    # The Dataloader is a generator that repeatedly calls the getitem method.
    # getitem is supposed to return (X, Y) for the specified index.
    def __getitem__(self, index):
        
        img1 = self.data[index][0]
        text1 = self.data[index][1]
        img2 = self.data[index][2]
        text2 = self.data[index][3]
        label = self.data[index][4]
        print(label)

        return (img1,text1,img2,text2,label)