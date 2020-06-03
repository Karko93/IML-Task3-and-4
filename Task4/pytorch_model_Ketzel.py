#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch.utils.data as datatorch
import torch.nn as nn
import pandas as pd
import torch
from PIL import Image
import cv2
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# In[2]:


train_triplets = np.loadtxt('train_triplets.txt', dtype= 'str')
test_triplets = np.loadtxt('test_triplets.txt', dtype= 'str')


# In[3]:


print(test_triplets.shape)


# In[4]:


print(train_triplets.shape)
#print()
train_triplets , val_triplets = train_test_split(train_triplets, test_size = 0.1)
print(train_triplets.shape, val_triplets.shape)
half_index = np.int64((val_triplets.shape[0]-val_triplets.shape[0]%2)/2)
print(half_index)
val_labels = np.int64(np.ones((val_triplets.shape[0],)))
print(val_labels.shape)
val_triplets[half_index:, 0], val_triplets[half_index:, 1] = val_triplets[half_index:, 1], val_triplets[half_index:, 0].copy()
val_labels[half_index:] = np.int64(0)


# In[ ]:





# In[5]:


train_dir = 'food/food'
train_files = os.listdir(train_dir)
test_files = os.listdir(train_dir)


class ImageTriplesSet(Dataset):
    def __init__(self , file_array, dir, mode='train', transform = None,labels =None):
        self.triple_list = list(map(tuple, file_array))
        self.mode = mode
        self.labels = labels
        self.dir = dir
        self.transform = transform
        
    def __len__(self):
        return len(self.triple_list)
    
    def __getitem__(self,idx):
        img1 = Image.open(os.path.join(self.dir, self.triple_list[idx][0] + '.jpg'))
        img2 = Image.open(os.path.join(self.dir, self.triple_list[idx][1] + '.jpg'))
        img3 = Image.open(os.path.join(self.dir, self.triple_list[idx][2] + '.jpg'))
        
        
        if self.transform is not None:
            img1 = self.transform(img1).numpy()
            img2 = self.transform(img2).numpy()
            img3 = self.transform(img3).numpy()
        if self.labels is None:
            return img1, img2, img3
        else:
            return img1, img2, img3, self.labels[idx]
            
        #concat_img = cv2.hconcat([img1, img2, img3]).astype('float32')
        #if self.mode == 'train':
            #label = self.labels[idx]
            #return concat_img , label
            
        #else:
            #return concat_img, int(self.triple_list[idx][:-4])
        
#data_transform = transforms.Compose([
  #  transforms.Resize(350,240),
  #  transforms.CenterCrop(240),
  #  transforms.ToTensor()
#])

data_transform = transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

train_dataset = ImageTriplesSet(train_triplets, train_dir, transform = data_transform, labels = None)
val_dataset = ImageTriplesSet(val_triplets, train_dir, transform= data_transform, labels = None)
test_dataset = ImageTriplesSet(test_triplets, train_dir, mode="test" ,transform = data_transform,labels = None)


# In[6]:


model = torch.hub.load('pytorch/vision', 'resnet34', pretrained=False)


# In[7]:


learning_rate = 0.001
batch_size = 64
epochs = 3
logstep = int(1000 // batch_size)

train_loader = datatorch.DataLoader(dataset=train_dataset, 
                         shuffle=True, 
                         batch_size=batch_size)

#test_loader = datatorch.DataLoader(dataset=test_dataset, shuffle = False, batch_size= batch_size)



#model.fc = nn.Sequential(nn.Linear(model.fc.in_features,512),
                                  #nn.ReLU(),
                                  #nn.Dropout(),
                                  #nn.Linear(512, 2))
            
model.fc = nn.Sequential(nn.Linear(model.fc.in_features,512),
                                  #nn.ReLU(),
                                  
                                  nn.Linear(512, 2048))

#net = TripletNet(resnet101())
           
            

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda:0")
model =model.to(device)
#net = torch.nn.DataParallel(net).cuda()
#cudnn.benchmark = True
 #create optimizer
criterion = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=1e-5,nesterov=True)

#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=10,verbose=True)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,1000,1500], gamma=0.5)

training_loss_vec = []
training_accuracy_vec = []
val_f1_score = []
    
#criterion = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,1000,1500], gamma=0.5)

    
    
    

# loop over epochs
model.train()
for e in range(epochs):
    training_loss = 0.
    training_accuracy = 0.
    for idx, (data1, data2, data3) in enumerate(train_loader):
    #for idx, (img,label) in enumerate(train_loader):
        data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        #img, label = img.cuda(), label.cuda()
        #embedded_a, embedded_p, embedded_n = net(data1, data2, data3)
        embedded_a, embedded_p, embedded_n = model(data1), model(data2), model(data3)
        loss = criterion(embedded_a, embedded_p, embedded_n)
        # call optimizer.zero_grad()
        optimizer.zero_grad()
        # compute predictions using model
        #y_pred =  model(img)
        # compute loss
        
        #loss = criterion(y_pred,label)
        # run backward method
        loss.backward()
        # run optimizer step
        optimizer.step()
        #scheduler.step()  ######
        # logging (optional)
        
        training_loss += loss.item()
        #y_pred_idx = torch.max(y_pred.detach().cpu(),dim=1)[1]
        #training_accuracy += torch.mean((y_pred_idx == label.cpu()).float()).item()
        if (idx+1) % logstep == 0: 
            training_loss_vec.append(training_loss/logstep)
            #training_accuracy_vec.append(training_accuracy/logstep)
            print('training loss: ', training_loss/logstep)
            training_loss, training_accuracy = 0.,0.
   




# In[8]:


val_loader = datatorch.DataLoader(dataset=val_dataset, shuffle = False, batch_size= 1)

val_labels_pred = []
model.eval()
for idx, (data1, data2, data3) in enumerate(val_loader):
    data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
    embedded_1, embedded_2, embedded_3 = model(data1), model(data2), model(data3)
    if torch.dist(embedded_1,embedded_3,2)>=torch.dist(embedded_1,embedded_2,2):
        val_labels_pred.append(1)
    else:
        val_labels_pred.append(0)

f1 = f1_score(val_labels_pred, val_labels)
print(f1)
# 0.5891517599538373


# In[9]:


#for i in range(0, len(val_labels_pred)): 
    #val_labels_pred[i] = int(val_labels_pred[i]) 
#f1 = f1_score(val_labels_pred, val_labels)

print(len(val_labels_pred,))


# In[10]:


test_loader = datatorch.DataLoader(dataset=test_dataset, shuffle = False, batch_size= 1)


# In[11]:


test_triplets_pred = []
model.eval()
for idx, (data1, data2, data3) in enumerate(test_loader):
    data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
    embedded_1, embedded_2, embedded_3 = model(data1), model(data2), model(data3)
    if torch.dist(embedded_1,embedded_3,2)>=torch.dist(embedded_1,embedded_2,2):
        test_triplets_pred.append(str(1))
    else:
        test_triplets_pred.append(str(0))
    


# In[12]:


print(len(test_triplets_pred))
print(str(1))


# In[13]:


with open('submission_Ketzel.txt', 'w') as f:
    for item in test_triplets_pred:
        f.write(item + '\n')


# In[ ]:




