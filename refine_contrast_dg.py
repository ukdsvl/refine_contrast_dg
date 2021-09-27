#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models
import torch.nn as nn
import copy
import time
import random

import glob
import seaborn as sns

import torch.optim as optim

import math
from torch.nn import functional as F

from PIL import Image


# In[ ]:

# import torch
torch.manual_seed(0)
# import random
random.seed(0)
# import numpy as np
np.random.seed(0)


# In[70]:


## General hyperparameters
batch_size = 32 # mini-batch size used for training
num_epochs = 30 # no of training epochs (one epoch = one pass over all training domains, across all classes)

fdim=2048  # whatever is the output of the base encoder, for eg, 2048 for Res-50

## Hyperparameters for contrastive learning (performs decently well)
queue_sz=100 # the memory module/ queue size
tau = 0.05 # temperature parameter in the objective
momentum = 0.999 # theta in momentum encoding step


# In[64]:


def not_ind_i(lst,ind_i): # returns all elements in list except pointed by the index ind_i
    if ind_i==0:
        return lst[ind_i+1:]
    elif ind_i==len(lst)-1:
        return lst[:-1]
    else:
        res=lst[:ind_i]
        res.extend(lst[ind_i+1:])
        return res
# lst=[102,11,299,32,468]
# for ind_i in range(len(lst)):
#     print(ind_i)
#     print(not_ind_i(lst,ind_i))    


# In[65]:


domains=['art_painting','cartoon','photo','sketch']
classes=['dog','elephant','giraffe','guitar','horse','house','person']

# domains=['art_painting']
# classes=['dog']

root_folder='/data/PACS' # path to the directory containing the PACS dataset for DG

domain_list=[]
class_list=[]
for (id_d,domain) in (enumerate(domains)):
    domain_list.append((id_d,domain))
for (id_c,class_) in (enumerate(classes)):
    class_list.append((id_c,class_))

print('domain_list: ',domain_list)
print('class_list: ',class_list)

train_filenames=[]
for class_ in classes:
    tmp_names=[]
    for domain in domains:
        tmp_names.append(sorted(glob.glob(root_folder+'/'+domain+'/'+class_+'/*',recursive=True)))
    train_filenames.append(tmp_names)

# id_im=0
# for id_c in range(len(train_filenames)):
#     for id_d in range(len(train_filenames[id_c])):
#         print(id_c,id_d)
#         print(train_filenames[id_c][id_d][id_im])


# In[67]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[68]:


def get_color_distortion(s=1.0):
    color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = T.RandomApply([color_jitter], p=0.8)
    
    # p is the probability of grayscale, here 0.2
    rnd_gray = T.RandomGrayscale(p=0.2)
    hor_flip = T.RandomHorizontalFlip(p=0.5)
    
    gauss_blur = T.GaussianBlur((3, 3), (1.0, 2.0))
    rnd_gauss_blur = T.RandomApply([gauss_blur], p=0.5)
    
    color_distort = T.Compose([rnd_color_jitter, rnd_gray, hor_flip, rnd_gauss_blur])
    
    return color_distort

class MyDataset(Dataset):
    def __init__(self, filenames,labels=False, mutation=True):
        self.file_names = filenames
        self.labels = labels
        self.mutation = mutation

    def __len__(self):
        return len(self.file_names)
    
    def tensorify(self, img):
        res = T.ToTensor()(img)
        res = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(res)
        return res
    
    def mutate_image(self, img):
        res = T.RandomResizedCrop(224)(img)
        res = get_color_distortion()(res)
        return res
    
    def __getitem__(self, idx):        
        
        #img_name = str(self.file_names[idx]).strip()
        img_name = self.file_names[idx] # idx'th element of the list "file_names" containing path to the idx'th image from the dataset
        image = Image.open(img_name).convert('RGB')
        

        if self.mutation:
            image = self.mutate_image(image)
            image = self.tensorify(image)
        else:
            image = T.Resize((224, 224))(image)
            image = self.tensorify(image)

        if self.labels:
            label = self.labels[idx]
            sample = {'image': image, 'label': label}
        else:
            sample = {'image': image}        

        return sample


# In[69]:


# defining base encoder

resnetq = models.resnet50(pretrained=True)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

resnetq.fc = Identity()

resnetk = copy.deepcopy(resnetq)

# moving the resnet architecture to device
resnetq.to(device)
resnetk.to(device)


# In[75]:


def get_queue(dataloader_traindata,queue_sz=100):
    flag = 0
    queue = None

    if queue is None:
        while True:

            with torch.no_grad():
                for (batch_idx, sample_batched) in enumerate(dataloader_traindata): 
                # for batch_idx in range(3):
#                     print('Processing mini-batch ', batch_idx,' for queue initialization ...')
                    x = sample_batched['image'].to(device)
                    k = resnetk(x) # Extract embeddings from the key encoder/ branch
                    # B=10
                    # dim=512
                    # k = torch.rand((B,dim))
                    fdim=k.size(1)
                    k = k.detach()  # [B,d], B: batch size
                    k = torch.div(k,torch.norm(k,dim=1).reshape(-1,1)) # l2 normalize each column vector, and maintain N,d form with reshape

                    if queue is None:
                        queue = k
                    else:
                        if queue.shape[0] < queue_sz:
                            queue = torch.cat((queue, k), 0) # concatenate mini-batches vertically; [K,d]    
                        else:
                            flag = 1

                    if flag == 1:
                        break

            if flag == 1:
                break

    return queue


# In[76]:


def get_pos_neg_queues(id_c,id_d,queue_sz=100):
    ind_class_other_domains=not_ind_i(train_filenames[id_c],id_d) # indexed class, other domains; len = N_d-1
    positive_queue=None
    for positive_domain in ind_class_other_domains:
        #print(len(positive_domain))
        positive_domain_dataloader = DataLoader(MyDataset(positive_domain), shuffle=True,
                                           num_workers = 6, batch_size = batch_size, drop_last = True)
        positive_domain_queue=get_queue(positive_domain_dataloader,queue_sz=min(queue_sz,len(positive_domain)))
        if positive_queue is None:
            positive_queue=positive_domain_queue
        else:
            positive_queue = torch.cat((positive_queue, positive_domain_queue), 0)
    print('Positive Queue Generated for class '+class_list[id_c][1]+' domain '+domain_list[id_d][1]+' with size '+str(positive_queue.size()))

    other_classes=not_ind_i(train_filenames,id_c) # remaining classes; len = N_c-1
    negative_queue=None

    for negative_class in other_classes:
        for neg_class_frm_dom in negative_class:
            negative_domain_dataloader = DataLoader(MyDataset(neg_class_frm_dom), shuffle=True,
                                               num_workers = 6, batch_size = batch_size, drop_last = True)
            negative_domain_queue=get_queue(negative_domain_dataloader,queue_sz=min(queue_sz,len(neg_class_frm_dom)))
            if negative_queue is None:
                negative_queue=negative_domain_queue
            else:
                negative_queue = torch.cat((negative_queue, negative_domain_queue), 0)
    print('Negative Queue Generated for class '+class_list[id_c][1]+' domain '+domain_list[id_d][1]+' with size '+str(negative_queue.size())) 
    return positive_queue,negative_queue


# In[ ]:


def loss_function(q, k, queue):

    N = q.shape[0]
    C = q.shape[1]

    pos = torch.exp(torch.div(torch.bmm(q.view(N,1,C), k.view(N,C,1)).view(N, 1),tau))
    neg = torch.sum(torch.exp(torch.div(torch.mm(q.view(N,C), torch.t(queue)),tau)), dim=1)
    denominator = neg + pos

    return torch.mean(-torch.log(torch.div(pos,denominator)))


# In[ ]:


class AttenHead(nn.Module):
    def __init__(self, fdim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.fatt = fdim//num_heads

        for i in range(num_heads):
            setattr(self, f'embd{i}', nn.Linear(fdim, self.fatt))
        for i in range(num_heads):
            setattr(self, f'fc{i}', nn.Linear(2*self.fatt, self.fatt))
        self.fc = nn.Linear(self.fatt*num_heads, fdim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, fx_in, fp_in):
        fp_in = fp_in.squeeze(0)# Return tensor with all  dimensions of input of size 1 removed. 
        d = math.sqrt(self.fatt)

        Nx = len(fx_in)
        
#         print(fx_in.size(),fp_in.size())
        f = torch.cat([fx_in, fp_in])
        f = torch.stack([getattr(self, f'embd{i}')(f) for i in range(self.num_heads)])  # head x N x fatt
        # f: torch.Size([1, 5, 6]), ie., [Nheads,(Nfx+Nfp),#dims] 
        # fx: torch.Size([1, 6]), i.e, [Nfx,#dims] 
        # fp: torch.Size([1, 4, 6]), i.e., [Nheads,Nfp,#dims] 
        fx, fp = f[:, :Nx], f[:, Nx:]

        w = self.dropout(F.softmax(torch.matmul(fx, torch.transpose(fp, 1, 2)) / d, dim=2))  # head x Nx x Np
        fa = torch.cat([torch.matmul(w, fp), fx], dim=2)  # head x Nx x 2*fatt
        fa = torch.stack([F.relu(getattr(self, f'fc{i}')(fa[i])) for i in range(self.num_heads)])  # head x Nx x fatt
        fa = torch.transpose(fa, 0, 1).reshape(Nx, -1)  # Nx x fdim
        fx = F.relu(fx_in + self.fc(fa))  # Nx x fdim
        w = torch.transpose(w, 0, 1)  # Nx x head x Np

        return fx, w


# In[ ]:


## gated fusion based representatives refinement ... 
def gated_fusion_refinement(fx,fp):
    fx_repeat=fx.repeat(fp.size(0),1)
    fxfpcat=torch.cat([fx_repeat, fp], dim=1)
#     print(fxfpcat,fxfpcat.size())

    
    #g_att = nn.Linear(fxfpcat.size(1), fxfpcat.size(1)//2,bias=False)
    z=g_att(fxfpcat).sigmoid()

#     print('z_vector:\n',z,z.size())

    fx_repeat_tanh=fx_repeat.tanh()
    # print(fx_repeat_tanh,fx_repeat_tanh.size())

    fp_tanh=fp.tanh()
    # print(fp_tanh,fp_tanh.size())

    fp_refined=fp_tanh*(1-z)+fx_repeat_tanh*z # refined means

    return fp_refined


# In[ ]:


## feature refinement and augmentation...
def refine_augment(fx,fp_refined):
    
    #atten = AttenHead(fx.size(1), num_heads=1)
    # print(atten)
    fxg, wx = atten(fx, fp_refined.unsqueeze(0)) # fxg: Refined feature for further use, wx: attention weights
    return fxg,wx


# In[ ]:


def get_augmented_feature(fx,fp):
    fp_refined=gated_fusion_refinement(fx,fp)
    fxg,wx=refine_augment(fx,fp_refined)
    return fxg

def get_augmented_batch(fx_batch,fp):
    fx_batch_aug=None
    for col in range(fx_batch.size(0)):
        fx_=fx_batch[col]
        fx_=torch.reshape(fx_, (1, fx_.size(0)))
        if fx_batch_aug==None:
            fx_batch_aug=get_augmented_feature(fx_,fp)
        else:
            fx_batch_aug=torch.cat((fx_batch_aug, get_augmented_feature(fx_,fp)), 0)
    return fx_batch_aug


# In[ ]:


# using SGD optimizer
# NOTE: Only query branch is going to be backpropagated

atten = AttenHead(fdim, num_heads=1).to(device)
g_att = nn.Linear(2*fdim, fdim,bias=False).to(device)
opt = optim.SGD( list(resnetq.parameters())+list(atten.parameters())+list(g_att.parameters()) , lr=0.001, momentum=0.9, weight_decay=1e-6)


# In[ ]:


## Sanity Check
# for id_c in range(len(train_filenames)):
#     for id_d in range(len(train_filenames[id_c])):
#         print(id_c,id_d, 'class: ',class_list[id_c][1],' domain: ',domain_list[id_d][1])
#         positive_queue,negative_queue=get_pos_neg_queues(id_c,id_d,queue_sz)
        
#         ind_class_ind_domain=train_filenames[id_c][id_d] # indexed class from indexed domain; len = n_files
#         ind_class_ind_domain_labels=[id_c for i in range(len(ind_class_ind_domain))]

#         ind_class_ind_domain_dataloader = DataLoader(MyDataset(ind_class_ind_domain,labels=ind_class_ind_domain_labels), shuffle=True,
#                                    num_workers = 6, batch_size = batch_size, drop_last = True)
        
# # print(len(ind_class_ind_domain),len(ind_class_other_domains),len(other_classes))
# # print(len(ind_class_ind_domain_labels),ind_class_ind_domain_labels)


# In[ ]:


losses_train=[]

# get query encoder in train mode
resnetq.train()
atten.train()
g_att.train()

# run a for loop for num_epochs
for epoch in range(num_epochs):
    
    t1=time.time()
    # a list to store losses for each epoch
    epoch_losses_train = []
    
    for id_c in range(len(train_filenames)):
        for id_d in range(len(train_filenames[id_c])):
            print(id_c,id_d, 'class: ',class_list[id_c][1],' domain: ',domain_list[id_d][1])
            positive_queue,negative_queue=get_pos_neg_queues(id_c,id_d,queue_sz)

            ind_class_ind_domain=train_filenames[id_c][id_d] # indexed class from indexed domain; len = n_files
            ind_class_ind_domain_labels=[id_c for i in range(len(ind_class_ind_domain))]

            ind_class_ind_domain_dataloader = DataLoader(MyDataset(ind_class_ind_domain,labels=ind_class_ind_domain_labels), shuffle=True,
                                       num_workers = 6, batch_size = batch_size, drop_last = True)
    


            # run a for loop for each batch
            for (batch_idx, sample_batched) in enumerate(ind_class_ind_domain_dataloader):
#             for batch_idx in range(3):
                print('Processing mini-batch ', batch_idx,' ...')

                # sample the mini-batch images, and move them to the device
                xq = sample_batched['image'].to(device)        

#                 ## dummy sanity check
#                 B=10
#                 dim=512
#                 q = torch.rand((B,dim))
                
                # get outputs of queries
                q = resnetq(xq)          

            
                #### Gated Fusion + Attention based refinement and augmentations
                k=get_augmented_batch(q,positive_queue)
                
                # detach the key as we won't be backpropagating by the key encoder
                k = k.detach()

                q = torch.div(q,torch.norm(q,dim=1).reshape(-1,1))
                k = torch.div(k,torch.norm(k,dim=1).reshape(-1,1))

                # get loss value
                # q: embedding of the anchor
                # k: embedding of the positive (augmented anchor)
                # negative_queue: embeddings in the memory queue
                loss = loss_function(q, k, negative_queue)

                opt.zero_grad()
                print('loss: ',loss.data.item())


                # put that loss value in the epoch losses list
                epoch_losses_train.append(loss.cpu().data.item())

                loss.backward()
                opt.step()


                positive_miniqueue,negative_miniqueue=get_pos_neg_queues(id_c,id_d,batch_size)
                # update the queues
                positive_queue = torch.cat((positive_queue, positive_miniqueue), 0) # Enqueue
                positive_queue = positive_queue[positive_miniqueue.size(0):,:] # Dequeue
                negative_queue = torch.cat((negative_queue, negative_miniqueue), 0) # Enqueue
                negative_queue = negative_queue[negative_miniqueue.size(0):,:] # Dequeue

                # update resnetk
                for theta_k, theta_q in zip(resnetk.parameters(), resnetq.parameters()):
                    theta_k.data.copy_(momentum*theta_k.data + theta_q.data*(1.0 - momentum))

    # append mean of epoch losses to losses_train, essentially this will reflect mean batch loss
    avg_minibatch_loss=sum(epoch_losses_train) / len(epoch_losses_train)
    losses_train.append(avg_minibatch_loss)

    print('Epoch: ',epoch,', Loss: ',avg_minibatch_loss)

    print('Time elapsed for the epoch:', time.time()-t1)

    # Plot the training losses Graph and save it
    fig = plt.figure(figsize=(10, 10))
    sns.set_style('darkgrid')
    plt.plot(losses_train)
    plt.legend(['Training Losses'])
    plt.savefig('model_results/losses.png')
    plt.close()

    # Store model and optimizer files
    np.savez("model_results/lossesfile", np.array(losses_train))
    torch.save(resnetq.state_dict(), 'model_results/trained_models/model_epoch_'+str(epoch)+'.pt')


# In[86]:




