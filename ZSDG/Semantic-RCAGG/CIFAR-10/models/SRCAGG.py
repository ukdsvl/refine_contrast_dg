from .classifier import Classifier
import torch
import torch.nn as nn
from lib.config import *
import math
from .featureExtractor import *

################################ Code required for SRCAGG ################################ 
from models import queue_var
import copy

queue_sz = queue_var.queue_sz # the memory module/ queue size
momentum = queue_var.momentum # theta in momentum encoding step
################################ Code required for SRCAGG ################################ 

def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)


class SRCAGG:
    def __init__(self):
        super(SRCAGG, self).__init__()

        learning_rate = 0.0001
        weight_decay = 0.00005

        #RMSPROP
        # learning_rate = 0.0003
        # weight_decay = 5e-6

        self.feature_extractor = featureExtractor().to(device)
        self.epoch = 0
        self.Classifier = Classifier().to(device)
        self.Classifier.apply(Xavier)
        
        self.key_encoder = copy.deepcopy(self.feature_extractor) # not going to be backpropagated
        
        fdim=300#self.Classifier.in_features
        self.atten = queue_var.AttenHead(fdim, num_heads=1)#.to(device)
        self.g_att = nn.Linear(2*fdim, fdim,bias=False)#.to(device)

        
        self.optimizer = torch.optim.Adamax(list(self.feature_extractor.parameters())+list(self.Classifier.parameters())+
                                            list(self.atten.parameters())+list(self.g_att.parameters()),
                                            lr=learning_rate,weight_decay=weight_decay)

        self.criterion_mse = nn.MSELoss()
        self.criterion_ce = nn.CrossEntropyLoss()
        self.vector = w2v

    ## gated fusion based representatives refinement ... 
    def gated_fusion_refinement(self, fx,fp):
        fx_repeat=fx.repeat(fp.size(0),1)
        fxfpcat=torch.cat([fx_repeat, fp], dim=1)
    #     print(fxfpcat,fxfpcat.size())


        #g_att = nn.Linear(fxfpcat.size(1), fxfpcat.size(1)//2,bias=False)
        z=self.g_att(fxfpcat).sigmoid()

    #     print('z_vector:\n',z,z.size())

        fx_repeat_tanh=fx_repeat.tanh()
        # print(fx_repeat_tanh,fx_repeat_tanh.size())

        fp_tanh=fp.tanh()
        # print(fp_tanh,fp_tanh.size())

        fp_refined=fp_tanh*(1-z)+fx_repeat_tanh*z # refined means

        return fp_refined

    ## feature refinement and augmentation...
    def refine_augment(self, fx,fp_refined):

        #atten = AttenHead(fx.size(1), num_heads=1)
        # print(atten)
        fxg, wx = self.atten(fx, fp_refined.unsqueeze(0)) # fxg: Refined feature for further use, wx: attention weights
        return fxg,wx


    def get_augmented_feature(self, fx,fp):
        fp_refined=self.gated_fusion_refinement(fx,fp)
        fxg,wx=self.refine_augment(fx,fp_refined)
        return fxg

    def get_augmented_batch_looped(self, fx_batch,fp):
        fx_batch_aug=None
        for col in range(fx_batch.size(0)):
            fx_=fx_batch[col]
            fx_=torch.reshape(fx_, (1, fx_.size(0)))
            if fx_batch_aug==None:
                fx_batch_aug=self.get_augmented_feature(fx_,fp)
            else:
                fx_batch_aug=torch.cat((fx_batch_aug, self.get_augmented_feature(fx_,fp)), 0)
        return fx_batch_aug


    def get_augmented_batch(self,fx_batch,fp):
        batch_list=list(fx_batch)
        batch_list_updated = list(map(lambda im_fx:
                                      self.get_augmented_feature(torch.reshape(im_fx, (1, im_fx.size(0))),fp), batch_list))

        fx_batch_aug = torch.stack([sub_list for sub_list in batch_list_updated], dim=0)
        fx_batch_aug = torch.squeeze(fx_batch_aug, 1)
        return fx_batch_aug

        

#     def train(self, X, Y, domainId):
#         self.optimizer.zero_grad()

#         out1 = self.feature_extractor(X)
#         loss1 = self.criterion_mse(out1, self.vector[Y])

#         out2 = self.Classifier(out1)
#         loss2 = self.criterion_ce(out2, Y)

#         cost = loss1 + loss2

#         cost.backward()

#         self.optimizer.step()
#         self.optimizer.zero_grad()

#         return cost.data.item()

    
    def update(self, minibatches):
        train_queues = queue_var.train_queues
        nclass=len(train_queues)
        ndomains=len(train_queues[0])
        loss_rc=0 # for accumulating the additive contrastive loss
        
        all_x=None
        all_y=None
        
        for id_c in range(nclass): # loop over classes
            for id_d in range(ndomains): # loop over domains
                mb_ids=(minibatches[id_d][1] == id_c).nonzero(as_tuple=True)[0]
                # indices of those egs from domain id_d, whose class label is id_c
                label_tensor=minibatches[id_d][1][mb_ids] # labels
                if mb_ids.size(0)==0:
                    #print('class has no element')
                    continue
                data_tensor=minibatches[id_d][0][mb_ids] # data
                # extract query features: torch.Size([negs, dim])         
                q = self.feature_extractor(data_tensor) 

                if all_x is None:
                    all_x = q
                    all_y = label_tensor
                else:
                    all_x = torch.cat((all_x, q), 0)
                    all_y = torch.cat((all_y, label_tensor), 0)
                
                positive_queue,negative_queue=queue_var.get_pos_neg_queues(id_c,id_d,train_queues)
                #### Gated Fusion + Attention based refinement and augmentations
                k=self.get_augmented_batch(q,positive_queue)
                
                # detach the key as we won't be backpropagating by the key encoder
                k = k.detach()

                #l2 normalize
                q = torch.div(q,torch.norm(q,dim=1).reshape(-1,1))
                k = torch.div(k,torch.norm(k,dim=1).reshape(-1,1))

                # get loss value
                # q: embedding of the anchor
                # k: embedding of the positive (augmented anchor)
                # negative_queue: embeddings in the memory queue
                loss_rc += queue_var.loss_function(q, k, negative_queue)
                
                data_emb = self.key_encoder(data_tensor) # extract features: torch.Size([negs, dim])
                data_emb = data_emb.detach()
                data_emb = torch.div(data_emb,torch.norm(data_emb,dim=1).reshape(-1,1))#l2 normalize
                
                # update queue for this class and this domain
                current_queue = train_queues[id_c][id_d]
                current_queue = torch.cat((current_queue, data_emb), 0)
                current_queue = current_queue[-queue_sz:] # keep only the last queue_sz entries
                train_queues[id_c][id_d] = current_queue
        

        all_pred=self.Classifier(all_x)
        ## ZSDG 
        loss1 = self.criterion_mse(all_x, self.vector[all_y])
        loss2 = self.criterion_ce(all_pred, all_y)

        loss = loss1 + loss2 + loss_rc
        ## ZSDG 
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update key_encoder
        for theta_k, theta_q in zip(self.key_encoder.parameters(), self.feature_extractor.parameters()):
            theta_k.data.copy_(momentum*theta_k.data + theta_q.data*(1.0 - momentum))

        queue_var.train_queues = train_queues # update the global variable
            
        #return {'loss': loss.item(),'loss_ce': loss_ce.item(),'loss_rc': loss_rc.item()}
        return loss.data.item()
