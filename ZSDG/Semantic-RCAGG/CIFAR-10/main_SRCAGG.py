import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import numpy as np
from models.AGG import AGG

import math
from lib.config import *

from lib.utils import PCA_TSNE
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
np.random.seed(seed)

import time
from models import queue_var

datasets.CIFAR10(root='../../data/', download=True, train=True)

classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print('device:', device)

def load_rotated_cifar(left_out_idx):

    train_x = []
    test_x = []
    train_y = []
    test_y = []
    random = []
    for i in range(10):
        random.append(np.random.permutation(5000))

    for i in range(6):
        angle = 360 - 15 * i
        transform = transforms.Compose([transforms.RandomRotation(degrees=(angle, angle)), transforms.ToTensor()])
        cifar_train = datasets.CIFAR10(root='../../data/', download=False, train=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=cifar_train, batch_size=50000, shuffle=False)

        full_data = next(iter(train_loader))

        targets = full_data[1]

        data = full_data[0]

        data_x = []
        data_y = []
        for j in range(10):
            idx = targets == j
            jth_target = targets[idx].to(device)
            jth_data = data[idx].to(device)
            jth_data = jth_data[random[j]]

            sample_x = jth_data[:400]
            sample_y = jth_target[:400]

            if i != left_out_idx:
                data_x.append(sample_x)
                data_y.append(sample_y)

            if i==left_out_idx:
                data_x.append(jth_data)
                data_y.append(jth_target)

        data_x = torch.cat(data_x).to(device)
        data_y = torch.cat(data_y).to(device)

        if i != left_out_idx:
            train_x.append(data_x)
            train_y.append(data_y)
        else:
            test_x = data_x
            test_y = data_y

    return train_x, train_y, test_x, test_y


def domain_specific_training(dom,clas):
    train_x, train_y, test_x, test_y = load_rotated_cifar(dom)
    length_of_domain = len(train_x[0])

    train_rst = []
    train_ms_rst = []
    zs_rst = []
    zs_mse_rst = []
    test_rst = []
    test_mse_rst = []

    for i in range(len(train_x)):
        for k in clas:
            idx = train_y[i] != k
            train_y[i] = train_y[i][idx]
            train_x[i] = train_x[i][idx]

    zs_y = []
    zs_x = []
    for k in clas:
        idx = test_y ==k
        zs_x.append(test_x[idx])
        zs_y.append(test_y[idx])

        idx = test_y != k
        test_y = test_y[idx]
        test_x = test_x[idx]

    zs_x = torch.cat(zs_x)
    zs_y = torch.cat(zs_y)

    train_x = torch.cat(train_x)
    train_y = torch.cat(train_y)

    train_x = train_x.view(5, length_of_domain-800, 3, 32, 32) # just rearranging train domain data, for indexing
    train_y = train_y.view(5, length_of_domain-800).long()

    length_of_domain -= 800

    # print(length_of_domain)
    # print(train_x[0].shape, train_y[0].shape, test_x.shape, test_y.shape)

    batch_size=50
    print('------------------------')

    model = SRCAGG()
    for epoch in range(100):
        print('epoch:',epoch)
        x_train = []
        y_train = []

        random = np.random.permutation(length_of_domain)

        for i in range(5): #every chosen train domain
            x = train_x[i]
            x_permuted = x[random]

            y = train_y[i]
            y_permuted = y[random]

            x_train.append(x_permuted)
            y_train.append(y_permuted)

        x_train = torch.cat(x_train).to(device)
        x_train = x_train.view(5,length_of_domain,3,32,32)

        y_train  = torch.cat(y_train).to(device)
        y_train = y_train.view(5,length_of_domain)
        
        ################################ Code required for SRCAGG ################################ 
        #print(x_train.size(),y_train.size()) #torch.Size([5, 3200, 3, 32, 32]) torch.Size([5, 3200])

        ## class ids might be disconnected due to the ZS setting, so, rename them as 0,1,...C-1
        unique_cids=list(y_train.unique())
        for ii in range(y_train.size(0)):
            for jj in range(y_train.size(1)):
                y_train[ii,jj]=unique_cids.index(y_train[ii,jj])

        ### Native ZSDG style
        #     for i in range(5): # for every domain, accumulate/aggregate
        #         avg_cost = 0
        #         for k in range(0,length_of_domain,batch_size): #sample mini-batches

        #             left_x = x_train[i][k:k+batch_size,:,:,:]
        #             labels = y_train[i][k:k+batch_size]

        #             avg_cost+= model.train(left_x,labels,i)

        #         avg_cost = avg_cost/(length_of_domain/batch_size)

        #     print(avg_cost)

        ### Domainbed style
        all_minibatches=[] # len(all_minibatches)=64 (3200/50), bsz=50
        for k in range(0,length_of_domain,batch_size): #sample mini-batches
            minibatches_device=[] #analogy to domainbed
            for i in range(5): # for every domain, accumulate/aggregate
                tmp_list=[]
                left_x = x_train[i][k:k+batch_size,:,:,:]
                labels = y_train[i][k:k+batch_size]
                tmp_list.append(left_x)
                tmp_list.append(labels)
                minibatches_device.append(tmp_list)
            all_minibatches.append(minibatches_device)

        print('Firstly, computing Queues for the algorithm ')
        queue_sz = queue_var.queue_sz # the memory module/ queue size
        minibatches_device=all_minibatches[0]
        num_classes=len(y_train.unique())
        num_domains=len(minibatches_device)
        # pre-populate the global list of queues ...
        # Later, minibatches might have some classes with no eg, therefore, this step is necessary
        # (though it looks redundant), as we want to ensure a proper order of storage.
        train_queues=[] # create an adhoc variable for speed-up
        for id_c in range(num_classes):
            tmp_queues=[]
            for id_d in range(num_domains):
                tmp_queues.append(None)
            train_queues.append(tmp_queues)
        #queue_var.train_queues=train_queues # assign to the global variable

        # create an array to store flags to indicate whether queues have reached their required sizes
        flag_arr = np.zeros([num_classes, num_domains], dtype = int)

        #### The creation of the initial list of queues
        #train_queues=queue_var.train_queues # create an adhoc variable for speed-up 
        # assigning directly into the global variable caused slow speed.
        tpcnt=0
        while not np.sum(flag_arr)==num_classes*num_domains: #until all queues have queue_sz elts
            minibatches_device=all_minibatches[tpcnt]
            print('\n tpcnt: ',tpcnt,' ... Completed',np.sum(flag_arr),' queues out of ',num_classes*num_domains)
            for id_c in range(num_classes): # loop over classes
                for id_d in range(num_domains): # loop over domains
                    if flag_arr[id_c][id_d]==1:
                        print('Queue (class ',id_c,', domain ',id_d,') is completely filled. ')
                        continue
                    else:
                        mb_ids=(minibatches_device[id_d][1] == id_c).nonzero(as_tuple=True)[0]
                        # indices of those egs from domain id_d, whose class label is id_c
                        label_tensor=minibatches_device[id_d][1][mb_ids] # labels
                        if mb_ids.size(0)==0:
                            print('class has no element')
                            continue
                        data_tensor=minibatches_device[id_d][0][mb_ids] # data
                        data_emb = model.key_encoder(data_tensor) # extract features: torch.Size([negs, dim])
                        data_emb = data_emb.detach()
                        data_emb = torch.div(data_emb,torch.norm(data_emb,dim=1).reshape(-1,1))#l2 normalize

                        current_queue=train_queues[id_c][id_d]
                        if current_queue is None:
                            current_queue = data_emb
                        elif current_queue.size(0) < queue_sz:
                            current_queue = torch.cat((current_queue, data_emb), 0)    
                        if current_queue.size(0) > queue_sz:
                            # keep only the last queue_sz entries
                            current_queue = current_queue[-queue_sz:] # keep only the last queue_sz entries
                        if current_queue.size(0) == queue_sz:
                            flag_arr[id_c][id_d]=1
                        train_queues[id_c][id_d] = current_queue
                        print('Queue (class ',id_c,', domain ',id_d,') : ',train_queues[id_c][id_d].size())
            tpcnt+=1
            if tpcnt==len(all_minibatches):
                tpcnt=0
        queue_var.train_queues=train_queues # assign to the global variable

        # sanity checking the queues obtained
        for id_c in range(num_classes):
            for id_d in range(num_domains):
                print('Queue (class ',id_c,', domain ',id_d,') : ',queue_var.train_queues[id_c][id_d].size())

        model.atten.train()
        model.g_att.train()
        
        
        mb_costs=[]
        for idx, minibatches in enumerate(all_minibatches):
            #if idx==2:
            #    break
            mb_cost=model.update(minibatches)
            mb_costs.append(mb_cost)
        avg_cost=sum(mb_costs)/len(mb_costs)    
        #print('Epoch ',epoch,' avg_cost: ',avg_cost)
        print(avg_cost)
        
        ################################ Code required for SRCAGG ################################ 
        
        if (epoch+1)%25==0:
            # print('After {} epochs'.format(epoch))

            criterion = nn.MSELoss()

            feature_extractor = model.feature_extractor
            feature_extractor.eval()

            classifier = model.Classifier
            classifier.eval()

            with torch.no_grad():
                x_train = x_train.view(5 * length_of_domain, 3, 32, 32)
                y_train = y_train.view(5 * length_of_domain).long()

                train_tensor = torch.utils.data.TensorDataset(x_train, y_train)
                train_loader = torch.utils.data.DataLoader(dataset=train_tensor, batch_size=500, shuffle=False)

                y_train = []
                train_out = []
                classifier_out = []
                for i, (X, Y) in enumerate(train_loader):
                    feat = feature_extractor(X)
                    out = classifier(feat)
                    train_out.append(feat)
                    classifier_out.append(out)
                    y_train.append(Y)

                train_out = torch.cat(train_out).to(device)
                y_train = torch.cat(y_train).to(device)
                classifier_out = torch.cat(classifier_out).to(device)

                # PCA_TSNE(np.array(train_out.cpu()),np.array(y_train.cpu()))

                predicted = []
                correct = 0
                for i in range(len(y_train)):
                    cost = []
                    for j in range(10):
                        cost.append(criterion(train_out[i], w2v[j]).item())

                    pred = cost.index(min(cost))
                    predicted.append(pred)
                    if pred == y_train[i]:
                        correct += 1

                accuracy = correct / len(y_train)
                print('Train MSE Accuracy of the model: {} %'.format(accuracy))
                train_ms_rst.append(100*accuracy)

                _, predicted = torch.max(classifier_out, dim=1)
                correct = sum(np.array((predicted == y_train).cpu()))
                # print(correct)
                accuracy = correct / len(y_train)
                print('Train Accuracy of the model: {} %'.format(accuracy))
                train_rst.append(100*accuracy)

            with torch.no_grad():
                zs_x = zs_x.view(10000, 3, 32, 32)
                zs_y = zs_y.view(10000).long()

                zs_tensor = torch.utils.data.TensorDataset(zs_x, zs_y)
                zs_loader = torch.utils.data.DataLoader(dataset=zs_tensor, batch_size=500, shuffle=False)

                y_zs = []
                zs_out = []
                zs_classifier_out = []
                for i, (X, Y) in enumerate(zs_loader):
                    feat = feature_extractor(X)
                    out = classifier(feat)
                    zs_out.append(feat)
                    zs_classifier_out.append(out)
                    y_zs.append(Y)

                zs_out = torch.cat(zs_out).to(device)
                y_zs = torch.cat(y_zs).to(device)
                zs_classifier_out = torch.cat(zs_classifier_out).to(device)

                # PCA_TSNE(np.array(train_out.cpu()),np.array(y_train.cpu()))

                predicted = []
                correct = 0
                for i in range(len(y_zs)):
                    cost = []
                    for j in range(10):
                        cost.append(criterion(zs_out[i], w2v[j]).item())
                    pred = cost.index(min(cost))
                    predicted.append(pred)
                    if pred == y_zs[i]:
                        correct += 1
                # print(predicted)
                accuracy = correct / len(y_zs)
                print('ZS MSE Accuracy of the model: {} %'.format(accuracy))

                predicted = []
                correct = 0
                correct1 = 0
                correct2=0
                for i in range(len(y_zs)):
                    cost = []
                    for j in clas:
                        cost.append(criterion(zs_out[i], w2v[j]).item())
                    # if i%500==0:
                    #     print(cost)
                    pred = clas[cost.index(min(cost))]
                    predicted.append(pred)
                    if pred == y_zs[i]:
                        correct += 1
                        if y_zs[i]==clas[0]:
                            correct1+=1
                        else:
                            correct2+=1

                # print(predicted)
                accuracy = correct / len(y_zs)
                print('ZS MSE Accuracy of the model: {} %'.format(accuracy))
                print('ZS MSE Accuracy of the model: {}.;;;{}:{} %'.format(accuracy,clas[0],correct1/5000))
                print('ZS MSE Accuracy of the model: {},;;;{}:{} %'.format(accuracy,clas[1],correct2/5000))
                zs_mse_rst.append(accuracy)

                _, predicted = torch.max(zs_classifier_out, dim=1)
                correct = sum(np.array((predicted == y_zs).cpu()))
                # print(correct)
                accuracy = correct / len(y_zs)
                print('ZS Accuracy of the model: {} %'.format(accuracy))
                zs_rst.append(100*accuracy)

                # PCA_TSNE(np.array(torch.cat((train_out, zs_out), 0).cpu()),(np.array(torch.cat((y_train, y_zs), 0).cpu())))

            with torch.no_grad():

                x_test = test_x.view((40000,3,32,32)).to(device)
                y_test = test_y.view(40000).to(device)

                test_tensor = torch.utils.data.TensorDataset(x_test, y_test)
                test_loader = torch.utils.data.DataLoader(dataset=test_tensor, batch_size=1000, shuffle=False)

                y_test = []
                test_out = []
                classifier_out = []
                for i, (X, Y) in enumerate(test_loader):
                    out = feature_extractor(X)
                    test_out.append(out)
                    classifier_out.append(classifier(out))
                    y_test.append(Y)

                test_out = torch.cat(test_out).to(device)
                y_test = torch.cat(y_test).to(device)
                classifier_out = torch.cat(classifier_out).to(device)

                _, predicted = torch.max(classifier_out, dim=1)
                correct = sum(np.array((predicted == y_test).cpu()))

                # print(correct)
                accuracy = correct / len(y_test)

                test_rst.append(100*accuracy)

                # PCA_TSNE(np.array(test_out.cpu()),np.array(y_test.cpu()))

                predicted = []
                correct = 0
                for i in range(len(y_test)):
                    cost = []
                    for j in range(10):
                        cost.append(criterion(test_out[i], w2v[j]).item())

                    pred = cost.index(min(cost))
                    predicted.append(pred)
                    if pred == y_test[i]:
                        correct += 1

                accuracy = correct / len(y_test)
                print('Test MSE Accuracy of the model: {} %'.format(accuracy))
                test_mse_rst.append(100*accuracy)
                del test_out

    return train_rst, train_ms_rst, zs_rst, zs_mse_rst, test_rst, test_mse_rst

import sys
seeds = [107,109,997,991,804,451,321,652,854,102]
for dom in range(5,6):
    import statistics

    zs = [[3,9],[3,5],[4,8],[1,4],[0,1]]
    for clas in zs:
        print(clas)

    for clas in zs:
        final_zs = []
        final_test_mse = []
        final_test = []

        print("----------------------Domain.{}.{}---------------------".format(dom, clas))

        for repeat in range(2): #considering only two seeds initially
            sys.stdout.flush()
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed_all(seeds[repeat])
            torch.manual_seed(seeds[repeat])
            np.random.seed(seeds[repeat])




            train_rst,train_ms , zs_rst,zs_mse, test_rst,test_mse = domain_specific_training(dom,clas=clas)
            # print('Train : ', max(train))
            # print('Train MSE :', max(train_ms))
            # print('ZS : ', max(zs))
            # print('ZS MSE : ',max(zs_mse))
            final_zs.append(max(zs_mse))



            # print('TEST : ', max(test_rst))
            final_test.append(max(test_rst))
            # print('Test MSE : ', max(test_mse))
            final_test_mse.append(max(test_mse))


        print('Zero Shot' , final_zs,'\n Mean:', statistics.mean(final_zs),'\n Std Dev',statistics.stdev(final_zs) )
        print('Test :', final_test,'\n Mean:', statistics.mean(final_test), '\n Std Dev',statistics.stdev(final_test))

        print('Test MSE :', final_test_mse,'\n Mean:', statistics.mean(final_test_mse), '\n Std Dev',statistics.stdev(final_test_mse))

