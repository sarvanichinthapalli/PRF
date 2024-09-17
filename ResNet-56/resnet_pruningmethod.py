import torch as th
import torch.nn as nn 
import numpy as np
import os
import math


class PruningMethod():
    
    def prune_filters(self,threshold,prunes,folder_name,prune_value,logger):
      conv_layer=0
      first_ele= None #-------
      print(prunes)
      
      for layer_name, layer_module in self.named_modules():

        if(isinstance(layer_module, th.nn.Conv2d)  and layer_name!='conv1'):

          weight = layer_module.weight.data.cpu().numpy()

          if(layer_name.find('conv1')!=-1): #___Prune out-channels____
            #print(layer_name,'.....',conv_layer,'.....', threshold[conv_layer])
            #CREATE PRUNE STEP FOLDER
            path1=None#os.path.join(folder_name+"conv"+str(conv_layer+1)+"/prune"+str(prunes))
            #os.makedirs(path1)

            in_channels=[i for i in range(layer_module.weight.shape[1])]

            weight_layers_rearranged_pruned = weight
            #print('2a...',weight_layers_rearranged_pruned.shape)
            weight_layers_rearranged = np.reshape(weight_layers_rearranged_pruned, [weight_layers_rearranged_pruned.shape[0], -1])
            #print('3a.....',weight_layers_rearranged.shape)
            prune_count= prune_value[conv_layer]
            #print('prune_count...',prune_count)

            if(prunes>1000): # for one_shot pruning
               #print('NORMAL  calling.....')
               n_clusters_,first_ele = self.cluster_weights_agglo(weight_layers_rearranged.T, threshold[conv_layer],prunes,path1,conv_layer,prune_count,logger) 
            else:
               #print('calling.....')
               n_clusters_,first_ele = self.cluster_weights_agglo(weight_layers_rearranged.T, threshold[conv_layer],prunes,path1,conv_layer,prune_count, logger,first=True) #50% pruning

            first_ele = sorted(first_ele)




            out_channels= first_ele
            layer_module.weight = th.nn.Parameter( th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])).to('cuda'))



          if(layer_name.find('conv2')!=-1): #___Prune in-channels____
             #print(layer_name,'.....',conv_layer,'.....', threshold[conv_layer])
             in_channels=first_ele
             '''weight_layers_rearranged = np.transpose(weight, (1, 0, 2, 3))
             weight_layers_rearranged_pruned = weight_layers_rearranged[first_ele]
             weight_layers_rearranged_pruned = np.transpose(weight_layers_rearranged_pruned, (1, 0, 2, 3))'''

             out_channels=[i for i in range(layer_module.weight.shape[0])]
             layer_module.weight = th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[:,in_channels])).to('cuda'))
             conv_layer+=1
         
          layer_module.in_channels=len(in_channels)
          layer_module.out_channels=len(out_channels)

          
          #print(self)


        if (isinstance(layer_module, th.nn.BatchNorm2d) and layer_name!='bn1' and layer_name.find('bn1')!=-1):
            #print(layer_name,'.....',conv_layer,'.....', threshold[conv_layer])
            out_channels= first_ele

            layer_module.weight=th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])).to('cuda'))
            layer_module.bias=th.nn.Parameter(th.FloatTensor(th.from_numpy(layer_module.bias.data.cpu().numpy()[out_channels])).to('cuda'))

            layer_module.running_mean= th.from_numpy(layer_module.running_mean.cpu().numpy()[out_channels]).to('cuda')
            layer_module.running_var=th.from_numpy(layer_module.running_var.cpu().numpy()[out_channels]).to('cuda')



            layer_module.num_features= len(out_channels)
            #print(self)
        if isinstance(layer_module, nn.Linear):

            break

