import torch as th
import torch.nn as nn 
import numpy as np
import os
import math

#seed = 1787
#np.random.seed(seed)

filters_selected=[]
#filters_remaining=[]
no_of_dimensions=-1

#prune_percentage=[0.02]*2+[0.04]*2+[0.05]*3+[0.1]*6
class PruningMethod():

   
    def prune_filters(self,threshold,prunes,folder_name,prune_percentage):
      conv_layer=0
      total=None #-------
      first_ele= None #-------
      #nb_remanining_filters = []
      print(prunes)
  
      for layer_name, layer in self.named_modules():
        #print(layer_name)

        if isinstance(layer, nn.Conv2d):

            #CREATE PRUNE STEP FOLDER
            path1=None#os.path.join(folder_name+"conv"+str(conv_layer+1)+"/prune"+str(prunes))
            #os.makedirs(path1)

            #print(layer_name)
            weight = layer.weight.data.cpu().numpy()
            bias = layer.bias.data.cpu().numpy()
            #print('1..',weight.shape)
            if first_ele is not None:
                weight_layers_rearranged = np.transpose(weight, (1, 0, 2, 3))
                weight_layers_rearranged_pruned = weight_layers_rearranged[first_ele]
                weight_layers_rearranged_pruned = np.transpose(weight_layers_rearranged_pruned, (1, 0, 2, 3))
            else:
                weight_layers_rearranged_pruned = weight
            #print('2a...',weight_layers_rearranged_pruned.shape)
            weight_layers_rearranged = np.reshape(weight_layers_rearranged_pruned, [weight_layers_rearranged_pruned.shape[0], -1])
            #print('3a.....',weight_layers_rearranged.shape)
            prune_count= math.ceil(weight_layers_rearranged.shape[0] * prune_percentage[conv_layer])
            #print('prune_count...',prune_count)

            if(prunes>1000): # for one_shot pruning
               #print('NORMAL  calling.....')
               n_clusters_,first_ele = self.cluster_weights_agglo(weight_layers_rearranged.T, threshold,prunes,path1,conv_layerprune_count) 
            else:
               #print('calling.....')
               n_clusters_,first_ele = self.cluster_weights_agglo(weight_layers_rearranged.T, threshold,prunes,path1,conv_layer,prune_count, first=True) #50% pruning


            conv_layer+=1



            first_ele = sorted(first_ele)

            weight_pruned = weight_layers_rearranged[first_ele]
            #print('4..',weight_pruned.shape)
            bias_pruned = bias[first_ele]


            weight_pruned = np.reshape(weight_pruned, [len(first_ele), weight_layers_rearranged_pruned.shape[1],weight_layers_rearranged_pruned.shape[2],weight_layers_rearranged_pruned.shape[3]])
            #print('5..',weight_pruned.shape)

            params_1 = np.shape(weight_pruned)
            layer.out_channels = params_1[0]
            layer.in_channels = params_1[1]

            weight_tensor = th.from_numpy(weight_pruned)
            bias_tensor = th.from_numpy(bias_pruned)
            layer.weight = th.nn.Parameter(weight_tensor)
            layer.bias = th.nn.Parameter(bias_tensor)

           
            #ii+=1
            #rr+=1

        if isinstance(layer, nn.BatchNorm2d) and first_ele is not None:
            bnorm_weight = layer.weight.data.cpu().numpy()
            bnorm_weight = bnorm_weight[first_ele]
            bnorm_bias = layer.bias.data.cpu().numpy()
            bnorm_bias = bnorm_bias[first_ele]

            bnorm_tensor = th.from_numpy(bnorm_weight)
            bias_tensor = th.from_numpy(bnorm_bias)
            layer.weight = th.nn.Parameter(bnorm_tensor)
            layer.bias = th.nn.Parameter(bias_tensor)

            layer.num_features = int(np.shape(bnorm_weight)[0])
            bnorm_rm = layer.running_mean.cpu().numpy()
            bnorm_rm = bnorm_rm[first_ele]
            bnorm_rv = layer.running_var.cpu().numpy()
            bnorm_rv = bnorm_rv[first_ele]
            running_mean = th.from_numpy(bnorm_rm)
            layer.running_mean = running_mean
            running_var = th.from_numpy(bnorm_rv)
            layer.running_var = running_var
            #rr+=1

        if isinstance(layer, nn.Linear):
            weight_linear = layer.weight.data.cpu().numpy()
            #print('1...',weight_linear.shape)
            weight_linear_rearranged = np.transpose(weight_linear, (1, 0))
            weight_linear_rearranged_pruned = weight_linear_rearranged[first_ele]
            weight_linear_rearranged_pruned = np.transpose(weight_linear_rearranged_pruned, (1, 0))
            #print('2....',weight_linear_rearranged_pruned.shape)

            layer.in_features = int(np.shape(weight_linear_rearranged_pruned)[1])
            linear_tensor = th.from_numpy(weight_linear_rearranged_pruned)
            layer.weight = th.nn.Parameter(linear_tensor)
            break

           
            

      
    def get_indices_topk(self,layer_bounds,i,prune_limit,prune_percentage):

      #global prune_percentage
      indices=int(len(layer_bounds)*prune_percentage[i])+1 #1
      p=len(layer_bounds) #3
      if (p-indices)<prune_limit: #3-1<3
         remaining=p-prune_limit
         indices=remaining
      k=sorted(range(len(layer_bounds)), key=lambda j: layer_bounds[j])[:indices]
      #print('indidces',k, 'len ',len(layer_bounds))
      return k

    def get_indices_bottomk(self,layer_bounds,i,prune_limit):

      k=sorted(range(len(layer_bounds)), key=lambda j: layer_bounds[j])[-prune_limit:]
      #print('indidces',k, 'len ',len(layer_bounds))
      return k

