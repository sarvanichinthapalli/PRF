
import torch as th
import torch.nn as nn
import torch.nn.init as init
from numpy.linalg import norm
#import pandas as pd
import numpy as np
import logging
import csv 
from time import localtime, strftime
import os 
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift, estimate_bandwidth
import scipy.cluster.hierarchy as hcluster
import scipy.cluster.hierarchy as hac
import scipy.cluster.hierarchy as fclusterdata
import time
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import pairwise_distances
import math



class Network():

    def weight_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            if self.a_type == 'relu':
                init.kaiming_normal_(m.weight.data, nonlinearity=self.a_type)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'leaky_relu':
                init.kaiming_normal_(m.weight.data, nonlinearity=self.a_type)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'tanh':
                g = init.calculate_gain(self.a_type)
                init.xavier_uniform_(m.weight.data, gain=g)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'sigmoid':
                g = init.calculate_gain(self.a_type)
                init.xavier_uniform_(m.weight.data, gain=g)
                init.constant_(m.bias.data, 0)
            else:
                raise
                return NotImplemented

    
    def create_folders(self,total_convs):

      main_dir=strftime("/Results/%b%d_%H:%M:%S%p", localtime() )+"_Res56/"
      import os
      current_dir =  os.path.abspath(os.path.dirname(__file__))
      par_dir = os.path.abspath(current_dir + "/../")
      parent_dir=par_dir+main_dir
      '''for i in range(total_convs):
        path1=os.path.join(parent_dir, "conv"+str(i+1))
        os.makedirs(path1)'''
      os.makedirs(parent_dir)

      return parent_dir

    def get_writerow(self,k):

      s='wr.writerow(['
      for i in range(k):
          s=s+'d['+str(i)+']'
          if(i<k-1):
             s=s+','
          else:
             s=s+'])'
      return s

    def get_logger(self,file_path):

        logger = logging.getLogger('gal')
        log_format = '%(asctime)s | %(message)s'
        formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)

        return logger

    def cluster_weights_agglo(self,weight, threshold,prunes,folder_name,conv_layer,prune_count,logger, first=False, average=True):

        rem= prune_count
        #t0 = time.time()
        #print(type(weight))
        #print('1...',weight.shape)
        weight = weight.T
        #print('1...',weight.shape, type(weight))
        weight = normalize(weight, norm='l2', axis=1)
        #print('2...',weight.shape)
        threshold =  1.0-threshold   # Conversion to distance measure
        #clusters = hcluster.fclusterdata(weight, threshold, criterion="distance", metric='cosine', depth=1, method='centroid')
        z = hac.linkage(weight, metric='cosine', method='complete')

        #print('z.......',len(z))
        labels = hac.fcluster(z, threshold, criterion="distance")
        #print('lables...',labels)
        


       
        #print('sorted lables...',len(labels),np.sort(labels))

        labels_unique = np.unique(labels)
        #print('labels_unique...',labels_unique.shape,labels_unique)
        n_clusters_ = len(labels_unique)

        a=np.array(labels)
        #print('a....',a)
        sort_idx = np.argsort(a)
        #print('sorted idx...',len(sort_idx), sort_idx)
        a_sorted1 = a[sort_idx]
        #print('a_sorted...',len(a_sorted1), a_sorted1, type(a_sorted1))


        if(first==False): # retain ONLY 1 from each cluster one shot

            unq_first = np.concatenate(([True], a_sorted1[1:] != a_sorted1[:-1]))
            #print('unq_first...',len(unq_first), unq_first)

            unq_items = a_sorted1[unq_first]
            unq_count = np.diff(np.nonzero(unq_first)[0])
            unq_idx = np.split(sort_idx, np.cumsum(unq_count))
            first_ele = [unq_idx[idx][-1] for idx in range(len(unq_idx))]
        else: # retain % of filters from each cluster
            highest=[]
            a_sorted= np.sort(labels)
            val2= sort_idx
            temp=[] 
            filters=[]      
            #_______For getting each cluster as a sublist__________________
            for i in range(len(a_sorted)):
			   
                val=a_sorted[i]
		
                if (i== len(a_sorted)-1 ):
                      temp.append(val2[i])
                      filters.append(temp)
                      temp=[]
				
                elif(i!= len(a_sorted)-1 and (val != a_sorted[i+1])):
                      temp.append(val2[i])
                      filters.append(temp)
                      temp=[]
                else:
                      temp.append(val2[i])
            #print(filters) #____________contains each cluster as a sublist_______
 

            #__________for sorting within each cluster based on each weighs L1/L2 norm __________
            l1_norms=[]
            for i in range(len(filters)):
               temp=[]
               for f in filters[i]:
                 temp.append(norm(weight[f],1)) #L1-norm
               l1_norms.append(np.around(temp,3))  

            sorted_filters=[]
            for i in range(len(filters)):
                 sorted_filters.append( [x for _,x in sorted(zip(l1_norms[i],filters[i]))])
                 #print('l1_norms[i].....',l1_norms[i],'...filters[i]..',filters[i],'....sorted_filters[i].....',sorted_filters[i],'...max..',[max(l1_norms[i])])#______l1_norms is not sorted and filters is sorted____________

                 highest.extend([max(l1_norms[i])]) #____CONDITION FOR SELCTING WITHIN EACH CLUSTER___


            #______________plot histogram______
            '''for i in range(len(l1_norms)):
                 plt.hist(l1_norms[i], weights=np.ones(len(l1_norms[i])) / len(l1_norms[i]))
                 plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
                 plt.savefig(folder_name+"/c"+str(i+1)+".png")
                 plt.clf()
                 plt.cla()
                 plt.close()'''


            #_________SORTING AMONG CLUSTERS_____________
            cluster_imp_sorted_filters=[]
            #cluster_imp_sorted_filters.extend( [x for _,x in sorted(zip(highest,sorted_filters))] )#___filters in each cluser sorted based on l1 norm___
            cluster_imp_sorted_filters.extend( [x for _,x in sorted(zip(highest,filters))] ) #___filters in each cluser are random___
            logger.info(cluster_imp_sorted_filters)

            #__________for printing__________________
            '''print('+++++++++++')
            for i in range(len(cluster_imp_sorted_filters)):
                 juju=[]
                 for k in cluster_imp_sorted_filters[i]:
                     juju.append(norm(weight[k],1))
                 print('set...', cluster_imp_sorted_filters[i],'...highest...',max(juju),'...l1_calc...',juju)'''
            #__________END PRinting_____________________


            filters= cluster_imp_sorted_filters#[0] #Methods 1,2,3,4
            #filters=sorted_filters #Method_original_l1_norm
            #filters= filters #Method_original_random

            ele=[]
            flattened_filters = [item for sublist in filters for item in sublist]

            #_____Method1______
            '''ele= flattened_filters[math.ceil(rem):]'''

            #________Method2____
            retain_indexes=[]
            index_count=0

            for sub_filters in filters:
                #print(rem)
                if(rem>0):
                   if(rem < len(sub_filters)):
                      index_count= index_count + rem
                      rem= rem - len(sub_filters[:rem])
                      #print('1...',index_count,'....',rem)
                      break
                      
                   elif(rem >= len(sub_filters)):
                      if(len(sub_filters)==1):
                          retain_indexes.extend([sub_filters[-1]])
                          index_count= index_count + len(sub_filters)
                          #print('3...',index_count,'....',rem)
                      else:
                          retain_indexes.append(sub_filters[-1])
                          rem= rem - len(sub_filters)+1
                          index_count= index_count + len(sub_filters)
                          #print('2...',index_count,'....',rem)

            retain_indexes.extend(flattened_filters[index_count:])
            ele= retain_indexes



            #_____Method3_____
            '''retain_indexes=[]
            index_count=0

            for sub_filters in filters:
                #print(rem)
                if(rem>0):
                   if(rem < len(sub_filters)):
                      #retain_indexes.extend(sub_filters[rem:])
                      index_count= index_count + len(sub_filters)
                      rem= rem - len(sub_filters)
                      print('1...',index_count,'....',rem)
                      break
                      
                   elif(rem >= len(sub_filters)):
                      if(len(sub_filters)==1):
                          #retain_indexes.extend([sub_filters[-1]])
                          rem=rem-1
                          index_count= index_count + len(sub_filters)
                          print('3...',index_count,'....',rem)
                      else:
                          #retain_indexes.append(sub_filters[-1])
                          rem= rem - len(sub_filters)
                          index_count= index_count + len(sub_filters)
                          print('2...',index_count,'....',rem)

            retain_indexes.extend(flattened_filters[index_count:])
            ele= retain_indexes'''


            #_______Method4____
            '''retain_indexes=[]
            index_count=0

            for sub_filters in filters:
                #print(rem)
                if(rem>0):
                   if(rem < len(sub_filters)):
                     
                      retain_indexes.extend([sub_filters[-1]])
                      index_count= index_count + len(sub_filters)
                      rem= rem - len(sub_filters)
                      print('1...',index_count,'....',rem)
                      break
                     
                   elif(rem >= len(sub_filters)):
                      if(len(sub_filters)==1):
                          retain_indexes.extend([sub_filters[-1]])
                          #rem=rem-1
                          index_count= index_count + len(sub_filters)
                          print('3...',index_count,'....',rem)
                      else:
                          retain_indexes.append(sub_filters[-1])
                          rem= rem - len(sub_filters) +1
                          index_count= index_count + len(sub_filters)
                          print('2...',index_count,'....',rem)

            retain_indexes.extend(flattened_filters[index_count:])
            ele= retain_indexes'''


            #_____original_l1norm_______
            '''for j in range(len(filters)):
                #print('------',weight[j].shape)
                #print('total in cluster=',len(filters[j]),'..........',filters[j],' norms=', l1_norms[j])
                if(len(filters[j])==1):
                  cut=0
                else:
                  cut= math.ceil(len(filters[j])/2) #prune 25% only

                retain_filters= filters[j][cut:]
                ele.extend(retain_filters)'''

            #_______original_random_______
            '''for j in range(len(filters)):

                if(len(filters[j])==1):
                  cut=0
                else:
                  cut= math.ceil(len(filters[j])/4) #prune 25% only

                retain_filters= filters[j][cut:]
                ele.extend(retain_filters)'''




            first_ele= ele

        

        #print('first ele.....',first_ele)
        return n_clusters_, first_ele
