import random
import torch as th
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from VGG16 import VGG16
import csv
from itertools import zip_longest
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from matplotlib.ticker import PercentFormatter
from torch.optim.lr_scheduler import MultiStepLR

seed =1787
random.seed(seed)
import os
os.environ['PYTHONHASHSEED'] = str(seed)
th.manual_seed(seed)
th.cuda.manual_seed(seed)
th.cuda.manual_seed_all(seed)
th.backends.cudnn.deterministic = True
os.environ["CUDA_VISIBLE_DEVICES"]="1"




N = 1
batch_size_tr = 100
batch_size_te = 100
epochs = 2#300

threshold= 0.46
new_epochs=90#90-custom_epochs
prune_percentage=[0.02]*2+[0.04]*2+[0.05]*3+[0.1]*6

#th.cuda.set_device(0)
gpu = th.cuda.is_available()

if not gpu:
    print('qqqq')

else:

    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = th.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = th.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=2) 


total_layers=15
total_convs=13
total_blocks=total_convs

decision_count=th.ones((total_convs))

short=False
tr_size = 50000
te_size=10000


activation = 'relu'

#tr_size = 300
#te_size=300
#short=True

if gpu:
    model=VGG16(10,activation).cuda()
else:
    model=VGG16(10,activation)

folder_name=model.create_folders(total_convs)
logger=model.get_logger(folder_name+'logger.log')

optimizer = th.optim.SGD(model.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[80,140,230], gamma=0.1)
criterion = nn.CrossEntropyLoss()

#ans=input("load from pretrained model= (t)true or (f)false")
ans='t'
if(ans=='t'):
  checkpoint = th.load('epoch_295.pth')
  model.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  scheduler.load_state_dict(checkpoint['scheduler'])
  epoch_train_acc = checkpoint['train_acc']
  epoch_test_acc = checkpoint['test_acc']
  print('loading completed')

elif(ans=='f'):

    best_train_acc=0
    best_test_acc=0
    for n in range(N):

        #current_iteration = 0
        mi_iteration=0
        for epoch in range(epochs):
          #start=current_iteration
          train_acc=[]
          for batch_num, (inputs, targets) in enumerate(trainloader):
            if(batch_num==3 and short):
              break
            if(batch_num%100==0):
              print(batch_num)
            inputs = inputs.cuda()
            targets = targets.cuda()
            model.train()
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            with th.no_grad():
              y_hat = th.argmax(output, 1)
              score = th.eq(y_hat, targets).sum()
              train_acc.append(score.item())
              #current_iteration += 1
          
          with th.no_grad():
            #model.train_accuracy.append((sum(train_acc)*100)/tr_size) 
            epoch_train_acc=  (sum(train_acc)*100)/tr_size        
            test_acc=[]
            model.eval()
            for batch_nums, (inputs2, targets2) in enumerate(testloader):
                if(batch_nums==3 and short):
                    break

                inputs2, targets2 = inputs2.cuda(), targets2.cuda()            
                output=model(inputs2)
                y_hat = th.argmax(output, 1)
                score = th.eq(y_hat, targets2).sum()
                test_acc.append(score.item())

            #model.test_accuracy.append((sum(test_acc)*100)/te_size)
            epoch_test_acc= (sum(test_acc)*100)/te_size
            '''if(epoch_test_acc > best_test_acc ):
              best_test_acc=epoch_test_acc
              best_train_acc=epoch_train_acc '''      

          #end=current_iteration
          print('\n---------------Epoch number: {}'.format(epoch),
                  '---Train accuracy: {}'.format(epoch_train_acc),
                  '----Test accuracy: {}'.format(epoch_test_acc),'--------------')
          scheduler.step()
          print(optimizer.param_groups[0]['lr'])
else:
   print('wrong ans entered')
   import sys
   sys.exit()

#ended_epoch=epoch
#ended_iteration=end
prunes=0

#_____________________Conv_layers_________________
a=[]
for layer_name, layer_module in model.named_modules():
  if(isinstance(layer_module, th.nn.Conv2d)):
    a.append(layer_module)



d=[]
for i in range(total_convs):
      d.append(a[i].weight.shape[0])

d.append(epoch_train_acc)
d.append(epoch_test_acc)


with open(folder_name+'vggPrune.csv', 'a', newline='') as myfile:
          wr = csv.writer(myfile)
          command=model.get_writerow(total_convs+2)
          eval(command)
myfile.close()

ended_epoch=0
best_train_acc=epoch_train_acc
best_test_acc=epoch_test_acc

state = {'model': model.state_dict(),
          'train_acc': best_train_acc,
          'test_acc':best_test_acc,
          'optimizer':optimizer.state_dict(),
          'scheduler':scheduler.state_dict()}
#th.save(state,folder_name+'initial_pruning.pth')

logger.info('method 2.....\n')
decision=True
best_test_acc=0.0


'''with th.no_grad():
   final_first_ele =[]
   final_n_clusters=[]
   for layer_name, layer_module in model.named_modules():
          
          if(isinstance(layer_module, th.nn.Conv2d)):
 
                  #--------clustering to find indices-------
                  
                  clusters, ele = model.cluster_weights_agglo(layer_module.weight, threshold)

                  final_clusters.append(clusters)               
                  final_first_ele.append(ele)'''


while(decision==True):


    with th.no_grad():
      
      #_________________________PRUNING_EACH_CONV_LAYER__________________________     
 
      model.prune_filters(threshold,prunes,folder_name,prune_percentage)
            

      #if(th.sum(decision_count)!=0):
      if(prunes== 0):
          decision=False  
 
      prunes+=1

    d1=[]
    for i1 in range(total_convs):
      d1.append(a[i1].weight.shape[0])
    logger.info(d1)

    print('new-model starts....for ',new_epochs,' epochs')
    #print(model)

    optimizer = th.optim.SGD(model.parameters(), lr=0.01,momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[40,70,90], gamma=0.1)
    #optimizer = th.optim.SGD(model.parameters(), lr=0.001,momentum=0.9, weight_decay=5e-4)
    #scheduler = MultiStepLR(optimizer, milestones=[40,70,90], gamma=0.1)
    
    best_test_acc=0.0

    for epoch in range(new_epochs):

          train_acc=[]
          test_acc=[]

          for batch_num, (inputs, targets) in enumerate(trainloader):

            if(batch_num==3 and short):
               break

            inputs = inputs.cuda()
            targets = targets.cuda()
            model= model.cuda()
            model.train()            
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            
            with th.no_grad():

               y_hat = th.argmax(output, 1)
               score = th.eq(y_hat, targets).sum()
               train_acc.append(score.item())
     

          with th.no_grad():            


              epoch_train_acc=(sum(train_acc)*100)/tr_size
              model.eval() 
          
              for batch_idx2, (inputs2, targets2) in enumerate(testloader):
                if(batch_idx2==3 and short):
                    break
                inputs2, targets2 = inputs2.cuda(), targets2.cuda()
                output=model(inputs2)
                y_hat = th.argmax(output, 1)
                score = th.eq(y_hat, targets2).sum()
                test_acc.append(score.item())

              epoch_test_acc=(sum(test_acc)*100)/te_size
              if(epoch_test_acc > best_test_acc ):
                      best_test_acc=epoch_test_acc
                      best_train_acc=epoch_train_acc
                      state = {'model': model.state_dict(),
                               'train_acc': best_train_acc,
                               'test_acc':best_test_acc,
                               'optimizer':optimizer.state_dict(),
                               'scheduler':scheduler.state_dict()}
                      th.save(state,folder_name+str(prunes)+'.pth')

          #print(optimizer.param_groups[0]['lr'])
          scheduler.step()
          
          logger.info('Epoch: {}/{}---Train:{:.3f}----Test: {:.3f}\n'.format(epoch,new_epochs-1,epoch_train_acc,epoch_test_acc))


    #----------------writing data-----------
    ended_epoch=ended_epoch+new_epochs

    d=[]
    for i in range(total_convs):
      d.append(a[i].weight.shape[0])
    d.append(best_train_acc)
    d.append(best_test_acc)
    with open(folder_name+'vggPrune.csv', 'a', newline='') as myfile:
          wr = csv.writer(myfile)
          command=model.get_writerow(total_convs+2)
          eval(command)

    myfile.close()
    #-------------------------end writing data---------------

from zipfile import ZipFile
import os,glob

directory = os.path.dirname(os.path.realpath(__file__)) #location of running file
file_paths = []
os.chdir(directory)
for filename in glob.glob("*.py"):
	filepath = os.path.join(directory, filename)
	file_paths.append(filepath)
	#print(filename)

print('Following files will be zipped:')
for file_name in file_paths:
	print(file_name)
saving_loc = folder_name #location of results
os.chdir(saving_loc)
# writing files to a zipfile
with ZipFile('python_files.zip','w') as zip:
	# writing each file one by one
	for file in file_paths:
		zip.write(file)
