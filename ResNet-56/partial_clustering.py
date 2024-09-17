import random
import torch as th
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from new_Resnet import resnet_56
import csv
from itertools import zip_longest
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from torch.optim.lr_scheduler import MultiStepLR
from thop import profile
seed =1787
random.seed(seed)
import os
os.environ['PYTHONHASHSEED'] = str(seed)
th.manual_seed(seed)
th.cuda.manual_seed(seed)
th.cuda.manual_seed_all(seed)
th.backends.cudnn.deterministic = True
th.cuda.set_device(0)
N = 1

batch_size_tr = 100
batch_size_te = 100
epochs = 182


threshold= [0.253]*9+[0.223]*9+[0.20]*9  
new_epochs=90
prune_value=[1]*9+[2]*9+[4]*9


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

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    trainloader = th.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    testloader = th.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=2) 



total_convs=9*3 #first conv also included
total_blocks=3
decision_count=th.ones((total_convs))

short=False
tr_size = 50000
te_size=10000


activation = 'relu'

#tr_size = 300
#te_size=300
#short=True

if gpu:
    model=resnet_56().cuda()
else:
    model=resnet_56()
#print(model)
folder_name=model.create_folders(total_convs)
logger=model.get_logger(folder_name+'logger.log')
#model.layer1[0].conv1= nn.Conv2d(10, 20, kernel_size=3, stride=1,padding=1, bias=False)

optimizer = th.optim.SGD(model.parameters(), lr=0.1,momentum=0.9, weight_decay=2e-4,nesterov=True)
#scheduler = MultiStepLR(optimizer, milestones=[80,150,240], gamma=0.1)
scheduler = MultiStepLR(optimizer, milestones=[91,136], gamma=0.1)
criterion = nn.CrossEntropyLoss()

#ans=input("load from pretrained model= (t)true or (f)false")
ans1='t'
if(ans1=='t'):
  checkpoint = th.load('epoch_180.pth')
  model.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  scheduler.load_state_dict(checkpoint['scheduler'])
  epoch_train_acc = checkpoint['train_acc']
  epoch_test_acc = checkpoint['test_acc']
  print('model loaded')

elif(ans1=='f'):

    best_train_acc=0
    best_test_acc=0

    for n in range(N):

        for epoch in range(epochs):

          train_acc=[]
          for batch_num, (inputs, targets) in enumerate(trainloader):
            if(batch_num==3 and short):
              break
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
          with th.no_grad():
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
            epoch_test_acc= (sum(test_acc)*100)/te_size
            '''if(epoch_test_acc > best_test_acc ):
              best_test_acc=epoch_test_acc
              best_train_acc=epoch_train_acc '''      

          print('\n---------------Epoch number: {}'.format(epoch),
                  '---Train accuracy: {}'.format(epoch_train_acc),
                  '----Test accuracy: {}'.format(epoch_test_acc),'--------------')
          scheduler.step()
          print(optimizer.param_groups[0]['lr'])
else:
   print('wrong ans entered')
   import sys
   sys.exit()

prunes=0

#_____________________Conv_layers_________________
a=[]
p=0
for layer_name, layer_module in model.named_modules():
  if(isinstance(layer_module, th.nn.Conv2d) and layer_name!='conv1' and layer_name.find('conv1')!=-1):
    #print(layer_name, threshold[p])
    p+=1
    a.append(layer_module)

macs_base=0
params_base=0
input = th.ones(1, 3, 32, 32).cuda()
macs_base, params_base = profile(model, inputs=(input, ))


macs, params = profile(model, inputs=(input, )) 
f_p= round(((1-((macs) / (macs_base)) )*100),2)
p_p= round(((1-(params / params_base))*100),2)


d=[]
for i in range(total_blocks):
      d.append(a[(i*9)+1].weight.shape[0]) #RESNET-56
d.append(epoch_train_acc)
d.append(epoch_test_acc)
d.append(f_p)
d.append(p_p)
with open(folder_name+'resnet56Prune.csv', 'a', newline='') as myfile:
          wr = csv.writer(myfile)
          command=model.get_writerow(total_blocks+2+2)
          eval(command)
myfile.close()

best_train_acc=epoch_train_acc
best_test_acc=epoch_test_acc

state = {'model': model.state_dict(),
          'train_acc': epoch_train_acc,
          'test_acc':epoch_test_acc,
          'optimizer':optimizer.state_dict(),
          'scheduler':scheduler.state_dict()}
#th.save(state,folder_name+'initial_pruning.pth')

logger.info('method 2.....\n')
decision=True
best_test_acc= 0.0
while(decision==True):

    logger.info('............Prunes={}'.format(prunes))
    with th.no_grad():

#_________________________PRUNING_EACH_CONV_LAYER__________________________     
 
      model.prune_filters(threshold,prunes,folder_name,prune_value,logger)

      #if(th.sum(decision_count)!=0):
      if(prunes==20):
          decision=False  
 
      prunes+=1

    

    d1=[]
    for i1 in range(total_blocks):
          d1.append(a[(i1*9)+1].weight.shape[0]) #RESNET-56
    logger.info(d1)
    
    print('new-model starts....for ',new_epochs,' epochs')
    
    optimizer = th.optim.SGD(model.parameters(), lr=0.01,momentum=0.9, weight_decay=2e-4,nesterov=True)
    scheduler = MultiStepLR(optimizer, milestones=[20,70], gamma=0.1)
    
    best_test_acc= 0.0
    for epoch in range(new_epochs):

          train_acc=[]
          test_acc=[]

          for batch_num, (inputs, targets) in enumerate(trainloader):

            if(batch_num==3 and short):
               break

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

    rem=50000
    for i in range(total_convs):
      if (a[(i)].weight.shape[0] <rem):
          rem=a[(i)].weight.shape[0]
      if (rem<=2):
          decision=False
        
    macs, params = profile(model, inputs=(input, )) 
    f_p= round(((1-((macs) / (macs_base)) )*100),2)
    p_p= round(((1-(params / params_base))*100),2)

    d=[]
    for i in range(total_blocks):     
      d.append(a[(i*9)+1].weight.shape[0]) #RESNET-56
    d.append(best_train_acc)
    d.append(best_test_acc)
    d.append(f_p)
    d.append(p_p)

    with open(folder_name+'resnet56Prune.csv', 'a', newline='') as myfile:
          wr = csv.writer(myfile)
          command=model.get_writerow(total_blocks+2+2)
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
