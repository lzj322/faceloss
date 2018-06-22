# -*- coding: utf-8 -*-
"""
Training a classifier
=====================

This is it. You have seen how to define neural networks, compute loss and make
updates to the weights of the network.

Now you might be thinking,

What about data?
----------------

Generally, when you have to deal with image, text, audio or video data,
you can use standard python packages that load data into a numpy array.
Then you can convert this array into a ``torch.*Tensor``.

-  For images, packages such as Pillow, OpenCV are useful.
-  For audio, packages such as scipy and librosa
-  For text, either raw Python or Cython based loading, or NLTK and
   SpaCy are useful.

Specifically for ``vision``, we have created a package called
``torchvision``, that has data loaders for common datasets such as
Imagenet, CIFAR10, MNIST, etc. and data transformers for images, viz.,
``torchvision.datasets`` and ``torch.utils.data.DataLoader``.

This provides a huge convenience and avoids writing boilerplate code.

For this tutorial, we will use the CIFAR10 dataset.
It has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’,
‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in CIFAR-10 are of
size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.

.. figure:: /_static/img/cifar10.png
   :alt: cifar10

   cifar10


Training an image classifier
----------------------------

We will do the following steps in order:

1. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Convolution Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data

1. Loading and normalizing CIFAR10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using ``torchvision``, it’s extremely easy to load CIFAR10.
"""
import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from tensorboardX import SummaryWriter
import net_sphere

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]

parser = argparse.ArgumentParser(description='PyTorch cifar10 test experiment')
parser.add_argument('--net','-n',default='resnet',type=str,choices=['resnet','naive'])
parser.add_argument('--loss',default='softmax',type=str,choices=['softmax','sphere','cosface'])
parser.add_argument('--lr',default=0.01,type=float)
parser.add_argument('--test_only','-t',action='store_true',help='only test')
parser.add_argument('--log_to_file',action='store_true',help='log file')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--ckpt', type=str,default='epoch_29_ckpt_95.72.pth')
parser.add_argument('--resume', type=str, default=False)
parser.add_argument('--exp_dir', type=str, default='')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=30)

args=parser.parse_args()
if args.net == 'resnet':
    transform = transforms.Compose(
        [transforms.Resize(224),
        transforms.ToTensor(),  
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
elif args.net == 'naive':
    transform = transforms.Compose(
        [transforms.ToTensor(),  
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
else:
    print ('error')
trainset = torchvision.datasets.CIFAR10(root='/root/Cloud/data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.workers)

testset = torchvision.datasets.CIFAR10(root='/root/Cloud/data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.workers)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########################################################################
# Let us show some of the training images, for fun.

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image

def may_make_dir(path):
  if path in [None, '']:
    return
  if not os.path.exists(path):
    os.makedirs(path)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=10):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)

class AverageMeter(object):
  """Modified from Tong Xiao's open-reid. 
  Computes and stores the average and current value"""

  def __init__(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = float(self.sum) / (self.count + 1e-20)

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


########################################################################
# 2. Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

model = args.net
if args.exp_dir !='':
    exp_dir = args.exp_dir
    may_make_dir(exp_dir)
else:
    exp_dir = os.path.join('./output',args.net)
    may_make_dir(exp_dir)

# model = 'Naive'


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 =   nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16,32,3)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32,32,1)
        if args.loss == 'softmax':
            self.fc3 = nn.Linear(32, 10)
        elif args.loss == 'sphere':
            self.fc3 = net_sphere.AngleLinear(32,10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        im_size = x.size()
        x = F.avg_pool2d(x,(im_size[-2],im_size[-1]))
        x = F.relu(self.conv4(x))
        x=x.view(-1,32)
        # print(x.size())
        x = self.fc3(x)
        return x

class resnet(nn.Module):
    def __init__(self,model):
        super(resnet,self).__init__()
        self.resnet_layer =nn.Sequential(*list(model.children())[:-1])
    def forward(self,x):
        x = self.resnet_layer(x)
        x = x.view(-1,num_fc)
        return x



if args.net=='naive':

    net=Net().cuda()
    initialize_weights(net)
elif args.net=='resnet':

    basenet = models.resnet18(pretrained=False)
    num_fc = basenet.fc.in_features
    root = '/root/Cloud/Pytorch_Pretrained/'
    res18_path = os.path.join(root, 'ResNet', 'resnet18-5c106cde.pth')
    basenet.load_state_dict(torch.load(res18_path))
    net = resnet(basenet)
    net = net.cuda()



########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum

import torch.optim as optim

if args.loss == 'sphere':
    criterion = net_sphere.AngleLoss()
else :
    criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
# ckpt_file = 'epoch_29_ckpt.pth'
# resume = False


########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize
def main():

    if args.resume:
        if args.ckpt_file != '':
            ckpt=torch.load(args.ckpt_file)
            net.load_state_dict(ckpt['state_dict'])
            curr_epoch=ckpt['epoch']+1
    else:
        curr_epoch = 0
            
    if args.log_to_file:
        writer = SummaryWriter(log_dir=os.path.join(exp_dir,'tensorboard'))

    train_loss = AverageMeter()
    for epoch in range(curr_epoch,curr_epoch+args.epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        exp_lr_scheduler(optimizer, epoch, init_lr=args.lr, lr_decay_epoch=10)
        for batch, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            if args.loss == 'softmax':
                net.fc = nn.Linear(num_fc,10).cuda()
                outputs = net.fc(outputs)
            elif args.loss == 'sphere':
                net.fc = net_sphere.AngleLinear(num_fc,10).cuda()
                outputs = net.fc(outputs)
            elif args.loss =='cosface':
                net.fc = net_sphere.MarginCosineProduct(num_fc,10).cuda()
                outputs = net.fc(outputs,labels)


            # print (outputs.size())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            train_loss.update(running_loss)
            if (batch+1) % 50 == 0:    # print every 1000 mini-batches
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, batch + 1, running_loss / 50))
                running_loss = 0.0
        if epoch>1:
            os.remove(os.path.join(exp_dir,'{}_{}_epoch_{}_ckpt.pth'.format(args.loss,args.net,epoch-1)))
        snapshot_name = '{}_{}_epoch_{}_ckpt.pth'.format(args.loss,args.net,epoch)
        torch.save({'epoch':epoch,'state_dict':net.state_dict()},os.path.join(exp_dir,snapshot_name))
        if args.log_to_file:
            writer.add_scalar('train_acc',train_loss.val,epoch)
    print('Finished Training')


    test(test_only=False)
########################################################################
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.

# dataiter = iter(testloader)
# images, labels = dataiter.next()
# # images = images.cuda()
# # labels = labels.cuda()
# # print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:

# outputs = net(Variable(images).cuda())

########################################################################
# The outputs are energies for the 10 classes.
# Higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
# _, predicted = torch.max(outputs.data, 1)



########################################################################
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.
def test(test_only=False):
    if args.test_only:
        ckpt = torch.load(os.path.join(exp_dir,args.ckpt))
        net.load_state_dict(ckpt['state_dict'])


    net.eval()
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images = images.cuda()
        labels = labels
        outputs = net(Variable(images).cuda())
        if args.loss == 'softmax':
            _, predicted = torch.max(outputs.data.cpu(), 1)
        elif args.loss == 'sphere':
            _, predicted = torch.max(outputs[0].data.cpu(), 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    test_acc = 100.0 * correct / total
    print('Accuracy of the network on the 10000 test images: %.3f %%' % (
        test_acc))


    ########################################################################
    # That looks waaay better than chance, which is 10% accuracy (randomly picking
    # a class out of 10 classes).
    # Seems like the network learnt something.
    #
    # Hmmm, what are the classes that performed well, and the classes that did
    # not perform well:

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images).cuda())
        if args.loss == 'softmax':
            _, predicted = torch.max(outputs.data.cpu(), 1)
        elif args.loss == 'sphere':
            _, predicted = torch.max(outputs[0].data.cpu(), 1)
        # print (type(predicted),type(labels))
        c = (predicted == labels).squeeze()
        for i in range(10):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %.3f %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    if args.test_only:
        test(test_only=True)
    else:
        main()
    
########################################################################
# Okay, so what next?
#
# How do we run these neural networks on the GPU?
#
# Training on GPU
# ----------------
# Just like how you transfer a Tensor on to the GPU, you transfer the neural
# net onto the GPU.
# This will recursively go over all modules and convert their parameters and
# buffers to CUDA tensors:
#
# .. code:: python
#
#     net.cuda()
#
#
# Remember that you will have to send the inputs and targets at every step
# to the GPU too:
#
# ::
#
#         inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
#
# Why dont I notice MASSIVE speedup compared to CPU? Because your network
# is realllly small.
#
# **Exercise:** Try increasing the width of your network (argument 2 of
# the first ``nn.Conv2d``, and argument 1 of the second ``nn.Conv2d`` –
# they need to be the same number), see what kind of speedup you get.
#
# **Goals achieved**:
#
# - Understanding PyTorch's Tensor library and neural networks at a high level.
# - Train a small neural network to classify images
#
# Where do I go next?
# -------------------
#
# -  `Train neural nets to play video games`_
# -  `Train a state-of-the-art ResNet network on imagenet`_
# -  `Train an face generator using Generative Adversarial Networks`_
# -  `Train a word-level language model using Recurrent LSTM networks`_
# -  `More examples`_
# -  `More tutorials`_
# -  `Discuss PyTorch on the Forums`_
# -  `Chat with other users on Slack`_
#
# .. _Train neural nets to play video games: https://goo.gl/uGOksc
# .. _Train a state-of-the-art ResNet network on imagenet: https://github.com/pytorch/examples/tree/master/imagenet
# .. _Train an face generator using Generative Adversarial Networks: https://github.com/pytorch/examples/tree/master/dcgan
# .. _Train a word-level language model using Recurrent LSTM networks: https://github.com/pytorch/examples/tree/master/word_language_model
# .. _More examples: https://github.com/pytorch/examples
# .. _More tutorials: https://github.com/pytorch/tutorials
# .. _Discuss PyTorch on the Forums: https://discuss.pytorch.org/
# .. _Chat with other users on Slack: http://pytorch.slack.com/messages/beginner/
