#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torchvision import models
import functools
from torch.nn import init

import os
import math
import time
import imageio
import natsort
import logging
import itertools
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


# In[2]:


class Config(object):
    def __init__(self):
        self.name = 'DCGAN'
        self.dataset_name = 'MNIST'
        self.output_path = './results/'
        
        self.num_workers = 8
        self.batch_size = 32
        self.num_epochs = 50
        self.ndf = 32
        self.ngf = 32
        self.nz = 100
        self.d_lr = 0.0002
        self.g_lr = 0.0002
        self.nc = 1
        self.fps = 5
        self.use_fixed = True
        self.weight_decay = 1e-4
        self.num_test_samples = 16
        self.num_r_samples = 16
        
        os.makedirs(self.output_path,exist_ok=True)
opt = Config()


# In[3]:


def get_data_loader(batch_size):
    # MNIST Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))])

    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)

    # Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


# In[5]:


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>



def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net
def get_norm_layer(norm_type='instance'):
   
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal',
             init_gain=0.02, no_antialias=False, no_antialias_up=False, gpu_ids=[], opt=None):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

   
    if netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=6, opt=opt)
    elif netG == 'resnet_4blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=4, opt=opt)
    
    return init_net(net, init_type, init_gain, gpu_ids, initialize_weights=('stylegan2' not in netG))


def get_filter(filt_size=3):
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt

def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer

class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='inst', activation='relu', pad_type='zero', nz=0):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, nz=nz)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)



class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(1),
                 nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if(no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
        model += [nn.ReflectionPad2d(1)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=3, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, layers=[], encode_only=False):
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            fake = self.model(input)
            return fake

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]


# In[6]:


input_nc=1
output_nc=1
ngf=32                 # of gen filters in the last conv layer
netG = 'resnet_6blocks' 
init_type ='xavier'    # choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
init_gain = 0.02       #'scaling factor for normal, xavier and orthogonal.')
no_antialias = True    #if specified, use stride=2 convs instead of antialiased-downsampling 
no_antialias_up = True #if specified, use [upconv(learned filter)] instead of [upconv(hard-coded [1,3,3,1] filter), conv]

netG=define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal',
             init_gain=0.02, no_antialias=False, no_antialias_up=False, gpu_ids=[], opt=None)


# In[7]:


def vgg_block(num_convs, in_channels, num_channels):
    layers=[]
    for i in range(num_convs):
        layers+=[nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=3, padding=2)]
        in_channels=num_channels
    layers +=[nn.ReLU()]
    layers +=[nn.MaxPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*layers)
 
class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.conv_arch=((1,1,64),(1,64,128),(2,128,256),(2,256,512),(2,512,512))
        layers=[]
        for (num_convs,in_channels,num_channels) in self.conv_arch:
            layers+=[vgg_block(num_convs,in_channels,num_channels)]
        self.features=nn.Sequential(*layers)
        self.dense1 = nn.Linear(512*4*4,4096)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(4096, 4096)
        self.drop2 = nn.Dropout(0.5)
        self.dense3 = nn.Linear(4096, 1)
 
    def forward(self,x):
        x=self.features(x)
        x=x.view(-1,512*4*4)
        x=self.dense3(self.drop2(F.relu(self.dense2(self.drop1(F.relu(self.dense1(x)))))))
        x=torch.sigmoid(x)
        return x
 


# In[8]:


train_loader = get_data_loader(opt.batch_size)


# In[9]:


train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []


# In[10]:


train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# In[11]:


def generate_images(epoch, path, fixed_noise, num_test_samples, netG, device, use_fixed):
    
    r_z = []
    for i in range(10):
        z = torch.randn(num_test_samples, 1, 28, 28, device=device)
        r_z.append(z)
        
    size_figure_grid = int(math.sqrt(num_test_samples))
    title = None
  
    generated_fake_images =[]
    if use_fixed:
        f_out = netG(fixed_noise)
        generated_fake_images.append(f_out)
        path += 'fixed_noise/'
        os.makedirs(path,exist_ok=True)
        title = 'Fixed Noise'
        
    if not use_fixed:
        for z_ in r_z:
            r_out = netG(z_)
            generated_fake_images.append(r_out)
        path += 'variable_noise/' 
        os.makedirs(path,exist_ok=True)
        title = 'Variable Noise'

    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(6,6))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i,j].get_xaxis().set_visible(False)
        ax[i,j].get_yaxis().set_visible(False)
     
    i=0
    for gen in generated_fake_images:
        for k in range(num_test_samples):
            i = k//4
            j = k%4
            ax[i,j].cla()
            ax[i,j].imshow(gen[k].data.cpu().numpy().reshape(28,28), cmap='Greys')

        if use_fixed:    
            label = 'Epoch_{}'.format(epoch+1)
            fig.text(0.5, 0.04, label, ha='center')
            fig.suptitle(title)
            fig.savefig(path+label+'.png')
        else:
            label = 'Epoch_{}_{}'.format(epoch+1,i)
            fig.text(0.5, 0.04, label, ha='center')
            fig.suptitle(title)
            fig.savefig(path+label+'.png')
            i+=1


# In[12]:


def save_gif(path, fps, fixed_noise=False):
    if fixed_noise==True:
        path += 'fixed_noise/'
    else:
        path += 'variable_noise/'
    images = glob(path + '*.png')
    images = natsort.natsorted(images)
    gif = []

    for image in images:
        gif.append(imageio.imread(image))
    imageio.mimsave(path+'animated.gif', gif, fps=fps)


# In[13]:


netG =netG.to(device)
netD = VGG().to(device)


# In[14]:


criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=opt.d_lr)
optimizerG = optim.Adam(netG.parameters(), lr=opt.g_lr)


# In[15]:


real_label = 1
fake_label = 0
num_batches = len(train_loader)
fixed_noise = torch.randn(opt.num_test_samples, 1, 28, 28, device=device)


# In[ ]:


try:
    state = torch.load(os.path.join('results','kl', 'models.pth'),map_location='cuda:1')
    netG.load_state_dict(state['g_state'])
    print("Pretained models loaded successfully")
except:
    print("Training from scratch")


num_iter = 0
D_losses = []
G_losses  = []
for epoch in range(opt.num_epochs):
    for i, (real_images, _) in enumerate(train_loader):
        bs = real_images.shape[0]
        y_real = torch.ones(bs).to(device)
        y_fake = torch.zeros(bs).to(device)  
        ##############################
        #   Training discriminator   #
        ##############################

        netD.zero_grad()
        real_images = real_images.to(device)
#         label = torch.full((bs,), real_label, device=device)

        output = netD(real_images).squeeze()
        lossD_real = criterion(output, y_real)
        lossD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(bs, 1, 28, 28, device=device)
        fake_images = netG(noise)
        
#         label.fill_(fake_label)
        output = netD(fake_images.detach()).squeeze()
        lossD_fake = criterion(output, y_fake)
        lossD_fake.backward()
        D_G_z1 = output.mean().item()
        lossD = lossD_real + lossD_fake
        optimizerD.step()
        D_losses.append(lossD.item())

        ##########################
        #   Training generator   #
        ##########################

        netG.zero_grad()
#         label.fill_(real_label)
        fake_images = netG(noise)
        output = netD(fake_images.detach()).squeeze()
        lossG = criterion(output, y_real)
        lossG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        num_iter += 1
        G_losses.append(lossG.item())
        if (i+1)%300 == 0:
            print('Epoch [{}/{}], step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, Discriminator - D(G(x)): {:.2f}, Generator - D(G(x)): {:.2f}'.format(epoch+1, opt.num_epochs, 
                                                        i+1, num_batches, lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))
        netG.eval()
        generate_images(epoch, opt.output_path, fixed_noise, opt.num_test_samples, netG, device, use_fixed=opt.use_fixed)
        generate_images(epoch, opt.output_path, fixed_noise, opt.num_r_samples, netG, device, use_fixed=False)
        netG.train()
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))

    # Save gif:
    save_gif(opt.output_path, opt.fps, fixed_noise=opt.use_fixed)
    
    state = {
                'epoch': epoch,
                'g_state': netG.state_dict(),
                'd_state': netD.state_dict(),
                }
    torch.save(state, "./results/dc_gan.pth")
    if epoch == 0:
        torch.save(state, "./results/models_0.pth")
    if epoch == opt.num_epochs/2:
        torch.save(state, "./results/models_n/2.pth")
    

show_train_hist(train_hist, save=True, path='./results/MNIST_DCGAN_train_hist.png')


# In[ ]:




