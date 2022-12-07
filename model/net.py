import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
import math


class ResBase50(nn.Module):
    def __init__(self):
        super(ResBase50, self).__init__()
        model_resnet50 = models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x
    
class _BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_BatchNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(self.num_features).cuda())
            self.bias = nn.Parameter(torch.Tensor(self.num_features).cuda())
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(self.num_features).cuda())
        self.register_buffer('running_var', torch.ones(self.num_features).cuda())
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def forward(self, input):
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training, self.momentum, self.eps)


class ResNet50_mod_name(nn.Module):
    def __init__(self):
        super(ResNet50_mod_name, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        
        
        self.conv111 = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False)
        self.bn111 = nn.BatchNorm2d(64)
        self.conv112 = nn.Conv2d(64, 64, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1)
        self.bn112 = nn.BatchNorm2d(64)
        self.conv113 = nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
        self.bn113 = nn.BatchNorm2d(256)
        # Downsample
        self.conv114 = nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
        self.bn114 = nn.BatchNorm2d(256)
        
        self.conv121 = nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False)
        self.bn121 = nn.BatchNorm2d(64)
        self.conv122 = nn.Conv2d(64, 64, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1)
        self.bn122 = nn.BatchNorm2d(64)
        self.conv123 = nn.Conv2d(64, 256, kernel_size=1, stride=1, groups=1, dilation=1, bias=False)
        self.bn123 = nn.BatchNorm2d(256)
        
        self.conv131 = nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False)
        self.bn131 = nn.BatchNorm2d(64)
        self.conv132 = nn.Conv2d(64, 64, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1)
        self.bn132 = nn.BatchNorm2d(64)
        self.conv133 = nn.Conv2d(64, 256, kernel_size=1, stride=1, groups=1, dilation=1, bias=False)
        self.bn133 = nn.BatchNorm2d(256)
        
        
        
        self.conv211 = nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False)
        self.bn211 = nn.BatchNorm2d(128)
        self.conv212 = nn.Conv2d(128, 128, kernel_size=3, stride=2, groups=1, dilation=1, bias=False, padding=1)
        self.bn212 = nn.BatchNorm2d(128)
        self.conv213 = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.bn213 = nn.BatchNorm2d(512)
        # Downsample
        self.conv214 = nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False)
        self.bn214 = nn.BatchNorm2d(512)
              
        self.conv221 = nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False)
        self.bn221 = nn.BatchNorm2d(128)
        self.conv222 = nn.Conv2d(128, 128, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1)
        self.bn222 = nn.BatchNorm2d(128)
        self.conv223 = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.bn223 = nn.BatchNorm2d(512)  
        
        self.conv231 = nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False)
        self.bn231 = nn.BatchNorm2d(128)
        self.conv232 = nn.Conv2d(128, 128, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1)
        self.bn232 = nn.BatchNorm2d(128)
        self.conv233 = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.bn233 = nn.BatchNorm2d(512) 
        
        self.conv241 = nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False)
        self.bn241 = nn.BatchNorm2d(128)
        self.conv242 = nn.Conv2d(128, 128, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1)
        self.bn242 = nn.BatchNorm2d(128)
        self.conv243 = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.bn243 = nn.BatchNorm2d(512) 
        
  

        self.conv311 = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False)
        self.bn311 = nn.BatchNorm2d(256)
        self.conv312 = nn.Conv2d(256, 256, kernel_size=3, stride=2, groups=1, dilation=1, bias=False, padding=1)
        self.bn312 = nn.BatchNorm2d(256)
        self.conv313 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.bn313 = nn.BatchNorm2d(1024)
        # Downsample
        self.conv314 = nn.Conv2d(512, 1024, kernel_size=1, stride=2, bias=False)
        self.bn314 = nn.BatchNorm2d(1024)
              
        self.conv321 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.bn321 = nn.BatchNorm2d(256)
        self.conv322 = nn.Conv2d(256, 256, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1)
        self.bn322 = nn.BatchNorm2d(256)
        self.conv323 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.bn323 = nn.BatchNorm2d(1024)  
        
        self.conv331 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.bn331 = nn.BatchNorm2d(256)
        self.conv332 = nn.Conv2d(256, 256, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1)
        self.bn332 = nn.BatchNorm2d(256)
        self.conv333 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.bn333 = nn.BatchNorm2d(1024) 
        
        self.conv341 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.bn341 = nn.BatchNorm2d(256)
        self.conv342 = nn.Conv2d(256, 256, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1)
        self.bn342 = nn.BatchNorm2d(256)
        self.conv343 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.bn343 = nn.BatchNorm2d(1024) 
        
        self.conv351 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.bn351 = nn.BatchNorm2d(256)
        self.conv352 = nn.Conv2d(256, 256, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1)
        self.bn352 = nn.BatchNorm2d(256)
        self.conv353 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.bn353 = nn.BatchNorm2d(1024) 
        
        self.conv361 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.bn361 = nn.BatchNorm2d(256)
        self.conv362 = nn.Conv2d(256, 256, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1)
        self.bn362 = nn.BatchNorm2d(256)
        self.conv363 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.bn363 = nn.BatchNorm2d(1024)
        
        
        
        self.conv411 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False)
        self.bn411 = nn.BatchNorm2d(512)
        self.conv412 = nn.Conv2d(512, 512, kernel_size=3, stride=2, groups=1, dilation=1, bias=False, padding=1)
        self.bn412 = nn.BatchNorm2d(512)
        self.conv413 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False)
        self.bn413 = nn.BatchNorm2d(2048)
        # Downsample
        self.conv414 = nn.Conv2d(1024, 2048, kernel_size=1, stride=2, bias=False)
        self.bn414 = nn.BatchNorm2d(2048)
              
        self.conv421 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False)
        self.bn421 = nn.BatchNorm2d(512)
        self.conv422 = nn.Conv2d(512, 512, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1)
        self.bn422 = nn.BatchNorm2d(512)
        self.conv423 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False)
        self.bn423 = nn.BatchNorm2d(2048)  
        
        self.conv431 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False)
        self.bn431 = nn.BatchNorm2d(512)
        self.conv432 = nn.Conv2d(512, 512, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1)
        self.bn432 = nn.BatchNorm2d(512)
        self.conv433 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False)
        self.bn433 = nn.BatchNorm2d(2048) 

        # self.conv431vae = nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False)
        # self.bn431vae = nn.BatchNorm2d(512)
        # self.conv432vae = nn.Conv2d(512, 512, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1)
        # self.bn432vae = nn.BatchNorm2d(512)
        # self.conv433vae = nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False)
        # self.bn433vae = nn.BatchNorm2d(2048) 
        
        self.unpool = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.unsample = nn.Upsample(size=7, mode='nearest')

        self.indices1 = []
    
    def decode(self, mu_s, mu_t, is_rep = False):
        # reparameterize
        if is_rep:
            std = torch.exp(0.5*mu_t)
            eps = torch.randn_like(std)
            z = mu_s + eps*std
            x = z
            x = self.unsample(x)
        else:
            x = mu_t
            x = self.unsample(x)
            

        #Layer 4 Deconv
        identity = x
        bn_de_512_10 = _BatchNorm(512)
        bn_de_512_11 = _BatchNorm(512)
        bn_de_2048_2 = _BatchNorm(2048)
        de_conv433 = nn.ConvTranspose2d(2048,512, kernel_size=1, stride=1, bias=False).cuda()
        de_conv432 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1).cuda()
        de_conv431 = nn.ConvTranspose2d(512, 2048, kernel_size=1, stride=1, bias=False).cuda()
        out = self.relu(bn_de_512_10(de_conv433(x)))
        out = self.relu(bn_de_512_11(de_conv432(out)))      
        out = bn_de_2048_2(de_conv431(out))
        out += identity
        x = self.relu(out)
        identity = x
        bn_de_512_8 = _BatchNorm(512)
        bn_de_512_9 = _BatchNorm(512)
        bn_de_2048_1 = _BatchNorm(2048)            
        de_conv423 = nn.ConvTranspose2d(2048, 512, kernel_size=1, stride=1, bias=False).cuda()
        de_conv422 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1).cuda()
        de_conv421 = nn.ConvTranspose2d(512, 2048, kernel_size=1, stride=1, bias=False).cuda()
        out = self.relu(bn_de_512_8(de_conv423(x)))
        out = self.relu(bn_de_512_9(de_conv422(out)))            
        out = bn_de_2048_1(de_conv421(out))
        out += identity
        x = self.relu(out)   
        identity = x
        bn_de_512_6 = _BatchNorm(512)
        bn_de_512_7 = _BatchNorm(512)
        bn_de_1024_6 = _BatchNorm(1024)
        bn_de_1024_7 = _BatchNorm(1024)
        de_conv414 = nn.ConvTranspose2d(2048, 1024, kernel_size=1, stride=2, bias=False, output_padding = 1).cuda()
        de_conv413 = nn.ConvTranspose2d(2048, 512, kernel_size=1, stride=1, bias=False).cuda()
        de_conv412 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, groups=1, dilation=1, bias=False, padding=1, output_padding = 1).cuda()
        de_conv411 = nn.ConvTranspose2d(512, 1024, kernel_size=1, stride=1, bias=False).cuda()
        out = self.relu(bn_de_512_6(de_conv413(x)))
        out = self.relu(bn_de_512_7(de_conv412(out)))            
        out = bn_de_1024_6(de_conv411(out))
        identity = bn_de_1024_7(de_conv414(x))
        out += identity
        x = self.relu(out)         
        #print ('size after deconv layer 4: ', x.size())


        #Layer 3 Deconv
        identity = x
        bn_de_256_15 = _BatchNorm(256)
        bn_de_256_16 = _BatchNorm(256)
        bn_de_1024_5 = _BatchNorm(1024)
        de_conv363 = nn.ConvTranspose2d(1024, 256, kernel_size=1, stride=1, bias=False).cuda()
        de_conv362 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1).cuda()
        de_conv361 = nn.ConvTranspose2d(256, 1024, kernel_size=1, stride=1, bias=False).cuda()
        out = self.relu(bn_de_256_15(de_conv363(x)))
        out = self.relu(bn_de_256_16(de_conv362(out)))            
        out = bn_de_1024_5(de_conv361(out))
        out += identity
        x = self.relu(out)     
        
        identity = x
        bn_de_256_13 = _BatchNorm(256)
        bn_de_256_14 = _BatchNorm(256)
        bn_de_1024_4 = _BatchNorm(1024)
        de_conv353 = nn.ConvTranspose2d(1024, 256, kernel_size=1, stride=1, bias=False).cuda()
        de_conv352 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1).cuda()
        de_conv351 = nn.ConvTranspose2d(256, 1024, kernel_size=1, stride=1, bias=False).cuda()
        out = self.relu(bn_de_256_13(de_conv353(x)))
        out = self.relu(bn_de_256_14(de_conv352(out)))            
        out = bn_de_1024_4(de_conv351(out))
        out += identity
        x = self.relu(out)  
        
        identity = x
        bn_de_256_11 = _BatchNorm(256)
        bn_de_256_12 = _BatchNorm(256)
        bn_de_1024_3 = _BatchNorm(1024)
        de_conv343 = nn.ConvTranspose2d(1024, 256, kernel_size=1, stride=1, bias=False).cuda()
        de_conv342 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1).cuda()
        de_conv341 = nn.ConvTranspose2d(256, 1024, kernel_size=1, stride=1, bias=False).cuda()
        out = self.relu(bn_de_256_11(de_conv343(x)))
        out = self.relu(bn_de_256_12(de_conv342(out)))            
        out = bn_de_1024_3(de_conv341(out))
        out += identity
        x = self.relu(out)  
        
        identity = x
        bn_de_256_9 = _BatchNorm(256)
        bn_de_256_10 = _BatchNorm(256)
        bn_de_1024_2 = _BatchNorm(1024)
        de_conv333 = nn.ConvTranspose2d(1024, 256, kernel_size=1, stride=1, bias=False).cuda()
        de_conv332 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1).cuda()
        de_conv331 = nn.ConvTranspose2d(256, 1024, kernel_size=1, stride=1, bias=False).cuda()
        out = self.relu(bn_de_256_9(de_conv333(x)))
        out = self.relu(bn_de_256_10(de_conv332(out)))            
        out = bn_de_1024_2(de_conv331(out))
        out += identity
        x = self.relu(out)  

        identity = x
        bn_de_256_7 = _BatchNorm(256)
        bn_de_256_8 = _BatchNorm(256)
        bn_de_1024_1 = _BatchNorm(1024)
        de_conv323 = nn.ConvTranspose2d(1024, 256, kernel_size=1, stride=1, bias=False).cuda()
        de_conv322 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1).cuda()
        de_conv321 = nn.ConvTranspose2d(256, 1024, kernel_size=1, stride=1, bias=False).cuda()
        out = self.relu(bn_de_256_7(de_conv323(x)))
        out = self.relu(bn_de_256_8(de_conv322(out)))            
        out = bn_de_1024_1(de_conv321(out))
        out += identity
        x = self.relu(out)  
        
        identity = x
        bn_de_256_5 = _BatchNorm(256)
        bn_de_256_6 = _BatchNorm(256)
        bn_de_512_4 = _BatchNorm(512)
        bn_de_512_5 = _BatchNorm(512)
        de_conv314 = nn.ConvTranspose2d(1024, 512, kernel_size=1, stride=2, bias=False, output_padding = 1).cuda()
        de_conv313 = nn.ConvTranspose2d(1024, 256, kernel_size=1, stride=1, bias=False).cuda()
        de_conv312 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, groups=1, dilation=1, bias=False, padding=1, output_padding = 1).cuda()
        de_conv311 = nn.ConvTranspose2d(256, 512, kernel_size=1, stride=1, bias=False).cuda()
        out = self.relu(bn_de_256_5(de_conv313(x)))
        out = self.relu(bn_de_256_6(de_conv312(out)))            
        out = bn_de_512_4(de_conv311(out))
        identity = bn_de_512_5(de_conv314(x))
        out += identity
        x = self.relu(out)  
        #print ('size after deconv layer 3: ', x.size())


        #Layer 2               
        identity = x
        bn_de_128_7 = _BatchNorm(128)
        bn_de_128_8 = _BatchNorm(128)    
        bn_de_512_3 = _BatchNorm(512)
        de_conv243 = nn.ConvTranspose2d(512, 128, kernel_size=1, stride=1, bias=False).cuda()
        de_conv242 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1).cuda()
        de_conv241 = nn.ConvTranspose2d(128, 512, kernel_size=1, stride=1, bias=False).cuda()
        out = self.relu(bn_de_128_7(de_conv243(x)))
        out = self.relu(bn_de_128_8(de_conv242(out)))            
        out = bn_de_512_3(de_conv241(out))
        out += identity
        x = self.relu(out)  
        
        identity = x
        bn_de_128_5 = _BatchNorm(128)
        bn_de_128_6 = _BatchNorm(128)    
        bn_de_512_2 = _BatchNorm(512)
        de_conv233 = nn.ConvTranspose2d(512, 128, kernel_size=1, stride=1, bias=False).cuda()
        de_conv232 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1).cuda()
        de_conv231 = nn.ConvTranspose2d(128, 512, kernel_size=1, stride=1, bias=False).cuda()
        out = self.relu(bn_de_128_5(de_conv233(x)))
        out = self.relu(bn_de_128_6(de_conv232(out)))            
        out = bn_de_512_2(de_conv231(out))
        out += identity
        x = self.relu(out)  

        identity = x
        bn_de_128_3 = _BatchNorm(128)
        bn_de_128_4 = _BatchNorm(128)    
        bn_de_512_1 = _BatchNorm(512)
        de_conv223 = nn.ConvTranspose2d(512, 128, kernel_size=1, stride=1, bias=False).cuda()
        de_conv222 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1).cuda()
        de_conv221 = nn.ConvTranspose2d(128, 512, kernel_size=1, stride=1, bias=False).cuda()
        out = self.relu(bn_de_128_3(de_conv223(x)))
        out = self.relu(bn_de_128_4(de_conv222(out)))            
        out = bn_de_512_1(de_conv221(out))
        out += identity
        x = self.relu(out)  
        
        identity = x
        bn_de_128_1 = _BatchNorm(128)
        bn_de_128_2 = _BatchNorm(128)
        bn_de_256_3 = _BatchNorm(256)
        bn_de_256_4 = _BatchNorm(256)  
        de_conv214 = nn.ConvTranspose2d(512, 256, kernel_size=1, stride=2, bias=False, output_padding = 1).cuda()
        de_conv213 = nn.ConvTranspose2d(512, 128, kernel_size=1, stride=1, bias=False).cuda()
        de_conv212 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, groups=1, dilation=1, bias=False, padding=1, output_padding = 1).cuda()
        de_conv211 = nn.ConvTranspose2d(128, 256, kernel_size=1, stride=1, bias=False).cuda()          
        out = self.relu(bn_de_128_1(de_conv213(x)))
        out = self.relu(bn_de_128_2(de_conv212(out)))            
        out = bn_de_256_3(de_conv211(out))
        identity = bn_de_256_4(de_conv214(x))
        out += identity
        x = self.relu(out)    
        #print ('size after deconv layer 2: ', x.size())


        #Layer 1
        identity = x
        bn_de_64_7 = _BatchNorm(64)
        bn_de_64_8 = _BatchNorm(64)
        bn_de_256_2 = _BatchNorm(256)
        de_conv133 = nn.ConvTranspose2d(256, 64, kernel_size=1, stride=1, groups=1, dilation=1, bias=False).cuda()
        de_conv132 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1).cuda()
        de_conv131 = nn.ConvTranspose2d(64, 256, kernel_size=1, stride=1, bias=False).cuda()
        out = self.relu(bn_de_64_7(de_conv133(x)))
        out = self.relu(bn_de_64_8(de_conv132(out)))            
        out = bn_de_256_2(de_conv131(out))
        out += identity
        x = self.relu(out)  

        identity = x
        bn_de_64_5 = _BatchNorm(64)
        bn_de_64_6 = _BatchNorm(64)
        bn_de_256_1 = _BatchNorm(256)
        de_conv123 = nn.ConvTranspose2d(256, 64, kernel_size=1, stride=1, groups=1, dilation=1, bias=False).cuda()
        de_conv122 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1).cuda()
        de_conv121 = nn.ConvTranspose2d(64, 256, kernel_size=1, stride=1, bias=False).cuda()
        out = self.relu(bn_de_64_5(de_conv123(x)))
        out = self.relu(bn_de_64_6(de_conv122(out)))            
        out = bn_de_256_1(de_conv121(out))
        out += identity
        x = self.relu(out)  

        identity = x
        bn_de_64_3 = _BatchNorm(64)
        bn_de_64_4 = _BatchNorm(64)
        bn_de_64_1 = _BatchNorm(64)
        bn_de_64_2 = _BatchNorm(64)

        de_conv114 = nn.ConvTranspose2d(256, 64, kernel_size=1, stride=1, bias=False).cuda()
        de_conv113 = nn.ConvTranspose2d(256, 64, kernel_size=1, stride=1, bias=False).cuda()
        de_conv112 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, groups=1, dilation=1, bias=False, padding=1).cuda()
        de_conv111 = nn.ConvTranspose2d(64, 64, kernel_size=1, stride=1, bias=False).cuda()

        out = self.relu(bn_de_64_3(de_conv113(x)))
        out = self.relu(bn_de_64_4(de_conv112(out))) 
        out = bn_de_64_1(de_conv111(out))
        identity = bn_de_64_2(de_conv114(x))
        out += identity
        x = self.relu(out)   
        #print ('size after deconv layer 1: ', x.size())


        x = self.unpool(x, self.indices1)
        de_conv1 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, bias=False).cuda()
        x = de_conv1(x)
        
        bn_de_3 = _BatchNorm(3)
        x_reconst = self.relu(bn_de_3(x))
        return x_reconst

    def forward(self, x, is_deconv=False):
        
        x = self.relu(self.bn1(self.conv1(x)))
        x, self.indices1 = self.maxpool(x)


        #Layer 1
        identity11 = x
        out11 = self.relu(self.bn111(self.conv111(x)))
        out11 = self.relu(self.bn112(self.conv112(out11)))
        out11 = self.bn113(self.conv113(out11))
        identity11 = self.bn114(self.conv114(x))
        out11 += identity11
        x = self.relu(out11)
        
        identity12 = x
        out12 = self.relu(self.bn121(self.conv121(x)))
        out12 = self.relu(self.bn122(self.conv122(out12)))
        out12 = self.bn123(self.conv123(out12))
        out12 += identity12
        x = self.relu(out12)
        
        identity13 = x
        out13 = self.relu(self.bn131(self.conv131(x)))
        out13 = self.relu(self.bn132(self.conv132(out13)))
        out13 = self.bn133(self.conv133(out13))
        out13 += identity13
        x = self.relu(out13)


        #Layer 2
        identity21 = x
        out21 = self.relu(self.bn211(self.conv211(x)))
        out21 = self.relu(self.bn212(self.conv212(out21)))
        out21 = self.bn213(self.conv213(out21))
        identity21 = self.bn214(self.conv214(x))
        out21 += identity21
        x = self.relu(out21)
        
        identity22 = x
        out22 = self.relu(self.bn221(self.conv221(x)))
        out22 = self.relu(self.bn222(self.conv222(out22)))
        out22 = self.bn223(self.conv223(out22))
        out22 += identity22
        x = self.relu(out22)
        
        identity23 = x
        out23 = self.relu(self.bn231(self.conv231(x)))
        out23 = self.relu(self.bn232(self.conv232(out23)))
        out23 = self.bn233(self.conv233(out23))
        out23 += identity23
        x = self.relu(out23)       

        identity24 = x
        out24 = self.relu(self.bn241(self.conv241(x)))
        out24 = self.relu(self.bn242(self.conv242(out24)))
        out24 = self.bn243(self.conv243(out24))
        out24 += identity24
        x = self.relu(out24) 

        
        #Layer 3
        identity31 = x
        out31 = self.relu(self.bn311(self.conv311(x)))
        out31 = self.relu(self.bn312(self.conv312(out31)))
        out31 = self.bn313(self.conv313(out31))
        identity31 = self.bn314(self.conv314(x))
        out31 += identity31
        x = self.relu(out31)
        
        identity32 = x
        out32 = self.relu(self.bn321(self.conv321(x)))
        out32 = self.relu(self.bn322(self.conv322(out32)))
        out32 = self.bn323(self.conv323(out32))
        out32 += identity32
        x = self.relu(out32)
        
        identity33 = x
        out33 = self.relu(self.bn331(self.conv331(x)))
        out33 = self.relu(self.bn332(self.conv332(out33)))
        out33 = self.bn333(self.conv333(out33))
        out33 += identity33
        x = self.relu(out33)       

        identity34 = x
        out34 = self.relu(self.bn341(self.conv341(x)))
        out34 = self.relu(self.bn342(self.conv342(out34)))
        out34 = self.bn343(self.conv343(out34))
        out34 += identity34
        x = self.relu(out34) 
        
        identity35 = x
        out35 = self.relu(self.bn351(self.conv351(x)))
        out35 = self.relu(self.bn352(self.conv352(out35)))
        out35 = self.bn353(self.conv353(out35))
        out35 += identity35
        x = self.relu(out35)       

        identity36 = x
        out36 = self.relu(self.bn361(self.conv361(x)))
        out36 = self.relu(self.bn362(self.conv362(out36)))
        out36 = self.bn363(self.conv363(out36))
        out36 += identity36
        x = self.relu(out36) 


        #Layer 4
        identity41 = x
        out41 = self.relu(self.bn411(self.conv411(x)))
        out41 = self.relu(self.bn412(self.conv412(out41)))
        out41 = self.bn413(self.conv413(out41))
        identity41 = self.bn414(self.conv414(x))
        out41 += identity41
        x = self.relu(out41)

        identity42 = x
        out42 = self.relu(self.bn421(self.conv421(x)))
        out42 = self.relu(self.bn422(self.conv422(out42)))
        out42 = self.bn423(self.conv423(out42))
        out42 += identity42
        x = self.relu(out42)
        
        # add vae latent space
        identity43 = x
        identity43vae = x

        out43 = self.relu(self.bn431(self.conv431(x)))
        out43 = self.relu(self.bn432(self.conv432(out43)))
        out43 = self.bn433(self.conv433(out43))
        out43 += identity43
        mu = self.relu(out43)        
        mu = self.avgpool(mu)
        
        return mu

    


class ResClassifier(nn.Module):
    def __init__(self, class_num=31, extract=True, dropout_p=0.5):
        super(ResClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.BatchNorm1d(1000, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
            )
        
        self.fc2 = nn.Linear(1000, class_num)
        self.extract = extract
        self.dropout_p = dropout_p

    def forward(self, x):
        x = x.view(x.size(0),2048)
        fc1_emb = self.fc1(x)
        if self.training:
            fc1_emb.mul_(math.sqrt(1 - self.dropout_p))            
        logit = self.fc2(fc1_emb)

        if self.extract:
            return fc1_emb, logit         
        return logit
