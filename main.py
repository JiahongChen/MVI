import os
import argparse
import tqdm
import os
import argparse
import numpy as np
import tqdm
from itertools import chain
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import time 
from utils import CLEFImage, weights_init, print_args
from model.net import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default='data/OfficeHome/list')
parser.add_argument("--source", default='Clipart')
parser.add_argument("--target", default='Product')
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--shuffle", default=True)
parser.add_argument("--num_workers", default=0)
parser.add_argument("--epoch", default=200, type=int)
parser.add_argument("--snapshot", default="")
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--class_num", default=65)
parser.add_argument("--extract", default=True)
parser.add_argument("--weight_L2norm", default=0.05)
parser.add_argument("--weight_entropy", default=0.1, type=float)
parser.add_argument("--dropout_p", default=0.5, type=float)
parser.add_argument("--task", default='None', type=str)
parser.add_argument("--post", default='-1', type=str)
parser.add_argument("--repeat", default='-1', type=str)
parser.add_argument("--result", default='record')
parser.add_argument("--save", default=False, type=bool)
parser.add_argument("--lambda_val", default=0.1, type=float)
parser.add_argument("--entropy_thres", default=0.0000001, type=float)
parser.add_argument('--thres_rec', type=float, default=0.1, help='coefficient for reconstruction loss')
parser.add_argument("--optimizer", default='SGD', type=str)
parser.add_argument('--multiGPU', type=bool, default=False,
                    help='enable train on multiple GPU or not, default is False')
args = parser.parse_args()
print_args(args)

t = time.time()

source_root = 'data/OfficeHome/'+args.source
source_label = os.path.join(args.data_root, args.source+'.txt')
target_root = 'data/OfficeHome/'+args.target
target_label = os.path.join(args.data_root, args.target+'.txt')

train_transform = transforms.Compose([
    transforms.Scale((256, 256)),
    transforms.RandomCrop((221, 221)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

source_set = CLEFImage(source_root, source_label, train_transform)
target_set = CLEFImage(target_root, target_label, train_transform)

source_loader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers)
target_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers)


# load pretrained ResNet50 model
netG = ResNet50_mod_name().cuda()
if args.multiGPU:
    state_dict = torch.load('model/resnet_model_multiGPU.pth')
else:
    state_dict = torch.load('model/resnet_model.pth')
netF = ResClassifier(class_num=args.class_num, extract=args.extract, dropout_p=args.dropout_p).cuda()
if torch.cuda.device_count() > 1 and args.multiGPU:
    print('===========Using Multiple GPU===========')
    netG = nn.DataParallel(netG)
    netF = nn.DataParallel(netF)

netG.load_state_dict(state_dict)
netF.apply(weights_init)

# compute classification error
def get_cls_loss(pred, gt):
    cls_loss = F.nll_loss(F.log_softmax(pred), gt)
    return cls_loss

# compute entropy loss
def get_entropy_loss(p_softmax):
    mask = p_softmax.ge(args.entropy_thres)
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    
    return args.weight_entropy * (entropy / float(p_softmax.size(0)))

# initialize tensor for storing \hat{X}_T, \hat{X}
reconst_t_zt = torch.ones([args.batch_size, 3, 221, 221]).cuda()
reconst_t_z = torch.ones([args.batch_size, 3, 221, 221]).cuda()

# initialize a L1 loss for domain consistency
ConsistencyCriterion = nn.L1Loss().cuda()

if args.optimizer == 'SGD':
    print ('Training using SGD')
    if args.multiGPU:
        opt_g = optim.SGD(netG.module.parameters(), lr=args.lr, weight_decay=0.0005)
        opt_f = optim.SGD(netF.module.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    else:
        opt_g = optim.SGD(netG.parameters(), lr=args.lr, weight_decay=0.0005)
        opt_f = optim.SGD(netF.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
elif args.optimizer == 'RMSprop':
    print ('Training using RMSprop')
    if args.multiGPU:
        opt_g = optim.RMSprop(netG.module.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=0.0005)
        opt_f = optim.RMSprop(netF.module.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=0.0005)    
    else:
        opt_g = optim.RMSprop(netG.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=0.0005)
        opt_f = optim.RMSprop(netF.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=0.0005)
max_correct = -1.0
correct_array = []

# start training
for epoch in range(1, args.epoch+1):
    source_loader_iter = iter(source_loader)
    target_loader_iter = iter(target_loader)
    print(">>training " + args.task + " epoch : " + str(epoch))

    netG.train()
    netF.train()
    for i, (t_imgs, _) in tqdm.tqdm(enumerate(target_loader_iter)):
        try:
            s_imgs, s_labels = source_loader_iter.next()
        except:
            source_loader_iter = iter(source_loader)
            s_imgs, s_labels = source_loader_iter.next()

        if s_imgs.size(0) != args.batch_size or t_imgs.size(0) != args.batch_size:
            continue

        s_imgs = Variable(s_imgs.cuda())
        s_labels = Variable(s_labels.cuda())     
        t_imgs = Variable(t_imgs.cuda())
        
        opt_g.zero_grad()
        opt_f.zero_grad()

        # apply feature extractor to input images
        s_bottleneck = netG(s_imgs) #mu_s
        t_bottleneck = netG(t_imgs) #mu_t

        # get classification results
        s_fc2_emb, s_logit = netF(s_bottleneck)
        t_fc2_emb, t_logit = netF(t_bottleneck)

        # get source domain classification error
        s_cls_loss = get_cls_loss(s_logit, s_labels)
        
        # compute entropy loss
        t_prob = F.softmax(t_logit)
        t_entropy_loss = get_entropy_loss(t_prob)

        # comput target domain reconstruction loss
        loss_recons_t_zt = F.l1_loss(reconst_t_zt, t_imgs, reduction='mean')

        # compute mutual information regularization
        loss_recons_t_z = F.l1_loss(reconst_t_z, t_imgs, reduction='mean')

        # KLD regularization for variational inference
        std = torch.exp(0.5*t_bottleneck)
        eps = torch.randn_like(std)
        feat_z = s_bottleneck + eps*std
        KLD = -1.0*F.kl_div(F.log_softmax(s_bottleneck), F.softmax(feat_z))

        # loss function
        loss = s_cls_loss + t_entropy_loss + args.thres_rec*(loss_recons_t_zt+loss_recons_t_z) + args.lambda_val*KLD
        
        loss.backward()
        
        if (i+1) % 5 == 0:
            print ("cls_loss: %.4f, loss_recons_t_zt: %.4f, loss_recons_t_z: %.4f, KLD: %.6f" % (s_cls_loss.item(), loss_recons_t_zt.item(), loss_recons_t_z.item(), KLD.item()))
        
        opt_g.step()
        opt_f.step()



        # Calculate \hat{X}_T, \hat{X} for the next iteration
        if args.multiGPU:
            reconst_t_zt = netG.module.decode(s_bottleneck, t_bottleneck, is_rep = False).detach()
            reconst_t_z = netG.module.decode(s_bottleneck, t_bottleneck, is_rep = True).detach()
        else:
            reconst_t_zt = netG.decode(s_bottleneck, t_bottleneck, is_rep = False).detach()
            reconst_t_z = netG.decode(s_bottleneck, t_bottleneck, is_rep = True).detach()



    if args.save:
        torch.save(netG.state_dict(), os.path.join("MVI_" + args.task + "_netG_" + args.post + '.' + args.repeat + '_' + str(epoch) + ".pth"))
        torch.save(netF.state_dict(), os.path.join("MVI_" + args.task + "_netF_" + args.post + '.' + args.repeat + '_'  + str(epoch) + ".pth"))

    # evaluate model
    netG.eval()
    netF.eval()
    correct = 0
    t_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
    for (t_imgs, t_labels) in t_loader:
        t_imgs = Variable(t_imgs.cuda())
        t_bottleneck = netG(t_imgs)
        t_fc2_emb, t_logit = netF(t_bottleneck)
        pred = F.softmax(t_logit)
        pred = pred.data.cpu().numpy()
        pred = pred.argmax(axis=1)
        t_labels = t_labels.numpy()
        correct += np.equal(t_labels, pred).sum()
    t_imgs = []
    t_bottleneck = []
    t_logit = []
    t_fc2_emb = []
    pred = []
    t_labels = []

    # compute classification accuracy for target domain
    correct = correct * 1.0 / len(target_set)
    correct_array.append(correct)

    if correct >= max_correct:
        max_correct = correct
    print ("Epoch {0} accuray: {1}; max acc: {2}".format(epoch, correct, max_correct))

# save results
print("max acc: ", max_correct)
result = open(os.path.join(args.result, "MVI_" + args.optimizer + "_" + args.task + "_" + str(max_correct) +"_lr_"+str(args.lr)+'_lambda_' + str(args.lambda_val) + '_recons_' + str(args.thres_rec)+'_entropy_' + str(args.entropy_thres) + "_dropout_"+str(args.dropout_p)+"_weight_entropy_"+str(args.weight_entropy)+"_batch_"+str(args.batch_size)+"_score.txt"), "a")
for c in correct_array:
    result.write(str(c) + "\n")
result.write("Max: "+ str(correct) + "\n")
elapsed = time.time() - t
print("elapsed: ", elapsed)
result.write(str(elapsed) + "\n")
result.close()