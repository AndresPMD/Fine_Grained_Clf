# -*- coding: utf-8 -*-

"""
    Fine-grained Classification based on textual cues
"""

# Python modules
import torch

import torch.nn as nn
import time
import torch
import numpy as np
import glob
import os
import json
from PIL import Image, ImageDraw, ImageFile

import torchvision
from torch.autograd import Variable
from torchvision import transforms

import pdb
import sys

# Own modules
from logger import LogMetric
from utils import *
from options import *
from data.data_generator import *
from models.models import load_model
from custom_optim import *
__author__ = "Andres Mafla Delgado; Sounak Dey"
__email__ = "amafla@cvc.uab.cat; sdey@cvc.uab.cat"


def test(args, net, cuda, num_classes):

    processed_imgs = 0
    # Switch to evaluation mode
    net.eval()

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(args.base_dir+'features/'):
        os.mkdir(args.base_dir+'features/')

    with torch.no_grad():
        image_list = os.listdir(args.base_dir+'images/')
        for image in image_list:
            img_path = args.base_dir+'images/' + image
            img = Image.open(img_path).convert('RGB')
            data = test_transform(img)
            data = data.view(-1, 3, 224, 224)

            textual_feature_path = args.base_dir+'fisher/' + image.split('.')[0] + '.json'
            with open(textual_feature_path, 'r') as fp:
                textual_feature = json.load(fp)
            textual_feature = np.resize(textual_feature, (1, 1, 38400))
            textual_feature = torch.from_numpy(textual_feature)
            textual_feature = textual_feature.type(torch.FloatTensor)

            if cuda:
                data, textual_feature = data.cuda(), textual_feature.cuda()
            data = Variable(data)

            output, attn_mask = net(data, textual_feature, sample_size=1)
            softmax = nn.Softmax(dim=1)
            features = softmax(output)
            features = features.cpu().numpy()
            features = features.tolist()
            with open (args.base_dir+'features/'+ image.split('.')[0] + '.json','w') as fp:
                json.dump(features, fp)

            processed_imgs += 1

    print ('%d Processed Images' %(processed_imgs))
    return

def main():
    print('Preparing data')

    if args.dataset == 'context':
        num_classes = 28
        weight_file = '/SSD/fine_grained_classification_with_textual_cues/backup/context_orig_fisherNet_fisher_yolo_phoc_concat/checkpoint7988.weights'
    else:
        num_classes = 20
        weight_file = '/SSD/fine_grained_classification_with_textual_cues/backup/bottles_orig_fisherNet_fisher_yolo_phoc_concat/checkpoint7690.weights'

    embedding_size = get_embedding_size(args.embedding)
    print('Loading Model')
    net = load_model(args, num_classes, embedding_size)
    checkpoint = load_checkpoint(weight_file)
    net.load_state_dict(checkpoint)



    print('Checking CUDA')
    if args.cuda and args.ngpu > 1:
        print('\t* Data Parallel **NOT TESTED**')
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.cuda:
        print('\t* CUDA ENABLED!')
        net = net.cuda()




    print('\n*** TEST ***\n')
    test(args, net, args.cuda, num_classes)
    print('*** Feature Extraction Completed ***')
    sys.exit()

if __name__ == '__main__':
    # Parse options
    args = Options_Test().parse()
    print('Parameters:\t' + str(args))

    # Check cuda & Set random seed
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()
    main()