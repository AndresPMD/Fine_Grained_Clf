#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.utils.data as data
from torchvision import transforms
import os
import pickle
import random
import sys
import json
from PIL import Image

sys.path.insert(0, '.')

import numpy as np
from skimage import io


def Bottle_dataset(args, embedding_size):
    # Random seed
    np.random.seed(args.seed)

    # Getting the classes and annotations
    # ******
    data_path = args.data_path
    with open(data_path +'/Drink_Bottle/split_'+ str(args.split) +'.json','r') as fp:
        gt_annotations = json.load(fp)

    # Load Embedding according to OCR
    if args.embedding == 'w2vec' or args.embedding == 'fasttext' or args.embedding == 'glove' or args.embedding == 'bert':
        with open(data_path + '/Drink_Bottle/' + args.ocr + '/text_embeddings/Drink_Bottle_' + args.embedding + '.pickle','rb') as fp:
            text_embedding = pickle.load(fp)
    elif args.embedding == 'phoc':
        text_embedding = {'embedding': 'phoc'}
    elif args.embedding == 'fisher':
        text_embedding = {'embedding': 'fisher'}
    else:
        print('OCR SELECTED NOT IMPLEMENTED')

    # Data Loaders

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_loader = Bottle_Train(args, gt_annotations, text_embedding, embedding_size, train_transform)

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_loader = Bottle_Test(args, gt_annotations, text_embedding, embedding_size, test_transform)

    return train_loader, test_loader, gt_annotations, text_embedding


class Bottle_Train(data.Dataset):
    def __init__(self, args, gt_annotations, text_embedding, embedding_size, transform=None):

        self.args = args
        self.gt_annotations = gt_annotations
        self.text_embedding = text_embedding
        self.embedding_size = embedding_size
        self.transform = transform
        self.image_list = list(gt_annotations['train'].keys())
        #Random.shuffle(self.image_list)


    def __len__(self):
        return len(self.gt_annotations['train'])

    def __getitem__(self, index):
        data_path = self.args.data_path

        assert index <= len(self), 'index range error'
        image_name = self.image_list[index].rstrip()
        image_path = data_path + '/Drink_Bottle/' + image_name
        img = Image.open(image_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        img_class = self.gt_annotations['train'][image_name]
        label = np.zeros(20)
        label[int(img_class) - 1] = 1
        label = torch.from_numpy(label)
        label = label.type(torch.FloatTensor)

        if self.args.embedding == 'w2vec' or self.args.embedding == 'fasttext' or self.args.embedding =='glove' or self.args.embedding == 'bert':
            text_embedding = np.asarray(self.text_embedding[image_name.replace('images/', '')])
        elif self.args.embedding == 'phoc':
            with open(data_path + '/Drink_Bottle/yolo_phoc/' + image_name.replace('images/','')[:-3] + 'json') as fp:
                phocs = json.load(fp)
                text_embedding = np.resize(phocs, (np.shape(phocs)[0], 604))
        elif self.args.embedding == 'fisher':

            if self.args.ocr == 'yolo_phoc':
                relative_path = '/Drink_Bottle/old_fisher_vectors/'
            elif self.args.ocr == 'e2e_mlt':
                relative_path = '/Drink_Bottle/fasttext_fisher/'
            else: print('Not Implemented')
            with open (data_path + relative_path +image_name.replace('images/','')[:-3] +'json')as fp:
                fisher_vector = json.load(fp)
                text_embedding = np.resize(fisher_vector, (1, 38400))
        # FISHER VECTORS DO NOT NEED MAX TEXTUAL
        if self.args.embedding != 'fisher':
            text_features = np.zeros((self.args.max_textual, self.embedding_size))
            if np.shape(text_embedding)[0] == 0:
                text_embedding = np.zeros((1,self.embedding_size))
            elif np.shape(text_embedding)[0] > self.args.max_textual:
                text_embedding = text_embedding[0:self.args.max_textual]
            text_features[:len(text_embedding)] = text_embedding
        else:
            text_features = text_embedding

        text_features = torch.from_numpy(text_features)
        text_features = text_features.type(torch.FloatTensor)

        return img, label, text_features

class Bottle_Test(data.Dataset):
    def __init__(self, args, gt_annotations, text_embedding, embedding_size, transform=None):
        self.args = args
        self.gt_annotations = gt_annotations
        self.text_embedding = text_embedding
        self.embedding_size = embedding_size
        self.transform = transform
        self.image_list = list(gt_annotations['test'].keys())

    def __len__(self):
        return len(self.gt_annotations['test'])

    def __getitem__(self, index):
        data_path  = self.args.data_path
        assert index <= len(self), 'index range error'
        image_name = self.image_list[index].rstrip()
        image_path = data_path + '/Drink_Bottle/' + image_name
        img = Image.open(image_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        img_class = self.gt_annotations['test'][image_name]
        label = np.zeros(20)
        label[int(img_class) - 1] = 1
        label = torch.from_numpy(label)
        label = label.type(torch.FloatTensor)

        if self.args.embedding == 'w2vec' or self.args.embedding == 'fasttext' or self.args.embedding =='glove' or self.args.embedding == 'bert':
            text_embedding = np.asarray(self.text_embedding[image_name.replace('images/', '')])
        elif self.args.embedding == 'phoc':
            with open(data_path + '/Drink_Bottle/yolo_phoc/' + image_name.replace('images/','')[:-3] + 'json') as fp:
                phocs = json.load(fp)
                text_embedding = np.resize(phocs, (np.shape(phocs)[0], 604))
        elif self.args.embedding == 'fisher':
            if self.args.ocr == 'yolo_phoc':
                relative_path = '/Drink_Bottle/old_fisher_vectors/'
            elif self.args.ocr == 'e2e_mlt':
                relative_path = '/Drink_Bottle/fasttext_fisher/'
            else: print('Not Implemented')
            with open (data_path + relative_path +image_name.replace('images/','')[:-3] +'json')as fp:
                fisher_vector = json.load(fp)
                text_embedding = np.resize(fisher_vector, (1, 38400))
        # FISHER VECTORS DO NOT NEED MAX TEXTUAL
        if self.args.embedding != 'fisher':
            text_features = np.zeros((self.args.max_textual, self.embedding_size))
            if np.shape(text_embedding)[0] == 0:
                text_embedding = np.zeros((1,self.embedding_size))
            elif np.shape(text_embedding)[0] > self.args.max_textual:
                text_embedding = text_embedding[0:self.args.max_textual]
            text_features[:len(text_embedding)] = text_embedding
        else:
            text_features = text_embedding

        text_features = torch.from_numpy(text_features)
        text_features = text_features.type(torch.FloatTensor)

        return img, label, text_features

