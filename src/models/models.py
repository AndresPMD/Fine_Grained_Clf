# -*- coding: utf-8 -*-
from __future__ import print_function, division
import sys
sys.path.insert(0,'.')

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

"""
Visual Encoder model
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# Custom fusion modules
from .fusion import *



def load_model(args, classes_number, embedding_size):

    if args.model == 'visualNet':
        return Resnet_CNN(args=args, num_classes=classes_number, embedding_size=embedding_size)
    elif args.model == 'lenet':
        return Lenet_CNN(args = args, num_classes= classes_number, embedding_size=embedding_size)
    elif args.model == 'baseNet':
        return BaseNet(args = args, num_classes= classes_number, embedding_size=embedding_size)
    elif args.model == 'fisherNet':
        return FisherNet(args = args, num_classes= classes_number, max_textual = 1, embedding_size=embedding_size, reduced_size = 512)
    elif args.model == 'orig_fisherNet':
        return Orig_FisherNet(args = args, num_classes= classes_number, max_textual = 1, embedding_size=embedding_size, reduced_size = 512)
    elif args.model == 'TextNet':
        return TextNet(args = args, num_classes= classes_number, embedding_size=embedding_size, reduced_size = 512)

    else:
        raise NameError(args.model + ' not implemented!')


class AttentionModel(nn.Module):
    def __init__(self, hidden_layer=380):
        super(AttentionModel, self).__init__()

        self.attn_hidden_layer = hidden_layer
        self.net = nn.Sequential(nn.Conv2d(2048, self.attn_hidden_layer, kernel_size=1),
                                 nn.Conv2d(self.attn_hidden_layer, 1, kernel_size=1))

    def forward(self, x):
        attn_mask = self.net(x) # Shape BS 1x7x7
        attn_mask = attn_mask.view(attn_mask.size(0), -1)
        attn_mask = nn.Softmax(dim=1)(attn_mask)
        attn_mask = attn_mask.view(attn_mask.size(0), 1, x.size(2), x.size(3))
        x_attn = x * attn_mask
        x = x + x_attn
        return x, attn_mask

class Lenet_CNN(nn.Module):
    def __init__(self, args, num_classes, embedding_size, pretrained=True):
        super(Lenet_CNN, self).__init__()
        self.args = args
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.pretrained = pretrained
        lenet = models.googlenet(pretrained)


        self.cnn_features = nn.Sequential(*list(lenet.children())[:-1])
        #
        #
        # for param in self.cnn_features.parameters():
        #     param.requires_grad = False

        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, num_classes)


    def forward(self, im, textual_features, sample_size):
        x = self.cnn_features(im)  # Size (BS x 2048 x 7 x 7)
        x = self.fc1_bn(x.view(sample_size, 1024))
        x = self.fc1(x)
        return x

class Resnet_CNN(nn.Module):
    def __init__(self, args , num_classes, embedding_size, pretrained=True, attention=True):
        super(Resnet_CNN, self).__init__()
        self.args = args
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.pretrained = pretrained
        resnet152 = models.resnet152(pretrained)

        for name, child in resnet152.named_children():
            if name not in ['layer4']:
                # print(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = False
            else:
                # print(name + ' is not frozen')
                for param in child.parameters():
                    param.requires_grad = True

        self.cnn_features = nn.Sequential(*list(resnet152.children())[:-2])
        #
        #
        # for param in self.cnn_features.parameters():
        #     param.requires_grad = False

        # Attention model
        self.attention = attention
        self.attn = AttentionModel()
        self.attn_bn = nn.BatchNorm2d(2048)

        self.fc1_bn = nn.BatchNorm1d(100352)
        self.fc1 = nn.Linear(100352, num_classes)


    def forward(self, im, textual_features, sample_size):
        x = self.cnn_features(im)  # Size (BS x 2048 x 7 x 7)
        if self.attention:
            x, attn_mask = self.attn(x)  # Size (BS x 2048)
        x = self.attn_bn(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(self.fc1_bn(x)))

        return x, attn_mask


class BaseNet(nn.Module):
    def __init__(self, args, num_classes, embedding_size = 300, pretrained=True, attention=True):
        super(BaseNet, self).__init__()
        self.args = args
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.embedding_size = embedding_size

        if self.args.fusion == 'block':
            self.fusion = Block([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'blocktucker':
            self.fusion = BlockTucker([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'tucker':
            self.fusion = Tucker ([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mutan':
            self.fusion = Mutan([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mlb':
            self.fusion = MLB([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mfb':
            self.fusion = MFB([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mfh':
            self.fusion = MFH([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)

        resnet152 = models.resnet152(pretrained)

        for name, child in resnet152.named_children():
            if name not in ['layer4']:
                #print(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = False
            else:
                #print(name + ' is not frozen')
                for param in child.parameters():
                    param.requires_grad = True

        self.cnn_features = nn.Sequential(*list(resnet152.children())[:-2])

        # Attention model
        self.attention = attention
        self.attn = AttentionModel()
        self.attn_bn = nn.BatchNorm2d(2048)

        self.fc1 = nn.Linear(100352, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)

        # Semantic Attention Weights
        self.fc_w = nn.Linear(1024, self.embedding_size, bias=False)

        # LAST LAYERS
        self.bn3 = nn.BatchNorm1d(1024 + self.embedding_size)
        self.fc3 = nn.Linear(1024 + self.embedding_size, num_classes)


        '''
        self.fc3 = nn.Linear(1024 + self.embedding_size, 300)

        # CLASSIF LAYER
        self.bn4 = nn.BatchNorm1d(300)
        self.fc4 = nn.Linear(300, num_classes)
        '''

    def forward(self, im, textual_features, sample_size):
        x = self.cnn_features(im)  # Size (BS x 2048 x 7 x 7)

        if self.attention:
            x, attn_mask = self.attn(x)  # Size (BS x 2048)

        x = self.attn_bn(x)
        x = x.view(x.size(0), -1)
        visual_features = F.relu(self.fc1_bn(self.fc1(x))) # Visual Features BS x 1024
        x = self.fc_w(visual_features) # BS x 300 or (embedding size)

        x = torch.bmm(x.view(sample_size, 1, self.embedding_size), textual_features.permute(0, 2, 1))
        x = torch.tanh(x)
        x = F.softmax(x, dim=2)
        # Attention over textual features
        x = torch.bmm(x, textual_features)
        # Reshape visual features before fusion
        # Fuse
        if self.args.fusion != 'concat':
            x = self.fusion([x.view(sample_size, -1),visual_features])
        else:
            x = torch.cat((x[:, 0, :], visual_features), 1)
        '''
        ranking_vector = F.relu(self.fc3(self.bn3(x)))
        x = F.dropout(self.fc4(self.bn4(ranking_vector)), p=0.3, training=self.training)
        '''
        x = F.dropout(self.fc3(self.bn3(x)), p=0.3, training=self.training)

        return x, attn_mask


class FisherNet(nn.Module):
    def __init__(self, args, num_classes, max_textual = 20, embedding_size = 38400, reduced_size = 512, pretrained=True, attention=True):
        super(FisherNet, self).__init__()
        self.args = args
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.embedding_size = embedding_size
        self.reduced_size = reduced_size
        self.max_textual = max_textual

        if self.args.fusion == 'block':
            self.fusion = Block([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'blocktucker':
            self.fusion = BlockTucker([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'tucker':
            self.fusion = Tucker ([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mutan':
            self.fusion = Mutan([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mlb':
            self.fusion = MLB([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mfb':
            self.fusion = MFB([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mfh':
            self.fusion = MFH([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)

        resnet152 = models.resnet152(pretrained)

        for name, child in resnet152.named_children():
            if name not in ['layer4']:
                #print(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = False
            else:
                #print(name + ' is not frozen')
                for param in child.parameters():
                    param.requires_grad = True

        self.cnn_features = nn.Sequential(*list(resnet152.children())[:-2])

        # Reduce Dimensionality of Fisher Vectors
        self.FV_bn1 = nn.BatchNorm1d(embedding_size)
        self.FV_fc1 = nn.Linear(embedding_size, 4096)
        self.FV_bn2 = nn.BatchNorm1d(4096)
        self.FV_fc2 = nn.Linear(4096, reduced_size)

        # Attention model
        self.attention = attention
        self.attn = AttentionModel()
        self.attn_bn = nn.BatchNorm2d(2048)

        self.fc1 = nn.Linear(100352, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)

        # Semantic Attention Weights
        self.fc_w = nn.Linear(1024, self.reduced_size, bias=False)

        # LAST LAYERS
        self.bn3 = nn.BatchNorm1d(1024 + self.reduced_size)
        self.fc3 = nn.Linear(1024 + self.reduced_size, num_classes)

    def forward(self, im, textual_features, sample_size):
        x = self.cnn_features(im)  # Size (BS x 2048 x 7 x 7)

        if self.attention:
            x, attn_mask = self.attn(x)  # Size (BS x 2048)

        x = self.attn_bn(x)
        x = x.view(x.size(0), -1)
        visual_features = F.relu(self.fc1_bn(self.fc1(x))) # Visual Features BS x 1024
        x = self.fc_w(visual_features) # BS x 300 or (embedding size)

        textual_features = F.relu(self.FV_fc1(self.FV_bn1(textual_features.view(sample_size, -1))))
        textual_features = F.dropout(F.relu(self.FV_fc2(self.FV_bn2(textual_features))), p=0.5, training=self.training)

        x = torch.mul(x, textual_features)
        x = torch.tanh(x)
        x =  torch.mul(x, textual_features)
        # Reshape visual features before fusion
        # Fuse
        if self.args.fusion != 'concat':
            x = self.fusion([x.view(sample_size, -1),visual_features])
        else:
            x = torch.cat((x, visual_features), 1)

        x = F.dropout(self.fc3(self.bn3(x)), p=0.5, training=self.training)

        return x, attn_mask


class Orig_FisherNet(nn.Module):
    def __init__(self, args, num_classes, max_textual = 20, embedding_size = 38400, reduced_size = 512, pretrained=True, attention=True):
        super(Orig_FisherNet, self).__init__()
        self.args = args
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.embedding_size = embedding_size
        self.reduced_size = reduced_size
        self.max_textual = max_textual

        if self.args.fusion == 'block':
            self.fusion = Block([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'blocktucker':
            self.fusion = BlockTucker([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'tucker':
            self.fusion = Tucker ([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mutan':
            self.fusion = Mutan([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mlb':
            self.fusion = MLB([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mfb':
            self.fusion = MFB([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mfh':
            self.fusion = MFH([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)

        resnet152 = models.resnet152(pretrained)

        for name, child in resnet152.named_children():
            if name not in ['layer4']:
                #print(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = False
            else:
                #print(name + ' is not frozen')
                for param in child.parameters():
                    param.requires_grad = True

        self.cnn_features = nn.Sequential(*list(resnet152.children())[:-2])

        # Reduce Dimensionality of Fisher Vectors
        self.FV_bn1 = nn.BatchNorm1d(embedding_size)
        self.FV_fc1 = nn.Linear(embedding_size, 4096)
        self.FV_bn2 = nn.BatchNorm1d(4096)
        self.FV_fc2 = nn.Linear(4096, reduced_size)

        # Attention model
        self.attention = attention
        self.attn = AttentionModel()
        self.attn_bn = nn.BatchNorm2d(2048)

        self.fc1 = nn.Linear(100352, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)

        # Semantic Attention Weights
        self.fc_w = nn.Linear(1024, self.reduced_size, bias=False)

        # LAST LAYERS
        self.bn3 = nn.BatchNorm1d(1024 + self.reduced_size)
        self.fc3 = nn.Linear(1024 + self.reduced_size, num_classes)

    def forward(self, im, textual_features, sample_size):
        x = self.cnn_features(im)  # Size (BS x 2048 x 7 x 7)

        if self.attention:
            x, attn_mask = self.attn(x)  # Size (BS x 2048)

        x = self.attn_bn(x)
        x = x.view(x.size(0), -1)
        visual_features = F.relu(self.fc1_bn(self.fc1(x))) # Visual Features BS x 1024
        x = self.fc_w(visual_features) # BS x 300 or (embedding size)

        # FISHER FEATURES
        textual_features = F.relu(self.FV_fc1(self.FV_bn1(textual_features.view(sample_size, -1))))
        #textual_features = F.dropout(F.relu(self.FV_fc2(self.FV_bn2(textual_features))), p=0.5, training=self.training)
        textual_features = F.dropout(self.FV_fc2(self.FV_bn2(textual_features)), p=0.5, training=self.training)


        x = torch.mul(x, textual_features)
        x = torch.tanh(x)
        x =  torch.mul(x, textual_features)
        # Reshape visual features before fusion
        # Fuse
        if self.args.fusion != 'concat':
            x = self.fusion([x.view(sample_size, -1),visual_features])
        else:
            x = torch.cat((x, visual_features), 1)

        x = F.dropout(self.fc3(self.bn3(x)), p=0.5, training=self.training)

        return x, attn_mask

class TextNet(nn.Module):
    def __init__(self, args, num_classes, embedding_size = 300, reduced_size=512, pretrained=True, attention=True):
        super(TextNet, self).__init__()
        self.args = args
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.embedding_size = embedding_size
        self.reduced_size = reduced_size

        if self.args.fusion == 'block':
            self.fusion = Block([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'blocktucker':
            self.fusion = BlockTucker([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'tucker':
            self.fusion = Tucker ([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mutan':
            self.fusion = Mutan([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mlb':
            self.fusion = MLB([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mfb':
            self.fusion = MFB([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mfh':
            self.fusion = MFH([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)

        resnet152 = models.resnet152(pretrained)

        for name, child in resnet152.named_children():
            if name not in ['layer4']:
                #print(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = False
            else:
                #print(name + ' is not frozen')
                for param in child.parameters():
                    param.requires_grad = True

        self.cnn_features = nn.Sequential(*list(resnet152.children())[:-2])

        # Attention model
        self.attention = attention
        self.attn = AttentionModel()
        self.attn_bn = nn.BatchNorm2d(2048)

        self.fc1 = nn.Linear(100352, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)

        # Semantic Attention Weights
        self.fc_w = nn.Linear(1024, self.reduced_size, bias=False)

        # LAST LAYERS
        self.bn3 = nn.BatchNorm1d(1024 + self.reduced_size)
        self.fc3 = nn.Linear(1024 + self.reduced_size, num_classes)

        # ADDITIONAL LAYERS TO TEST SELF LEARNING OF MORPHOLOGY
        self.bn_text1 = nn.BatchNorm1d(self.args.max_textual)
        self.fc_text1 = nn.Linear(self.embedding_size, 550)

        self.bn_text2 = nn.BatchNorm1d(self.args.max_textual)
        self.fc_text2 = nn.Linear(550, self.reduced_size)

    def forward(self, im, textual_features, sample_size):

        x = self.cnn_features(im)  # Size (BS x 2048 x 7 x 7)

        if self.attention:
            x, attn_mask = self.attn(x)  # Size (BS x 2048)

        x = self.attn_bn(x)
        x = x.view(x.size(0), -1)
        visual_features = F.relu(self.fc1_bn(self.fc1(x))) # Visual Features BS x 1024
        x = self.fc_w(visual_features) # BS x 300 or (embedding size)


        # SELF LEARNING?
        textual_features = self.bn_text1(textual_features)
        textual_features = F.leaky_relu(self.fc_text1(textual_features))

        textual_features = self.bn_text2(textual_features)
        textual_features = F.leaky_relu(self.fc_text2(textual_features))


        # USUAL PIPELINE
        x = torch.bmm(x.view(sample_size, 1, self.reduced_size), textual_features.permute(0, 2, 1))
        x = torch.tanh(x)
        x = F.softmax(x, dim=2)
        # Attention over textual features
        x = torch.bmm(x, textual_features)
        # Reshape visual features before fusion
        # Fuse
        if self.args.fusion != 'concat':
            x = self.fusion([x.view(sample_size, -1),visual_features])
        else:
            x = torch.cat((x[:, 0, :], visual_features), 1)
        '''
        ranking_vector = F.relu(self.fc3(self.bn3(x)))
        x = F.dropout(self.fc4(self.bn4(ranking_vector)), p=0.3, training=self.training)
        '''
        x = F.dropout(self.fc3(self.bn3(x)), p=0.3, training=self.training)

        return x, attn_mask


def normalize(x):
    return x / x.norm(dim=1, keepdim=True)

