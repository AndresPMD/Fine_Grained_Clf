# -*- coding: utf-8 -*-

"""
    Fine-grained Classification based on textual cues
"""

# Python modules
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import torch
import numpy as np
import glob
import os
import pickle

import torchvision
from torch.autograd import Variable
from sklearn.metrics import average_precision_score

import pdb
import sys

# Own modules
from logger import LogMetric
from utils import *
from options import Options
from data.data_generator import *
from models.models import load_model
from custom_optim import *
__author__ = "Andres Mafla Delgado; Sounak Dey"
__email__ = "amafla@cvc.uab.cat; sdey@cvc.uab.cat"



def adjust_learning_rate(optimizer, epoch):
    """
        Updates the learning rate given an schedule and a gamma parameter.
    """

    if epoch in args.schedule:
        args.learning_rate *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate
        print("Updating Learning rate to: ", args.learning_rate)

def train(data_loader, net, optim, cuda, criterion, epoch, log_int, num_classes, batch_size):
    batch_time = LogMetric.AverageMeter()
    batch_loss = LogMetric.AverageMeter()
    processed_batches = 0
    save_epoch = 5

    end = time.time()
    # Switch to train mode
    net.train()
    # For Accuracy
    total = 0.0
    correct = 0.0
    acc_per_class = [0.0] * num_classes
    class_total = [0.0] * num_classes

    # For Precision
    precision_per_class = [0.0] * num_classes

    for i, (data, labels, textual_features) in enumerate(data_loader):
        sample_size = np.shape(data)[0]
        if cuda:
            data, labels, textual_features = data.cuda(), labels.cuda(), textual_features.cuda()

        data = Variable(data)
        labels = Variable(labels)
        textual_features = Variable(textual_features)

        optim.zero_grad()

        output, attn_mask = net(data, textual_features, sample_size)
        loss = criterion(output, torch.max(labels, 1)[1])

        # Update the actor
        loss.backward()
        optim.step()

        net.zero_grad()

        # Save values
        batch_loss.update(loss.mean().item(), data.shape[0])
        batch_time.update(time.time()-end)
        end = time.time()

        processed_batches += 1
        seen = batch_size * processed_batches

        # Accuracy
        __, predicted = torch.max(output.data, 1)
        predicted = predicted.cpu().numpy()
        total += labels.size(0)
        labels = labels.data.cpu().numpy()
        labels = np.argmax(labels, axis=1)

        correct += (predicted == labels).sum().item()
        correct_list = (predicted == labels)
        for ix, label in enumerate(labels):
            acc_per_class[int(label)] += correct_list[ix]
            class_total[int(label)] += 1

        # Precision
        predicted = output.data.cpu().numpy()
        y_true = np.zeros((sample_size, num_classes))
        for sample in range (sample_size):
            label = labels[sample]
            y_true[sample][label] = 1
            precision_per_class[label] += average_precision_score(y_true[sample], predicted[sample])

        if processed_batches % 10 == 0:
            print('[Epoch: %d, Data Points Seen: %5d] Loss is: %.4f' % (epoch, seen, loss.data.cpu()))

        # VISUALIZATION
        if i < 1 and (epoch % save_epoch == 0) and args.outimg == 'True':
            out_folder = "%s/Epoch_%d_train" % (args.outimg_path, epoch)
            if not os.path.isdir(out_folder):
                os.makedirs(out_folder)

            pickle_file = '{}/batch_{}_input_torch.pickle'.format(out_folder, i)
            with open (pickle_file,'wb') as fp:
                pickle.dump(data.cpu().detach().numpy(), fp)

            pickle_file = '{}/batch_{}_attn_torch.pickle'.format(out_folder, i)
            with open (pickle_file,'wb') as fp:
                pickle.dump(attn_mask.cpu().detach().numpy(), fp)

            torchvision.utils.save_image(data, '{}/batch_{}_input_images.jpg'.format(out_folder, i), nrow=8,
                                         padding=2)
            torchvision.utils.save_image(attn_mask, '{}/batch_{}_attentionss.jpg'.format(out_folder, i), nrow=8,
                                         padding=2)


    print('Epoch: [{0}]: Loss {loss.avg:.4f}; Avg Time x Batch {b_time.avg:.4f}'.format(epoch, loss=batch_loss, b_time=batch_time))
    print('Correct: %d  Total: %d' % (correct, total))
    print('Accuracy on TRAIN SET: %.2f %%' % (100 * correct / total))
    mean_avg_acc = 0.0
    for ix, (cls, tot) in enumerate(zip(acc_per_class, class_total)):
        mean_avg_acc += cls / tot
        #print('Accuracy for %d class: %f' % (ix + 1, cls / tot))
    print('Average Accuracy: %.2f' % (100*(mean_avg_acc / num_classes)))

    total_precision = [0.0] * num_classes
    for ix, value in enumerate (precision_per_class):
        total_precision[ix] = precision_per_class[ix]/class_total[ix]
        #print ('Average Precision for %d class: %.4f' % (ix + 1, total_precision[ix] ))
    batch_mAP =sum(total_precision) / num_classes
    print('Mean Average Precision (mAP) on Train is: %.4f' % (100 * batch_mAP))

    return loss

def test(data_loader, net, cuda, num_classes, batch_size):
    batch_time = LogMetric.AverageMeter()
    performance = LogMetric.AverageMeter()

    processed_batches = 0

    end = time.time()

    # Switch to evaluation mode
    net.eval()
    # Accuracy
    total = 0.0
    correct = 0.0
    acc_per_class = [0.0] * num_classes
    class_total = [0.0] * num_classes

    # For Precision
    precision_per_class = [0.0] * num_classes

    with torch.no_grad():
        for i, (data, labels, textual_features) in enumerate(data_loader):
            sample_size = np.shape(data)[0]
            if cuda:
                data, labels, textual_features = data.cuda(), labels.cuda(), textual_features.cuda()

            data = Variable(data)
            output, attn_mask = net(data, textual_features, sample_size)

            processed_batches += 1
            seen = batch_size * processed_batches

            __, predicted = torch.max(output.data, 1)
            predicted = predicted.cpu().numpy()
            total += labels.size(0)
            labels = labels.data.cpu().numpy()
            labels = np.argmax(labels, axis=1)

            correct += (predicted == labels).sum().item()
            correct_list = (predicted == labels)
            for ix, label in enumerate(labels):
                acc_per_class[int(label)] += correct_list[ix]
                class_total[int(label)] += 1

            # Precision
            predicted = output.data.cpu().numpy()
            y_true = np.zeros((sample_size, num_classes))
            for sample in range(sample_size):
                label = labels[sample]
                y_true[sample][label] = 1
                precision_per_class[label] += average_precision_score(y_true[sample], predicted[sample])

            if processed_batches % 30 == 0:
                print('Processing batch: ', processed_batches)

            # Save values
            batch_time.update(time.time()-end)
            end = time.time()
    print('Correct: %d  Total: %d' % (correct, total))
    print('Accuracy of the network on the test set: %.2f %%' % (100 * correct / total))
    mean_avg_acc = 0.0
    for ix, (cls, tot) in enumerate(zip(acc_per_class, class_total)):
        mean_avg_acc += cls / tot
        #print('Accuracy for %d class: %f' % (ix + 1, cls / tot))
    print('Average Accuracy of the network on the test set: %.2f' % (100*(mean_avg_acc / num_classes)))

    total_precision = [0.0] * num_classes
    for ix, value in enumerate(precision_per_class):
        total_precision[ix] = precision_per_class[ix] / class_total[ix]
        print('Average Precision for %d class: %.4f' % (ix + 1, total_precision[ix]))
    batch_mAP =sum(total_precision) / num_classes
    print('Mean Average Precision (mAP) on TEST is: %.4f' % (100 * batch_mAP))

    return (batch_mAP)

def main():
    print('Preparing data')
    num_classes = get_num_classes(args)
    embedding_size = get_embedding_size(args.embedding)

    train_data, test_data, gt_annotations, text_embedding = load_data(args, embedding_size)

    #sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.prefetch,
                              pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=args.prefetch,
                              pin_memory=True)



    print('Creating Model')
    net = load_model(args, num_classes, embedding_size)

    print('Optimizer: ', args.optim)
    if args.optim == 'sgd':
        optim = torch.optim.SGD(net.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.decay, nesterov=True)
    elif args.optim =='adam':
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay)
    elif args.optim =='radam':
        optim_base = RAdam(filter(lambda p: p.requires_grad, net.parameters()), args.learning_rate)
        optim = Lookahead(optim_base, k=5, alpha=0.5)
    else: print("Optimizer not implemented")

    # Weight Tensor for Criterion
    weights = get_weight_criterion(args.dataset)

    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    evaluation = None

    print('Checking CUDA')
    if args.cuda and args.ngpu > 1:
        print('\t* Data Parallel **NOT TESTED**')
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.cuda:
        print('\t* CUDA ENABLED!')
        net = net.cuda()
        criterion = criterion.cuda()

    # Init variables
    early_stop_counter, start_epoch, best_perf = 0, 0, 0

    if args.load is not None:
        checkpoint = load_checkpoint(args.load)
        net.load_state_dict(checkpoint)

    for epoch in range(start_epoch, args.epochs):
        # Update learning rate
        adjust_learning_rate(optim, epoch)
        if args.test == 'False':
            print('\n*** TRAIN ***\n')
            loss = train(train_loader, net, optim, args.cuda, criterion, epoch, args.log_interval, num_classes, args.batch_size)
        print('\n*** TEST ***\n')
        performance = test(test_loader, net, args.cuda, num_classes, args.batch_size)

        # Early-Stop + Save model

        if performance > best_perf:
            best_perf = performance
            best_epoch = epoch
            early_stop_counter = 0
            if args.save_weights == 'True':
                save_checkpoint(net, best_perf, directory=args.save, file_name='checkpoint')
        else:
            if early_stop_counter == args.early_stop:
                print('\nEarly stop reached!')
                break
            early_stop_counter += 1

    # Load Best model in case of save it
    print("\nBest Performance is: %f at Epoch No. %d" % (best_perf, best_epoch))


    print('*** Training Completed ***')
    sys.exit()

if __name__ == '__main__':
    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))

    # Check cuda & Set random seed
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.save is not None:
        #print('Initialize logger')
        #ind = len(glob.glob(args.log + '*_run-batchSize_{}'.format(args.batch_size)))
        #log_dir = args.log + '{}_run-batchSize_{}/'.format(ind, args.batch_size)
        args.save = args.save + '{}_{}_{}_{}_{}/'.format(args.dataset, args.model, args.embedding, args.ocr, args.fusion)

        # Create logger
        #print('Log dir:\t' + log_dir)
        print('Save dir:\t' + args.save)

        #logger = LogMetric.Logger(log_dir, force=True)

    main()

