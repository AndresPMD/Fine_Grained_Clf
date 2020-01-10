from nltk.corpus import wordnet
from itertools import product
import re
import numpy as np
import pdb

with open('/SSD/Datasets/Context/classes.txt') as fp:
    tot_classes = fp.read().splitlines()
list1 = tot_classes
list2 = tot_classes
sims = []
pattern='[a-zA-Z]'
hierarchial_embedding = np.zeros((len(tot_classes), len(tot_classes)))


for word1, word2 in product(list1, list2):
    em_indx = tot_classes.index(word1)
    em_indy = tot_classes.index(word2)
    if len(re.findall(pattern, word1))>1:
        word1 = word1.split(re.findall(pattern, word1)[1])[0] + '_' + re.findall(pattern, word1)[1] + \
                word1.split(re.findall(pattern, word1)[1])[1]
        if word1 == 'Massage_Center':
            word1 = 'Massage_Parlor'
        elif word1 == 'Packing_Store':
            word1 = 'Packing'
        elif word1 == 'Pawn_Shop':
            word1 = 'Pawn'
        elif word1 == 'Steak_House':
            word1 = 'Steak'
        elif word1 == 'Tea_House':
            word1 = 'Tea'
    if len(re.findall(pattern, word2))>1:
        word2 = word2.split(re.findall(pattern, word2)[1])[0] + '_' + re.findall(pattern, word2)[1] + \
                word2.split(re.findall(pattern, word2)[1])[1]
        if word2 == 'Massage_Center':
            word2 = 'Massage_Parlor'
        elif word2 == 'Packing_Store':
            word2 = 'Packing'
        elif word2 == 'Pawn_Shop':
            word2 = 'Pawn'
        elif word2 == 'Steak_House':
            word2 = 'Steak'
        elif word2 == 'Tea_House':
            word2 = 'Tea'
    syns1 = wordnet.synsets(word1)
    syns2 = wordnet.synsets(word2)
    # for sense1, sense2 in product(syns1, syns2):
    #     d = wordnet.wup_similarity(sense1, sense2)
    #     sims.append((d, syns1, syns2))

    d = wordnet.wup_similarity(syns1[0], syns2[0])
    hierarchial_embedding[em_indx, em_indy] = d

print('Done')
# for each_class in tot_classes:
#     all_possible_synset = wordnet.synsets(each_class, 'n')
