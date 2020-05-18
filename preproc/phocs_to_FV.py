
import scipy.io
import numpy as np
import logging
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture as GMM
import time
import json
import os


def build_phoc(words, phoc_unigrams, unigram_levels,
               bigram_levels=None, phoc_bigrams=None,
               split_character=None, on_unknown_unigram='error'):
    '''
    Calculate Pyramidal Histogram of Characters (PHOC) descriptor (see Almazan 2014).
    Args:
        word (str): word to calculate descriptor for
        phoc_unigrams (str): string of all unigrams to use in the PHOC
        unigram_levels (list of int): the levels for the unigrams in PHOC
        phoc_bigrams (list of str): list of bigrams to be used in the PHOC
        phoc_bigram_levls (list of int): the levels of the bigrams in the PHOC
        split_character (str): special character to split the word strings into characters
        on_unknown_unigram (str): What to do if a unigram appearing in a word
            is not among the supplied phoc_unigrams. Possible: 'warn', 'error'
    Returns:
        the PHOC for the given word
    '''
    # prepare output matrix
    logger = logging.getLogger('PHOCGenerator')
    if on_unknown_unigram not in ['error', 'warn']:
        raise ValueError('I don\'t know the on_unknown_unigram parameter \'%s\'' % on_unknown_unigram)
    phoc_size = len(phoc_unigrams) * np.sum(unigram_levels)
    if phoc_bigrams is not None:
        phoc_size += len(phoc_bigrams) * np.sum(bigram_levels)
    phocs = np.zeros((len(words), phoc_size))
    # prepare some lambda functions
    occupancy = lambda k, n: [float(k) / n, float(k + 1) / n]
    overlap = lambda a, b: [max(a[0], b[0]), min(a[1], b[1])]
    size = lambda region: region[1] - region[0]

    # map from character to alphabet position
    char_indices = {d: i for i, d in enumerate(phoc_unigrams)}

    # iterate through all the words
    for word_index, word in enumerate(words):
        if split_character is not None:
            word = word.split(split_character)
        n = len(word)
        for index, char in enumerate(word):
            char_occ = occupancy(index, n)
            if char not in char_indices:
                if on_unknown_unigram == 'warn':
                    logger.warn('The unigram \'%s\' is unknown, skipping this character', char)
                    continue
                else:
                    logger.fatal('The unigram \'%s\' is unknown', char)
                    raise ValueError()
            char_index = char_indices[char]
            for level in unigram_levels:
                for region in range(level):
                    region_occ = occupancy(region, level)
                    if size(overlap(char_occ, region_occ)) / size(char_occ) >= 0.5:
                        feat_vec_index = sum([l for l in unigram_levels if l < level]) * len(
                            phoc_unigrams) + region * len(phoc_unigrams) + char_index
                        phocs[word_index, feat_vec_index] = 1

                # add bigrams
        if phoc_bigrams is not None:
            ngram_features = np.zeros(len(phoc_bigrams) * np.sum(bigram_levels))
            ngram_occupancy = lambda k, n: [float(k) / n, float(k + 2) / n]
            for i in range(n - 1):
                ngram = word[i:i + 2]
                phoc_dict = {k: v for v, k in enumerate(phoc_bigrams)}
                if phoc_dict.get(ngram, 666) == 666:
                    continue
                occ = ngram_occupancy(i, n)
                for level in bigram_levels:
                    for region in range(level):
                        region_occ = occupancy(region, level)
                        overlap_size = size(overlap(occ, region_occ)) / size(occ)
                        if overlap_size >= 0.5:
                            ngram_features[region * len(phoc_bigrams) + phoc_dict[ngram]] = 1
            phocs[word_index, -ngram_features.shape[0]:] = ngram_features

    return phocs

def phoc(raw_word):
    '''
    :param raw_word: string of word to be converted
    :return: phoc representation as a np.array (1,604)
    '''

    word =[raw_word]
    word_lowercase = word[0].lower()
    word = [word_lowercase]
    phoc_unigrams = 'abcdefghijklmnopqrstuvwxyz0123456789'
    unigram_levels = [2,3,4,5]
    bigram_levels=[]
    bigram_levels.append(2)

    phoc_bigrams = []
    i = 0
    with open('bigrams_new.txt','r') as f:
        for line in f:
            a = line.split()
            phoc_bigrams.append(a[0].lower())
            #phoc_bigrams.append(list(a[0])[0])
            #phoc_bigrams.append(list(a[0])[1])
            i = i +1
            if i >= 50:break


    qry_phocs = build_phoc(words = word, phoc_unigrams = phoc_unigrams, unigram_levels = unigram_levels,
                           bigram_levels = bigram_levels, phoc_bigrams = phoc_bigrams)

    return qry_phocs


def text_cleaner (dirty_text):
    # CLEANS NOT WANTED CHARACTERES AND SENDS STRING 
    clean_text = ''.join(c for c in dirty_text if c not in '(){}<>;:!@#$%^&*_-=+-*/[]\' \"?>.<,')
    return clean_text

def fisher_vector(xx, gmm):
    """Computes the Fisher vector on a set of descriptors.
    Parameters
    ----------
    xx: array_like, shape (N, D) or (D, )
        The set of descriptors
    gmm: instance of sklearn mixture.GMM object
        Gauassian mixture model of the descriptors.
    Returns
    -------
    fv: array_like, shape (K + 2 * D * K, )
        Fisher vector (derivatives with respect to the mixing weights, means
        and variances) of the given descriptors.
    Reference
    ---------
    J. Krapac, J. Verbeek, F. Jurie.  Modeling Spatial Layout with Fisher
    Vectors for Image Categorization.  In ICCV, 2011.
    http://hal.inria.fr/docs/00/61/94/03/PDF/final.r1.pdf
    """
    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
        - Q_xx_2
        - Q_sum * gmm.means_ ** 2
        + Q_sum * gmm.covariances_
        + 2 * Q_xx * gmm.means_)

    # Merge derivatives into a vector.
    #return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))
    return np.hstack((d_mu.flatten(), d_sigma.flatten()))


## Load Jaderberg 90K dictionary

dictionary = '90K_dictionary_Jaderberg.txt'

## Create PHOC Matrix from most common English words
file = open(dictionary, 'r')
lines = file.readlines()
phoc_matrix = np.zeros((len(lines), 604))
for i,line in tqdm(enumerate (lines)):
    phoc_matrix[i] = phoc(text_cleaner(line.replace('\n','')))
print('The shape of the data is: ', np.shape(phoc_matrix))    

## FIRST L2 NORM PHOCS
norm_phoc_matrix = preprocessing.normalize(phoc_matrix, norm = 'l2')

## SCALER THE PHOCS...
scaler = StandardScaler()
scaler.fit(norm_phoc_matrix)
data = scaler.transform(norm_phoc_matrix)

## PCA
pca = PCA(n_components=300)
pca.fit(data)
pca_data = pca.transform(data)
print(np.shape(pca_data))
print('PCA Complete!')

## TRAIN GMM model
start = time.time()
# Original on Raw PHOCs with PCA
gmm = GMM(n_components = 64, covariance_type = 'diag')
print(np.shape(pca_data))
gmm.fit(pca_data)

finish = time.time()

print('GMM Training Complete!')
print('Total time (s): ', finish  - start)

## OBTAIN FISHER VECTORS FROM PHOCS

fishers_path = './Fisher_vectors/'
if not os.path.exists(fishers_path):
    os.mkdir(fishers_path)

phocs_path = '/SSD/pytorch-yolo2-master/results/SVT/'
files = os.listdir(phocs_path)


for file in tqdm(files):
    # Read PHOCs from json file:
    with open(phocs_path + file)as fp:
        phocs = json.load(fp)
    phocs = np.resize(phocs, (np.shape(phocs)[0], 604))

    if np.shape(phocs)[0] == 0:
        phocs = np.zeros([1,604])
    
    phoc_normalized = preprocessing.normalize(phocs, norm ='l2')
    scaler_phoc = scaler.transform(phoc_normalized)
    phoc_PCA = pca.transform(scaler_phoc)
    
    phoc_FV = fisher_vector(phoc_PCA, gmm)
    phoc_FV = preprocessing.normalize(phoc_FV.reshape(1,-1), norm ='l2')
    
    phoc_FV = phoc_FV.tolist()
    with open (fishers_path+file[:-3]+'json','w')as fp:
        json.dump(phoc_FV,fp)
print('Complete!')
