import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words
import matplotlib.pyplot as plt
import numpy.linalg



def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    # ----- TODO -----
    hist, bin_edges = np.histogram( wordmap, bins=K, range=[0,K])
    norm = np.linalg.norm(hist,ord=1)
    hist = hist/norm
    return hist

def get_unnormed_feature_from_wordmap(opts, wordmap):
    '''
    Compute unnormed histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    # ----- TODO -----
    hist, bin_edges = np.histogram( wordmap, bins=K, range=[0,K])
    return hist

def norm_and_weight(weight, hist):
    norm = np.linalg.norm(hist,ord=1)
    return weight*hist/norm

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^(L+1)-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    # ----- TODO -----
    num = 2**L
    #print(wordmap.shape)
    for i in range(wordmap.shape[0]%num,num):
        wordmap = np.vstack([wordmap,np.zeros((1,wordmap.shape[1]))])
    for i in range(wordmap.shape[1]%num,num):
        wordmap = np.hstack([wordmap,np.zeros((wordmap.shape[0],1))])
    #print(wordmap.shape)
    wordlist = np.hsplit(wordmap,num)
    wordlist = [np.vsplit(word,num) for word in wordlist]
    hist_all = []
    #pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for i in range(0,num):
        for j in range(0,num):
            #print(wordlist[i][j])
            #print(0.5*pool.apply(get_unnormed_feature_from_wordmap, args=(opts,wordlist[i][j])))
            hist_all.append(get_unnormed_feature_from_wordmap(opts,wordlist[j][i]))
            #hist_all[i*num+j,:] = (0.5*get_unnormed_feature_from_wordmap(opts,wordlist[j][i]))
    #pool.close()
    start = 0
    for i in range(L,0,-1):
        for j in range(0,2**i,2):
            for k in range(0, 2**i, 2):
                hist_all.append(np.add(np.add(hist_all[start+j*2**i+k],hist_all[start+j*2**i+k+1]),
                                        np.add(hist_all[start+(j+1)*2**i+k],hist_all[start+(j+1)*2**i+k+1])))
                weight = 2**(i-L-1)
                hist_all[start+j*2**i+k] = norm_and_weight(weight, hist_all[start+j*2**i+k])
                hist_all[start+j*2**i+k+1] = norm_and_weight(weight, hist_all[start+j*2**i+k+1])
                hist_all[start+(j+1)*2**i+k] = norm_and_weight(weight, hist_all[start+(j+1)*2**i+k])
                hist_all[start+(j+1)*2**i+k+1] = norm_and_weight(weight, hist_all[start+(j+1)*2**i+k+1])
                if i == 1:
                    hist_all[-1] = norm_and_weight(weight, hist_all[-1])
        start = start+4**i
    hist_all = np.stack(hist_all,axis=0)
    return hist_all
    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    '''

    # ----- TODO -----
    img_file = join(opts.data_dir, img_path)
    img = Image.open(img_file)
    img = np.array(img).astype(np.float32)/255
    filter_responses = visual_words.extract_filter_responses(opts, img)
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    feature = get_feature_from_wordmap_SPM(opts,wordmap)
    return feature

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    # ----- TODO -----
    
    pool = multiprocessing.Pool(n_worker)
    features = [pool.apply(get_image_feature, args=(opts, file, dictionary)) for file in train_files]
    pool.close()

    ## example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )
    pass

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * hist_dist: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    word_hist = word_hist.reshape(1,-1)
    histograms = histograms.reshape(histograms.shape[0],-1)
    hist_dist = np.minimum(word_hist,histograms)
    hist_dist = np.sum(hist_dist, axis=1)
    return hist_dist

def predict_label(opts, img_path, features, dictionary, train_labels):
    test_feature = get_image_feature(opts, img_path, dictionary)
    return train_labels[np.argmax(distance_to_set(test_feature, features))]
    
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']
    features = trained_system['features']
    train_labels = trained_system['labels']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # ----- TODO -----
    matrix = np.zeros((8,8))
    pool = multiprocessing.Pool(n_worker)
    for i in range(0,len(test_files)):
        pre_label = pool.apply(predict_label, args=(opts, test_files[i], features, dictionary, train_labels))
        matrix[test_labels[i]][pre_label] += 1
    pool.close()
    return matrix, (np.trace(matrix)/len(test_files))

