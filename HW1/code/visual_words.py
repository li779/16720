import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy

def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    filter_scales = opts.filter_scales
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
        img = np.concatenate([img for i in range(0,3)],2)
    
    
    filter_nums = len(filter_scales)
    shape = img.shape+tuple([filter_nums*4])
    filter_responses = np.empty(shape)
    img = skimage.color.rgb2lab(img)
    for i in range(0,3):
        for j in range(0,filter_nums):
            filter_responses[:,:,i,0+filter_nums*j] = scipy.ndimage.gaussian_filter(img[:,:,i],sigma = filter_scales[j],order=0)
            filter_responses[:,:,i,1+filter_nums*j] = scipy.ndimage.gaussian_laplace(img[:,:,i],sigma = filter_scales[j])
            filter_responses[:,:,i,2+filter_nums*j] = scipy.ndimage.gaussian_filter(img[:,:,i],sigma = filter_scales[j],order=(0,1))
            filter_responses[:,:,i,3+filter_nums*j] = scipy.ndimage.gaussian_filter(img[:,:,i],sigma = filter_scales[j],order=(1,0))
    filter_responses = np.reshape(filter_responses,(img.shape[0],img.shape[1],-1))
    return filter_responses

def compute_dictionary_one_image(opts,filename):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # ----- TODO -----
    img_path = join(opts.data_dir, filename)
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    filter_responses = extract_filter_responses(opts,img)
    res = np.empty((opts.alpha,filter_responses.shape[2]));
    for i in range(0,opts.alpha):
        x = np.random.randint(img.shape[0]-1,size=1)
        y = np.random.randint(img.shape[1]-1,size=1)
        res[i,:] = filter_responses[x,y,:]
    return res

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    # ----- TODO -----
    train_nums = len(train_files)
    print(str(train_nums)+"training samples")
    pool = multiprocessing.Pool(n_worker)
    response = [pool.apply(compute_dictionary_one_image, args=(opts, i)) for i in train_files]
    pool.close()  
    filter_responses = np.concatenate(response,axis=0)
    #print(filter_responses.shape)
    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(filter_responses)
    dictionary = kmeans.cluster_centers_
    np.save(join(out_dir, 'dictionary.npy'), dictionary)
    pass

    ## example code snippet to save the dictionary
    # np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO ----
    response = extract_filter_responses(opts,img)
    wordmap = np.empty(img.shape[0:2])
    for row in range(0,img.shape[0]):
        for col in range(0,img.shape[1]):
            distance = scipy.spatial.distance.cdist(np.expand_dims(response[row,col,:],axis=0),dictionary)
            wordmap[row,col] = np.argmin(distance)
    return wordmap
    
    

