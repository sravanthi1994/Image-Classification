#Using Transfer Learning for the Kaggle Yelp Challenge 
#Extracting feature vectors from the AlexNet net at the fully connected layers for each image
#Taking the mean of all feature vectors for all the images that correspond to a business id 
# we have extracted feature vectors at all the fc layers 

import numpy as np
import sys
import h5py
import caffe
import os
import pandas as pd 

path_to_data = '/Users/samantha/yelp/'
path_to_caffe = '/Users/samantha/dev/caffe/'

sys.path.insert(0, path_to_caffe + 'python')
if not os.path.isfile(path_to_caffe + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print("Pretrained CaffeNet model doesnot exist, Downloading ...")
    !/Users/mallem/dev/caffe/scripts/download_model_binary.py /Users/mallem/dev/caffe/models/bvlc_reference_caffenet
print("Downloaded the pretained model")   

#layer takes the values of fc6,fc7,fc7
def get_features(images, layer = 'fc7'):
    net = caffe.Net(path_to_caffe + 'models/bvlc_reference_caffenet/deploy.prototxt',
                path_to_caffe + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)
    
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    Transform = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    Transform.set_transpose('data', (2,0,1))
    Transform.set_mean('data', np.load(path_to_caffe + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) 
    Transform.set_raw_scale('data', 255)  
    Transform.set_channel_swap('data', (2,1,0)) 
    #Resizing the image to 3*227*227
    num_images= len(images)
    net.blobs['data'].reshape(num_images,3,227,227)
    net.blobs['data'].data[...] = map(lambda x: Transform.preprocess('data',caffe.io.load_image(x)), images)
    out = net.forward()

    return net.blobs[layer].data   

# extract image features and save it to .h5
 
trainfeatures = h5py.File(path_to_data+'train_image_fc7features.h5','w')
filenames = trainfeatures.create_dataset('photo_id',(0,), maxshape=(None,),dtype='|S54')
feature = trainfeatures.create_dataset('feature',(0,4096), maxshape = (None,4096))
trainfeatures.close()

train_photos = pd.read_csv(path_to_data+'train_photo_to_biz_ids.csv')
train_folder = path_to_data+'train_photos/'
train_images = [os.path.join(train_folder, str(x)+'.jpg') for x in train_photos['photo_id']]  # Appending .jpg to complete the full filename

trainingimagecount = len(train_images)
print "Number of training images: ", trainingimagecount
interval_size = 500

# Training Images
for i in range(0, trainingimagecount, interval_size): 
    images = train_images[i: min(i+interval_size, trainingimagecount)]
    features = get_features(images, layer='fc7')
    count = i+features.shape[0]
    trainf= h5py.File(path_to_data+'train_image_fc7features.h5','r+')
    trainf['photo_id'].resize((count,))
    trainf['photo_id'][i: count] = np.array(images)
    trainf['feature'].resize((count,features.shape[1]))
    trainf['feature'][i: count, :] = features
    trainf.close()
    if count%10000==0 or count==trainingimagecount:
        print "Done processesisng teh Train Images: ", count

interval_size = 500

testfeatures = h5py.File(path_to_data+'test_image_fc7features.h5','w')
filenames = testfeatures.create_dataset('photo_id',(0,), maxshape=(None,),dtype='|S54')
feature = testfeatures.create_dataset('feature',(0,4096), maxshape = (None,4096))
testfeatures.close()


test_photos = pd.read_csv(path_to_data+'test_photo_to_biz.csv')
test_folder = path_to_data+'test_photos/'
test_images = [os.path.join(test_folder, str(x)+'.jpg') for x in test_photos['photo_id'].unique()]  
testingimagecount = len(test_images)
print "Number of test images: ", testingimagecount

# Repeated for Test Images
for i in range(0, testingimagecount, interval_size): 
    images = test_images[i: min(i+interval_size, testingimagecount)]
    features = extract_features(images, layer='fc7')
    count = i+features.shape[0]
    
    testf= h5py.File(path_to_data+'test_image_fc7features.h5','r+')
    testf['photo_id'].resize((count,))
    testf['photo_id'][i: count] = np.array(images)
    testf['feature'].resize((count,features.shape[1]))
    testf['feature'][i: count, :] = features
    testf.close()
    if count%10000==0 or count==testingimagecount:
        print "Done processing the Test images: ", count

