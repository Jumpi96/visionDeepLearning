#Packages
from __future__ import print_function
import scipy.io
import numpy as np
import tensorflow as tf
from six.moves import range
import h5py

#Downloaded SVHN files
train_filename='D:/Code/Udacity - Deep Learning/Project/SVHN2/train_32x32.mat'
extra_filename='D:/Code/Udacity - Deep Learning/Project/SVHN2/extra_32x32.mat'
test_filename='D:/Code/Udacity - Deep Learning/Project/SVHN2/test_32x32.mat'
new_data_filename='D:/Code/Udacity - Deep Learning/Project/SVHN2/SVHN_simple.h5'

#Load images
def load_svhn(filename):
	mat = scipy.io.loadmat(filename)
	images=mat['X']	
	images=np.transpose(images,axes=(3,0,1,2))
	
	labels=mat['y']

	return images,labels

[training_images,training_labels]=load_svhn(train_filename)
[extra_images,extra_labels]=load_svhn(extra_filename)
[test_images,test_labels]=load_svhn(test_filename)

#Parametros
image_size = 32
rgb = 1
numbers_per_img = 1
num_labels=10

#------Cross-validation balanced dataset creation-------------
def balanced_subsample(y, s):
    sample = []
    for label in np.unique(y):
        images = np.where(y==label)[0]
        random_sample = np.random.choice(images, size=s, replace=False)
        sample += random_sample.tolist()
    return sample
# Pick 400 samples from the training samples
train_samples = balanced_subsample(training_labels, 400)
# Pick 200 samples from the extra dataset
extra_samples = balanced_subsample(extra_labels, 200)

cv_images, cv_labels = np.copy(training_images[train_samples]), np.copy(training_labels[train_samples])

# Avoid duplicates
training_images = np.delete(training_images, train_samples, axis=0)
training_labels = np.delete(training_labels, train_samples, axis=0)

cv_images = np.concatenate([cv_images, np.copy(extra_images[extra_samples])])
cv_labels = np.concatenate([cv_labels, np.copy(extra_labels[extra_samples])])

extra_images = np.delete(extra_images, extra_samples, axis=0)
extra_labels = np.delete(extra_labels, extra_samples, axis=0)

training_images = np.concatenate([training_images, extra_images])
training_labels = np.concatenate([training_labels, extra_labels])

#-----Reformat labels and convert images to grayscale----------
def reformat(dataset,labels):
    dataset[:,:,:,0]=(((dataset[:,:,:,0]*0.2989)+(dataset[:,:,:,1]*0.5870)
            +(dataset[:,:,:,2]*0.1140))/3)
    newDataset=np.zeros((dataset.shape[0],image_size,image_size,rgb),dtype=np.uint8)
    newDataset[:,:,:,0]=dataset[:,:,:,0]

    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    labels=labels[:,0,:]

    return newDataset,labels
    
[training_images,training_labels]=reformat(training_images,training_labels)

[cv_images,cv_labels]=reformat(cv_images,cv_labels)
[test_images,test_labels]=reformat(test_images,test_labels)

#-----Normalization----------
train_mean = np.mean(training_images, axis=0)

train_std = np.std(training_images, axis=0)

training_images = (training_images - train_mean) / train_std
test_images = (test_images - train_mean)  / train_std
cv_images = (train_mean - cv_images) / train_std


#------Save new dataset--------
h5f = h5py.File(new_data_filename, 'w')

h5f.create_dataset('test_images', data=test_images)
h5f.create_dataset('test_labels', data=test_labels)
h5f.create_dataset('cv_images', data=cv_images)
h5f.create_dataset('cv_labels', data=cv_labels)
h5f.create_dataset('training_images', data=training_images)
h5f.create_dataset('training_labels', data=training_labels)

h5f.close()
