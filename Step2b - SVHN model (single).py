#Packages
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import range
import h5py

#Loading data

filename='D:/Code/Udacity - Deep Learning/Project/SVHN2/SVHN_simple.h5'
h5f = h5py.File(filename, 'r')

training_images = h5f['training_images'][:]
training_labels = h5f['training_labels'][:]
test_images = h5f['test_images'][:]
test_labels = h5f['test_labels'][:]
cv_images = h5f['cv_images'][:]
cv_labels = h5f['cv_labels'][:]

h5f.close()

#Parameters
image_size=32
batch_size = 64
patch_size= 5
depth = 32
rgb=1
depth2 = 64
num_labels=10
num_hidden=256
features=4096

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

#MODEL
#INPUT => CONV => RELU => POOL => CONV => RELU => POOL => FC => RELU => FC

graph=tf.Graph()

with graph.as_default():
    
    #Data
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, rgb))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_cv_dataset = tf.constant(cv_images,dtype=tf.float32)
    tf_test_dataset = tf.constant(test_images,dtype=tf.float32)
	
    #Variables
    layer1_weights=tf.Variable(tf.truncated_normal([patch_size,patch_size,rgb,depth],stddev=0.1))
    layer1_biases=tf.Variable(tf.zeros([depth]))
    
    layer2_weights=tf.Variable(tf.truncated_normal([patch_size,patch_size,depth,depth2],stddev=0.1))
    layer2_biases= tf.Variable(tf.constant(1.0, shape=[depth2]))
        
    fully_connected_weights=tf.Variable(tf.truncated_normal([features, num_hidden], stddev=0.1))
    fully_connected_biases=tf.constant(1.0,shape=[num_hidden])
    
    classifier_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    classifier_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
    
    #Model
    def model(data,train=True):
        #INPUT => CONV => RELU => POOL
        conv=tf.nn.conv2d(data,layer1_weights,[1,1,1,1], padding='SAME')
        hidden=tf.nn.relu(conv+layer1_biases)
        pool=tf.nn.max_pool(hidden, [1,2,2,1], [1,2,2,1], padding='SAME')

        #CONV => RELU => POOL
        conv=tf.nn.conv2d(pool,layer2_weights,[1,1,1,1], padding='SAME')
        hidden=tf.nn.relu(conv+layer2_biases)
        pool=tf.nn.max_pool(hidden, [1,2,2,1], [1,2,2,1], padding='SAME')
        
        #Dropout
        if train:
            dropout=0.50      
            pool = tf.nn.dropout(pool, dropout)            
        
        
        #Reshape
        shape=pool.get_shape().as_list()
        reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
        
        #FC -> FC 
        layer=tf.matmul(reshape,fully_connected_weights) + fully_connected_biases
        hidden=tf.nn.relu(layer)
        
        return tf.matmul(hidden,classifier_weights)+classifier_biases
		
    logits=model(tf_train_dataset)

    loss=tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=tf_train_labels))
		
    #Learning rate decay
    global_step = tf.Variable(0)
    learning_rate=tf.train.exponential_decay(
                                             0.05,
                                             global_step,
                                             10000,
                                             0.96,
                                             staircase=True)

    #Optimizer
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss,global_step=global_step)
		
    #Predictions
    train_prediction=tf.nn.softmax(logits)
    cv_prediction=tf.nn.softmax(model(tf_cv_dataset,False))
    test_prediction=tf.nn.softmax(model(tf_test_dataset,False))
	
    num_steps=9375 #1 epoch
    
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Iniciado')
        for step in range(num_steps):
            offset=(step*batch_size)%(training_labels.shape[0]-batch_size)
            batch_data=training_images[offset:(offset+batch_size),:,:,:]
            batch_labels=training_labels[offset:(offset+batch_size),:]
            feed_dict = {tf_train_dataset:batch_data,tf_train_labels:batch_labels}
            _,l,lr,predictions=session.run([optimizer,loss,learning_rate,train_prediction],
				feed_dict=feed_dict)
            if (step%500==0):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch learning rate: %.6f' % (lr))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(
		      cv_prediction.eval(), cv_labels))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

			