#Paquetes necesarios
from __future__ import print_function
from array import array as pyarray
import struct
import numpy as np
import tensorflow as tf
#from six.moves import cPickle as pickle
from six.moves import range

####################1RA PARTE########################

#Archivos descargados
trainandcv_filename='D:/Code/Udacity - Deep Learning/Project/MNIST/train-images.idx3-ubyte'
trainandcv_lbl='D:/Code/Udacity - Deep Learning/Project/MNIST/train-labels.idx1-ubyte'
test_filename='D:/Code/Udacity - Deep Learning/Project/MNIST/t10k-images.idx3-ubyte'
test_lbl='D:/Code/Udacity - Deep Learning/Project/MNIST/t10k-labels.idx1-ubyte'

#Cargar MNIST
def load_mnist(images_filename,labels_filename):
	file_images=open(images_filename, 'rb')
	magic_nr, size,rows,cols= struct.unpack(">IIII",file_images.read(16))
	img=pyarray("B",file_images.read())
	file_images.close()
	
	file_labels=open(labels_filename, 'rb')
	magic_nr, size = struct.unpack(">II",file_labels.read(8))
	lbl = pyarray("b", file_labels.read())
	file_labels.close()
	
	digits=np.arange(10)
	ind = [k for k in range(size) if lbl[k] in digits]
	N=len(ind)
	
	images=np.zeros((N, rows, cols),dtype=np.uint8)
	labels=np.zeros((N,1),dtype=np.int8)
	for i in range(len(ind)):
		images[i]=np.array(img[ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
		labels[i]=lbl[ind[i]]

	return images,labels

[training_cv_images,training_cv_labels]=load_mnist(trainandcv_filename,trainandcv_lbl) #CARGANDO BIEN!!!
[test_images,test_labels]=load_mnist(test_filename,test_lbl)

#Método para concatenar imagenes y crear nuevas 20000 imagenes con 3
image_size = 28 #128
rgb = 1
numbers_per_img = 1

def create_3mnist(images,labels):
	final_images=len(images)//numbers_per_img
	
	images_3 = np.ndarray(shape=(len(images)//numbers_per_img,image_size,image_size,rgb),dtype=np.uint8)
	labels_3 = np.ndarray(shape=(len(labels)//numbers_per_img,numbers_per_img))
	
	for i in range(final_images):
		newImage=np.concatenate((np.zeros((28,22),dtype=np.uint8),images[i],images[i+final_images],images[i+final_images*2],np.zeros((28,22),dtype=np.uint8)),axis=1)
		newImage=np.concatenate((np.zeros((50,128),dtype=np.uint8),newImage,np.zeros((50,128),dtype=np.uint8)),axis=0)
		images_3[i,:,:,0]=newImage[:,:]
		images_3[i,:,:,1]=newImage[:,:]
		images_3[i,:,:,2]=newImage[:,:]
		
		labels_3[i]=[labels[i],labels[i+final_images],labels[i+final_images*2]]
		
	return images_3,labels_3
 
def create_mnist(images,labels):	
	images_ = np.ndarray(shape=(len(images)//numbers_per_img,image_size,image_size,rgb),dtype=np.uint8)
	images_[:,:,:,0]=images[:,:,:]
	images_[:,:,:,1]=images[:,:,:]
	images_[:,:,:,2]=images[:,:,:]
		
	return images_,labels
	

#Creando training set y cv set
training_images=training_cv_images[0:45000]
training_labels=training_cv_labels[0:45000]
cv_images=training_cv_images[45000:60000]
cv_labels=training_cv_labels[45000:60000]



#####################2DA PARTE######################

batch_size = 16
patch_size= 5
depth = 16
num_labels=10
num_hidden=128
features=256

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def reformat_labels(dataset,labels):
    dataset = dataset.reshape(
                              (-1, image_size, image_size, rgb)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    labels=labels[:,0,:]
    return dataset,labels
    
[training_images,training_labels]=reformat_labels(training_images,training_labels)
[cv_images,cv_labels]=reformat_labels(cv_images,cv_labels)
[test_images,test_labels]=reformat_labels(test_images,test_labels)
 

#MODELO
#INPUT => CONV => RELU => POOL => CONV => RELU => POOL => FC => RELU => FC

graph=tf.Graph()

with graph.as_default():
    
    #Data
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, rgb))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_cv_dataset = tf.constant(cv_images,dtype=tf.float32)
    tf_test_dataset = tf.constant(test_images,dtype=tf.float32)
	
    #Variables
    #INPUT => CONV => RELU => POOL
    layer1_weights=tf.Variable(tf.truncated_normal([patch_size,patch_size,rgb,depth],stddev=0.1))
    layer1_biases=tf.Variable(tf.zeros([depth]))
    
    layer2_weights=tf.Variable(tf.truncated_normal([patch_size,patch_size,depth,depth],stddev=0.1))
    layer2_biases= tf.Variable(tf.constant(1.0, shape=[depth]))
        
    fully_connected_weights=tf.Variable(tf.truncated_normal([features, num_hidden], stddev=0.1))
    fully_connected_biases=tf.constant(1.0,shape=[num_hidden])    
    
    classifier_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    classifier_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
    
    #Model
    def model(data):
        #INPUT => CONV => RELU => POOL
        conv=tf.nn.conv2d(data,layer1_weights,[1,2,2,1], padding='SAME')
        hidden=tf.nn.relu(conv+layer1_biases)
        pool=tf.nn.max_pool(hidden, [1,2,2,1], [1,2,2,1], padding='SAME')
        #CONV => RELU => POOL
        conv=tf.nn.conv2d(pool,layer2_weights,[1,1,1,1], padding='SAME')
        hidden=tf.nn.relu(conv+layer2_biases)
        pool=tf.nn.max_pool(hidden, [1,2,2,1], [1,2,2,1], padding='SAME')
        
        #FC => RELU => FC
        shape=pool.get_shape().as_list()
        reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape,fully_connected_weights)+fully_connected_biases)
        
        #DROPOUT
        #dropout=0.50      
        #hidden = tf.nn.dropout(hidden, dropout)
        
        return tf.matmul(hidden,classifier_weights)+classifier_biases
		
    #Training
    logits=model(tf_train_dataset)
    loss=tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=logits))
		
    #Optimizador
    optimizer = tf.train.GradientDescentOptimizer(0.00312).minimize(loss)
		
    #Predicciones
    train_prediction=tf.nn.softmax(logits)
    cv_prediction=tf.nn.softmax(model(tf_cv_dataset))
    test_prediction=tf.nn.softmax(model(tf_test_dataset))
	
    #Ejecucion
    num_steps=1001
    
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Iniciado')
        for step in range(num_steps):
            offset=(step*batch_size)%(training_labels.shape[0]-batch_size)
            batch_data=training_images[offset:(offset+batch_size),:,:,:]
            batch_labels=training_labels[offset:(offset+batch_size),:]
            feed_dict = {tf_train_dataset:batch_data,tf_train_labels:batch_labels}
            _,l,predictions=session.run([optimizer,loss,train_prediction],
				feed_dict=feed_dict)
            if (step%50==0):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(
		      cv_prediction.eval(), cv_labels))
                print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

			