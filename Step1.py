#Paquetes necesarios
from __future__ import print_function
from array import array as pyarray
import struct
import numpy as np
from __future__ import print_function
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

####################1RA PARTE########################

#Archivos descargados
trainandcv_filename='MNIST/train-images.idx3-ubyte'
trainandcv_lbl='MNIST/train-labels.idx1-ubyte'
test_filename='MNIST/t10k-images.idx3-ubyte'
test_lbl='MNIST/t10k-labels.idx1-ubyte'

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
image_size = 128
rgb = 3

def create_3mnist(images,labels):
	numbers_per_img = 3
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
	
[training_cv_images,training_cv_labels]=create_3mnist(training_cv_images,training_cv_labels) #FUNCIONAA
[test_images,test_labels]=create_3mnist(test_images,test_labels)

#Creando training set y cv set
training_images=training_cv_images[0:15500]
training_labels=training_cv_labels[0:15500]
cv_images=training_cv_images[15500:20000]
cv_labels=training_cv_labels[15500:20000]


#####################2DA PARTE######################

def accuracy(predictions, labels): #Copiado, método para un solo numero
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

#MODELO
#1 - 1x1 convolution - TO DO
  
batch_size = 16
patch_size = 5 #?????????????????
depth = 16 #Ke onda???
num_labels=10

graph=tf.Graph

with graph.as_default():
    
    #Data
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, rgb))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_cv_dataset = tf.constant(cv_images)
    tf_test_dataset = tf.constant(cv_labels)
	
	#Variables
	path1_weights=tf.Variable(tf.truncated_normal([patch_size,patch_size,rgb,depth],stddev=0.1))
	path1_biases=tf.Variable(tf.zeros([depth])
	#...
	
	#Model
	def model(data):
		conv=tf.nn.conv2d(data,path1_weights,[1,1,1,1],padding='SAME')
		hidden=tf.nn.relu(conv+layer1_biases)
		#...
		return tf.matmul(hidden,path1_weights)+layer1_biases
		
	#Training
	logits=model(tf_train_dataset)
	loss=tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=logits))
		
	#Optimizador
	optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
		
	#Predicciones
	train_prediction=tf.nn.softmax(logits)
	valid_prediction=tf.nn.softmax(model(tf_valid_dataset))
	test_prediction=tf.nn.softmax(model(tf_test_dataset))
	
	#Ejecucion
	num_steps=1001
	
	with tf.Session(graph=graph) as session:
		tf.global_variables_initializer().run()
		print('Iniciado')
		for step in range(num_steps):
			offset=(step*batch_size)%train_labels.shape[0]-batch_size)
			batch_data=train_dataset[offset:(offset+batch_size),:,:,:]
			batch_labels=train_labels[offset:(offset+batch_size),:]
			feed_dict = {tf_train_dataset:batch_data,tf_train_labels:batch_labels}
			_,l,predictions=session_run([optimizer,loss,train_prediction],
				feed_dict=feed_dict)
			if (step%50==0):
			print('Minibatch loss at step %d: %f' % (step, l))
			print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
			print('Validation accuracy: %.1f%%' % accuracy(
				valid_prediction.eval(), valid_labels))
			print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
		