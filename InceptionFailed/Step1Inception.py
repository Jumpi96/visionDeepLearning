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
image_size = 128
rgb = 3
numbers_per_img = 3

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
	
[training_cv_images,training_cv_labels]=create_3mnist(training_cv_images,training_cv_labels) #FUNCIONAA
[test_images,test_labels]=create_3mnist(test_images,test_labels)

#Creando training set y cv set
training_images=training_cv_images[0:15500]
training_labels=training_cv_labels[0:15500]
cv_images=training_cv_images[15500:20000]
cv_labels=training_cv_labels[15500:20000]


#####################2DA PARTE######################

batch_size = 16
depth = 8
num_labels=10
num_hidden=64 
features=48048

def accuracy(predictions, labels): 
    correct_uno=np.sum(np.argmax(predictions[:,0,:],1) == np.argmax(labels[:,0,:],1))
    correct_dos=np.sum(np.argmax(predictions[:,1,:],1) == np.argmax(labels[:,1,:],1))
    correct_tres=np.sum(np.argmax(predictions[:,2,:],1) == np.argmax(labels[:,2,:],1))
    return (100.0 * (correct_uno+correct_dos+correct_tres) / predictions.shape[0]*predictions.shape[1])

def reformat_labels(data):
    labels = np.zeros((len(data),3,10),dtype=np.float32)
    labels = (np.arange(num_labels) == data[:,:,None]).astype(np.float32)
    return labels

    
training_labels=reformat_labels(training_labels)
cv_labels=reformat_labels(cv_labels)
test_labels=reformat_labels(test_labels)
#MODELO
#1 - 1x1 convolution
#2 - 1x1 --> 3x3 convolution
#3 - 1x1 --> 5x5 convolution - DOING ---> COMPLETAR ESTA SECCION



graph=tf.Graph()

with graph.as_default():
    
    #Data
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, rgb))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, numbers_per_img, num_labels))
    tf_cv_dataset = tf.constant(cv_images,dtype=tf.float32)
    tf_test_dataset = tf.constant(test_images,dtype=tf.float32)
	
    #Variables
    path1_weights=tf.Variable(tf.truncated_normal([1,1,rgb,depth],stddev=0.1))
    path1_biases=tf.Variable(tf.zeros([depth]))
    
    path2_weights_conv1=tf.Variable(tf.truncated_normal([1,1,rgb,depth],stddev=0.1))
    path2_biases_conv1=tf.Variable(tf.zeros([depth]))
    path2_weights_conv2=tf.Variable(tf.truncated_normal([3,3,depth,depth],stddev=0.1))
    path2_biases_conv2=tf.Variable(tf.constant(1.0,shape=[depth]))
    
    path3_weights_conv1=tf.Variable(tf.truncated_normal([1,1,rgb,depth],stddev=0.1))
    path3_biases_conv1=tf.Variable(tf.zeros([depth]))
    path3_weights_conv2=tf.Variable(tf.truncated_normal([5,5,depth,depth],stddev=0.1))
    path3_biases_conv2=tf.Variable(tf.constant(1.0,shape=[depth]))
    
    path4_weights_conv1=tf.Variable(tf.truncated_normal([3,3,rgb,depth],stddev=0.1))
    path4_biases_conv1=tf.Variable(tf.zeros([depth]))
    path4_weights_conv2=tf.Variable(tf.truncated_normal([1,1,depth,depth],stddev=0.1))
    path4_biases_conv2=tf.Variable(tf.constant(1.0,shape=[depth]))
    
    fully_connected_weights=tf.Variable(tf.truncated_normal([features, num_hidden], stddev=0.1))
    fully_connected_biases=tf.constant(1.0,shape=[num_hidden])
    
    classifier1_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    classifier1_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
    classifier2_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    classifier2_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
    classifier3_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    classifier3_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
	
    #Model
    def model(data): #Probando stride 2 en 1x1 convolution
        conv1=tf.nn.conv2d(data,path1_weights,[1,2,2,1],padding='VALID')
        hidden1=tf.nn.relu(conv1+path1_biases)
        shape1=hidden1.get_shape().as_list()
        reshape1 = tf.reshape(hidden1, [shape1[0], shape1[1] * shape1[2] * shape1[3]])
        
        conv2_1=tf.nn.conv2d(data,path2_weights_conv1,[1,2,2,1],padding='VALID')
        hidden2_1=tf.nn.relu(conv2_1+path2_biases_conv1)
        conv2_2=tf.nn.conv2d(hidden2_1,path2_weights_conv2,[1,2,2,1],padding='VALID')
        hidden2_2=tf.nn.relu(conv2_2+path2_biases_conv2)
        shape2=hidden2_2.get_shape().as_list()
        reshape2 = tf.reshape(hidden2_2, [shape2[0], shape2[1] * shape2[2] * shape2[3]])
        
        conv3_1=tf.nn.conv2d(data,path3_weights_conv1,[1,2,2,1],padding='VALID')
        hidden3_1=tf.nn.relu(conv3_1+path3_biases_conv1)
        conv3_2=tf.nn.conv2d(hidden3_1,path3_weights_conv2,[1,2,2,1],padding='VALID')
        hidden3_2=tf.nn.relu(conv3_2+path3_biases_conv2)                        
        shape3=hidden3_2.get_shape().as_list()
        reshape3 = tf.reshape(hidden3_2, [shape3[0], shape3[1] * shape3[2] * shape3[3]])
        
        conv4_1=tf.nn.conv2d(data,path4_weights_conv1,[1,3,3,1],padding='VALID')
        hidden4_1=tf.nn.relu(conv4_1+path4_biases_conv1)
        pool4_1=tf.nn.max_pool(hidden4_1, [1,3,3,1], [1,3,3,1], padding='VALID')
        conv4_2=tf.nn.conv2d(pool4_1,path4_weights_conv2,[1,2,2,1],padding='VALID')
        hidden4_2=tf.nn.relu(conv4_2+path4_biases_conv2)
        shape4=hidden4_2.get_shape().as_list()
        reshape4 = tf.reshape(hidden4_2, [shape4[0], shape4[1] * shape4[2] * shape4[3]])        
        
        vector=tf.concat(1,[reshape1,reshape2,reshape3,reshape4])
        fc_hidden=tf.nn.relu(tf.matmul(vector,fully_connected_weights)+fully_connected_biases)
        
        matmul1=tf.matmul(fc_hidden,classifier1_weights)+classifier1_biases
        matmul2=tf.matmul(fc_hidden,classifier2_weights)+classifier2_biases
        matmul3=tf.matmul(fc_hidden,classifier3_weights)+classifier3_biases

        return [matmul1,matmul2,matmul3]
		
    #Training
    logits=model(tf_train_dataset)
    loss=tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=logits))
		
    #Optimizador
    optimizer = tf.train.GradientDescentOptimizer(0.00625).minimize(loss) #Estaba en 0.05
		
    #Predicciones
    train_prediction=tf.nn.softmax(logits)
    cv_prediction=tf.nn.softmax(model(tf_cv_dataset))
    cv_prediction=tf.transpose(cv_prediction,perm=[1,0,2])
    test_prediction=tf.nn.softmax(model(tf_test_dataset))
    test_prediction=tf.transpose(test_prediction,perm=[1,0,2])
	
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
            predictions=np.transpose(predictions,axes=(1,0,2))
            if (step%50==0):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(
		      cv_prediction.eval(), cv_labels))
                print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

			