#Caroline Mazini Rodrigues 211854

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from multiprocessing import Process
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.externals import joblib
from tqdm import tqdm
import time
import numpy as np
import scipy as sc
from sys import argv
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cluster import KMeans 
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras import optimizers
from keras.layers import AveragePooling2D, GlobalAveragePooling2D, Input
from keras.models import Model
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering
import math
from sklearn.semi_supervised import LabelPropagation,LabelSpreading


dir = './Grenfell/train/'
dir_val = './Grenfell/validation/'

def loadNetwork(train_layer):
    
    print("Loading VGG16 weights from Places365...")
    feature_extractor = VGG16(include_top=True, weights='vgg16-places365_weights_tf_dim_ordering_tf_kernels.h5', input_tensor=None, input_shape=None, pooling=None, classes=365)

    # Freeze the layers except the last train_layers
    for layer in feature_extractor.layers[:-train_layer]:
        layer.trainable = False

    for layer in feature_extractor.layers:
        print(layer, layer.trainable)

    
    feature_extractor.layers.pop()
    feature_extractor.outputs = [feature_extractor.layers[-1].output]
    feature_extractor.layers[-1].outbound_nodes = []
    print("Loading done.")
    
    print("Creating top...")
    #changing the last softmax layer
    x = feature_extractor.layers[-1].output
    x = Dropout(0.5)(x)
    x = Dense(4, activation='softmax', name='predictions')(x)
    feature_extractor = Model(inputs=feature_extractor.inputs, outputs=x)

    feature_extractor.summary()

    print("Top created and added.")

    return feature_extractor






#generate data from training and validation without augmentation
def generateAugData():

    print("Generating features...")

    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30.,
        brightness_range=[0.6,1.4],
        zoom_range=0.2,
        fill_mode='nearest'
    )
 
    generator = datagen.flow_from_directory(
        dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical'
        )

    val_datagen = ImageDataGenerator(
    #rescale=1./255,
        preprocessing_function=preprocess_input
        )

    val_generator = val_datagen.flow_from_directory(
        dir_val,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        shuffle = False)

    return generator, val_generator


#generate data from training and validation without augmentation
def generateData():

    print("Generating features...")

    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
 
    generator = datagen.flow_from_directory(
        dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        shuffle = False)

    val_datagen = ImageDataGenerator(
    #rescale=1./255,
        preprocessing_function=preprocess_input
        )

    val_generator = val_datagen.flow_from_directory(
        dir_val,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        shuffle = False)

    return generator, val_generator

#use the loaded network to generate features without training
def generateFeatures(feature_extractor):

    #exclude the last softmax layer
    feature_extractor.layers.pop()
    feature_extractor.outputs = [feature_extractor.layers[-1].output]
    feature_extractor.layers[-1].outbound_nodes = []
    x = feature_extractor.layers[-1].output

    feature_extractor2 = Model(inputs=feature_extractor.inputs, outputs=x)

    return feature_extractor2

#train the last layers
def fineTuningModel(generator,val_generator,feature_extractor):

    feature_extractor.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['categorical_accuracy','fmeasure'])
    # Train the model

    history = feature_extractor.fit_generator(
      generator,
      steps_per_epoch=4*generator.samples/generator.batch_size ,
      epochs=2,
      validation_data=val_generator,
      validation_steps=val_generator.samples/val_generator.batch_size,
      verbose=1)


    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    fmeasure = history.history['fmeasure']
    val_fmeasure = history.history['val_fmeasure']
 
    epochs = range(len(acc))
 
    plt.plot(epochs, acc, 'b*-', label='Training acc',)
    plt.plot(epochs, val_acc, 'r*-', label='Validation acc')
    plt.grid(True)     
    plt.title('Training and Validation Normalized Accuracy')
    plt.legend()
    plt.savefig('accuracy.png')
 
    plt.figure()
 
    plt.plot(epochs, fmeasure, 'b*-', label='Training fmeasure')
    plt.plot(epochs, val_fmeasure, 'r*-', label='Validation fmeasure')
    plt.grid(True)
    plt.title('Training and Validation F-measure')
    plt.legend()

    plt.savefig('fmeasure.png')
 
    plt.show()
 
    # Save the model
    feature_extractor.save('placesVGG16fineTuned.h5')


    return feature_extractor

#classification after finetuning
def classification(feature_extractor,generator,val_generator):

    feature_extractor = fineTuningModel(generator,val_generator,feature_extractor)

    features = feature_extractor.predict_generator(val_generator, steps=val_generator.samples/val_generator.batch_size,verbose=1)

    predicted_classes = np.argmax(features,axis=1)

    return predicted_classes


#use the network as feature extractor then train a SVM
def classificationSVM(feature_extractor,generator,val_generator, c):#c=9 melhor

    features = feature_extractor.predict_generator(generator, steps=generator.samples/generator.batch_size,verbose=1)

    svm = SVC(C=c, kernel='rbf')
    classifier = svm.fit(features, generator.classes)

    val_features = feature_extractor.predict_generator(val_generator, steps=val_generator.samples/val_generator.batch_size,verbose=1)


    predicted_classes = classifier.predict(val_features)
    
    return predicted_classes

#use the network as feature extractor then cluster samples
def clusterization(feature_extractor,val_generator,seeds,k):
    
    features = feature_extractor.predict_generator(val_generator, steps=val_generator.samples/val_generator.batch_size,verbose=1)

    #without using initialized seeds 
    if(seeds==None):
        seeds = 'k-means++'
    #using initialized seeds per cluster
    else: 
        seeds = createSeeds(seeds,features)
    
    kM = KMeans(n_clusters=k, init = seeds).fit(features)

    return kM.labels_
    
#use the network as feature extractor then cluster samples for different number os clusters and generative elbow curve
def elbow(feature_extractor,val_generator,k):
    KM = []

    features = feature_extractor.predict_generator(val_generator, steps=val_generator.samples/val_generator.batch_size,verbose=1)

    for k in range(2,k):
        kM = KMeans(n_clusters=k).fit(features)
        KM.append(math.sqrt((kM.inertia_)/val_generator.samples))


    # elbow curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(2,k+1), KM, 'b*-')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Finding Elbow: Average distance')
    plt.savefig('2-'+str(k)+'.png') 
    plt.show()

#use the network as feature extractor then train a label spreding (semi supervised)
def semiLabelSpreding(feature_extractor,generator,val_generator, kernel, neighbors, gamma,alpha):
    semi = LabelSpreading(kernel=kernel, n_neighbors=neighbors, gamma=gamma, alpha=alpha, tol=0.001, max_iter=1000000)
    
    features = feature_extractor.predict_generator(generator, steps=generator.samples/generator.batch_size,verbose=1)

    classes = generator.classes

    for i in range(0,generator.samples):
        if(generator.filenames[i][0]=='N'):
            classes[i] = -1 

    semi.fit(features,classes)

    val_features = feature_extractor.predict_generator(val_generator, steps=val_generator.samples/val_generator.batch_size,verbose=1)
    predicted_classes = semi.predict(val_features)

    return predicted_classes

#use the network as feature extractor then train a label propagation (semi supervised)
def semiLabelPropagation(feature_extractor,generator,val_generator, kernel, neighbors, gamma):
    semi = LabelPropagation(kernel=kernel, n_neighbors=neighbors, gamma=gamma, alpha=None, tol=0.001, max_iter=1000000)

    features = feature_extractor.predict_generator(generator, steps=generator.samples/generator.batch_size,verbose=1)

    classes = generator.classes

    for i in range(0,generator.samples):
        if(generator.filenames[i][0]=='N'):
            classes[i] = -1 

    semi.fit(features,classes)

    val_features = feature_extractor.predict_generator(val_generator, steps=val_generator.samples/val_generator.batch_size,verbose=1)
    predicted_classes = semi.predict(val_features)

    return predicted_classes

#create matrix of initial seeds for clustering
def createSeeds(index, features):

    seeds = np.empty([len(index),4096])

    for i in range(0,len(index)):
        seeds[i,:] = features[index[i],:]

    return seeds

#calculate measures F1, normalized accuracy and confusion matrix for predictions
def measures(ground_truth,predicted_classes):

    print(classification_report(ground_truth, predicted_classes))

    print('Accuracy: '+str(accuracy_score(ground_truth, predicted_classes, normalize=True, sample_weight=None)))

    count = np.zeros((6,6))

    for i in range(0, val_generator.samples):
        count[int(predicted_classes[i]),ground_truth[i]] = count[int(predicted_classes[i]),ground_truth[i]]+1

    print(count)

#clustering with num times and take majoritary vote
def voteCluster(feature_extractor,val_generator, num,index,k):

    features = feature_extractor.predict_generator(val_generator, steps=val_generator.samples/val_generator.batch_size,verbose=1)

    labels = np.empty([num, val_generator.samples])

    seeds = np.empty([k,4096])

    for i in range(0,num):

        for j in range(0,k):
            seeds[j,:] = features[index[j],:]
            index[j] = index[j]+1

        kM = KMeans(n_clusters=k, init = seeds).fit(features)
        labels[i,:] = kM.labels_

    classes = sc.stats.mode(labels,axis=0)[0]

    predicted_classes = np.empty(val_generator.samples)

    for i in range(0,val_generator.samples):
        predicted_classes[i] = int(classes[0,i])

   
    return predicted_classes

##################################MAIN#####################################


feature_extractor = loadNetwork(3)

#used if CNN will just extract features, not used in classification method 
feature_extractor = generateFeatures(feature_extractor)

[generator, val_generator] = generateData()

#[generator, val_generator] = generateAugData()


#supervised approaches#############################

#only method that will not use generateFeatures, this will train the network instead
#predicted_classes = classification(feature_extractor,generator,val_generator)

predicted_classes = classificationSVM(feature_extractor,generator,val_generator, 5)

###################################################

#unsupervised approaches###########################

#elbow(feature_extractor,val_generator,18)

#the seeds are used for clusterization, if seeds=None it will be random atributed
#seeds = None
#seeds = [0,60, 120, 180, 240,360]
#seeds = [30,90, 130, 150, 250,360]
#seeds = [10,50, 140, 170, 260,380]


#predicted_classes = clusterization(feature_extractor,val_generator,seeds,6)
#predicted_classes = voteCluster(feature_extractor,val_generator, 3, seeds,6)


###################################################


#semi-supervised approaches########################

#predicted_classes = semiLabelPropagation(feature_extractor,generator,val_generator, 'knn', 5, 20)

#predicted_classes = semiLabelSpreding(feature_extractor,generator,val_generator, 'knn', 3, 20,0.3)

###################################################


#evatuation
ground_truth = val_generator.classes

measures(ground_truth,predicted_classes)
