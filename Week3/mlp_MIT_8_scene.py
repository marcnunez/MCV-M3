import os
import getpass
import sys

from colored import stylize, fg

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import xlsxwriter

from scipy.misc import imresize


def main():
    
    # user defined variables
    IMG_SIZE = 32
    BATCH_SIZE = 16
    if (len(sys.argv)>0):
        print(len(sys.argv))
        DIRECTORY_PATH = str(sys.argv[1])
    else:
        DIRECTORY_PATH = "one"
    
    workbook = xlsxwriter.Workbook('/home/grupo07/week3/out/' + DIRECTORY_PATH + '.xlsx')
    worksheet = workbook.add_worksheet()

    
    DATASET_DIR = '/home/grupo07/mcv/datasets/MIT_split'
    ACTIVATIONS = ['selu', 'softsign', 'relu', 'tanh', 'hard_sigmoid', 'exponential']
    for ACTIVATION in ACTIVATIONS:
        RELATIVE_PATH = '/home/grupo07/week3/out/' + DIRECTORY_PATH + '/' + ACTIVATION + '/'

        for UNIT in range(9,12):
            UNIT_POW = 2**UNIT
            
            if not os.path.exists(DATASET_DIR):
                print(stylize('ERROR: dataset directory ' + DATASET_DIR + ' do not exists!\n', fg('blue')))
                quit()
            
            print('Building MLP model...\n')
            
            # Build the Multi Layer Perceptron model
            model = Sequential()
            model.add(Reshape((IMG_SIZE * IMG_SIZE * 3,), input_shape=(IMG_SIZE, IMG_SIZE, 3), name='first'))
            model.add(Dense(units=UNIT_POW, activation=ACTIVATION, name='second'))
            if (DIRECTORY_PATH == 'two'):
                model.add(Dense(units=UNIT_POW, activation=ACTIVATION, name='third'))
            elif (DIRECTORY_PATH == 'three'):
                model.add(Dense(units=UNIT_POW, activation=ACTIVATION, name='third'))
                model.add(Dense(units=UNIT_POW, activation=ACTIVATION, name='fourth'))
    
            model.add(Dense(units=8, activation=ACTIVATION, name='last'))
            model.compile(loss='categorical_crossentropy',
                          optimizer='sgd',
                          metrics=['accuracy'])
            
            print(model.summary())
            UNIT_POW = str(UNIT_POW)
            MODEL_FNAME = RELATIVE_PATH + ACTIVATION +'_'+ UNIT_POW + '_mlp.h5'
            plot_model(model, to_file=(RELATIVE_PATH + ACTIVATION +'_'+ UNIT_POW + 'modelMLP.png'), show_shapes=True, show_layer_names=True)
            
            print('Done!\n')


            if os.path.exists(MODEL_FNAME):
                print('WARNING: model file ' + MODEL_FNAME + ' exists and will be overwritten!\n')
            
            print('Start training...\n')
            
            # this is the dataset configuration we will use for training
            # only rescaling
            train_datagen = ImageDataGenerator(
                rescale=1. / 255,
                horizontal_flip=True)
            
            # this is the dataset configuration we will use for testing:
            # only rescaling
            test_datagen = ImageDataGenerator(rescale=1. / 255)
            
            # this is a generator that will read pictures found in
            # subfolers of 'data/train', and indefinitely generate
            # batches of augmented image data
            train_generator = train_datagen.flow_from_directory(
                DATASET_DIR + '/train',  # this is the target directory
                target_size=(IMG_SIZE, IMG_SIZE),  # all images will be resized to IMG_SIZExIMG_SIZE
                batch_size=BATCH_SIZE,
                classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
                class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels
            
            # this is a similar generator, for validation data
            validation_generator = test_datagen.flow_from_directory(
                DATASET_DIR + '/test',
                target_size=(IMG_SIZE, IMG_SIZE),
                batch_size=BATCH_SIZE,
                classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
                class_mode='categorical')
            
            history = model.fit_generator(
                train_generator,
                steps_per_epoch=1881 // BATCH_SIZE,
                epochs=50,
                validation_data=validation_generator,
                validation_steps=807 // BATCH_SIZE)
            
            print('Done!\n')
            print('Saving the model into ' + MODEL_FNAME + ' \n')
            model.save_weights(MODEL_FNAME)  # always save your weights after training or during training
            print('Done!\n')
            
            # summarize history for accuracy
#            plt.plot(history.history['acc'])
#            plt.plot(history.history['val_acc'])
#            plt.title('model accuracy')
#            plt.ylabel('accuracy')
#            plt.xlabel('epoch')
#            plt.legend(['train', 'validation'], loc='upper left')
#            plt.savefig(RELATIVE_PATH  + ACTIVATION +'_'+ UNIT_POW +'_accuracy.jpg')
#            plt.close()
#            # summarize history for loss
#            plt.plot(history.history['loss'])
#            plt.plot(history.history['val_loss'])
#            plt.title('model loss')
#            plt.ylabel('loss')
#            plt.xlabel('epoch')
#            plt.legend(['train', 'validation'], loc='upper left')
#            plt.savefig(RELATIVE_PATH + ACTIVATION +'_'+ UNIT_POW +'_loss.jpg')
            
            
#            
#            # to get the output of a given layer
#            # crop the model up to a certain layer
#            model_layer = Model(inputs=model.input, outputs=model.get_layer('second').output)
#            
#            # get the features from images
#            directory = DATASET_DIR + '/test/coast'
#            x = np.asarray(Image.open(os.path.join(directory, os.listdir(directory)[0])))
#            x = np.expand_dims(imresize(x, (IMG_SIZE, IMG_SIZE, 3)), axis=0)
#            print('prediction for image ' + os.path.join(directory, os.listdir(directory)[0]))
#            features = model_layer.predict(x / 255.0)
#            print(features)
#            print('Done!')
            
    workbook.close()



if __name__ == "__main__":
    main()
