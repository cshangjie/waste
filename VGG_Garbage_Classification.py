import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting
import tensorflow
import os
import cv2
import keras
import sklearn 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, SGD, Adam, Nadam
from tensorflow.keras.regularizers import l1, l2, L1L2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

tensorflow.random.set_seed(0)
np.random.seed(0)

path = "./dataset/Garbage_Dataset/Garbage_Dataset"

"""# Data preprocessing"""

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 20,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        horizontal_flip = True,
        vertical_flip = True,
        fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(
        rescale = 1./255
)
test_datagen = ImageDataGenerator(
        rescale = 1./255
)

img_shape = (224, 224, 3) # default values

train_batch_size = 32
val_batch_size = 32

train_generator = train_datagen.flow_from_directory(
            path + '/train',
            target_size = (img_shape[0], img_shape[1]),
            batch_size = train_batch_size,
            class_mode = 'categorical',)

validation_generator = validation_datagen.flow_from_directory(
            path + '/valid',
            target_size = (img_shape[0], img_shape[1]),
            batch_size = val_batch_size,
            class_mode = 'categorical',
            shuffle=False)

test_generator = test_datagen.flow_from_directory(
            path + '/test',
            target_size = (img_shape[0], img_shape[1]),
            batch_size = val_batch_size,
            class_mode = 'categorical',
            shuffle=False,)

"""# Building model

## Pretrained Convolutional Base (VGG16)
"""

vgg = VGG16(weights = 'imagenet',
              include_top = False,
              input_shape = img_shape)

"""## Fine-tuning

### Freeze VGG layers
"""

# Freeze the layers except the last 3 layers
for layer in vgg.layers[:-3]:
    layer.trainable = False

"""### Model definition"""

# Create the model
model = Sequential()
 
# Add the vgg convolutional base model
model.add(vgg)
 
# Add new layers
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(6, activation='softmax'))

model.summary()

"""# Train the model"""

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Nadam(lr=1e-4),
              metrics=['acc'])

# Train the model
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint('VGG16_best_weights.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples/train_generator.batch_size ,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples/validation_generator.batch_size,
    verbose=0,
    callbacks = [mc])

model.save("VGG16_Garbage_Classifier.h5")

"""### Training history"""

train_acc = history.history['acc']
val_acc = history.history['val_acc']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(train_acc) + 1)

plt.plot(epochs, train_acc, 'b*-', label = 'Training accuracy')
plt.plot(epochs, val_acc, 'r', label = 'Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('VGG_train_val_acc.png')
plt.figure()

plt.plot(epochs, train_loss, 'b*-', label = 'Training loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('VGG_train_val_loss.png')
plt.show()

"""# Prediction on test set"""

# data = np.load('../input/test-data/test_data.npz')
# x_test, y_test = data['x_test'], data['y_test']
"""# Model evaluation

## Test set accuracy
"""

test_path = './dataset/Garbage_Dataset/Garbage_Dataset/test'

#Loading train datasets
test_data = []
test_labels = []
classes = 7 #data belonges to 7 class
for i in os.listdir(test_path):
    dir = test_path + '/' + i
    if (i == ".DS_Store"):
      continue
    for j in os.listdir(dir):
        img_path = dir + '/' + j
        img = cv2.imread(img_path,-1)
        img = cv2.resize(img,(224,224),interpolation = cv2.INTER_NEAREST)
        test_data.append(img)
        test_labels.append(dict[i])


test_data = np.array(test_data)
test_labels = np.array(test_labels)
print(test_data.shape, test_labels.shape)

X_test = test_data
y_test = test_labels 


Y_test = (np_utils.to_categorical(y_test))
print(X_test.shape)
print(Y_test.shape)

probs = model.predict(test_generator,steps = len(test_generator), verbose = 1)


preds = np.argmax(probs,axis = 1)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,preds)

# We evaluate the accuracy and the loss in the test set
scores = model.evaluate(test_generator, steps = len(test_generator), verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

LABELS=['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

y_pred = model.predict(test_generator)


prediction = np.argmax(y_pred,axis =1)


y_hat_pred = []
for i in prediction:
  y_hat_pred.append(LABELS[i])

confusion_matrix = np.zeros((6,6), dtype=np.uint8)
per_class_acc = np.zeros(6)
for i in range(y_test.shape[1]):
    idxs = np.argmax(y_test, axis=1)==i
    this_label = y_test[idxs]
    num_samples_per_class = np.count_nonzero(idxs)
    one_hot = tensorflow.one_hot(np.argmax(model.predict(x_test[idxs]), axis=1), depth=6).eval(session=tensorflow.Session())
    confusion_matrix[i] = np.sum(one_hot, axis=0)
    per_class_acc[i] = confusion_matrix[i,i]/num_samples_per_class

LABELS=['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt = 'd')
plt.title('Confusion Matrix')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.savefig("VGG_Confusion_Matrix.png")
plt.show()