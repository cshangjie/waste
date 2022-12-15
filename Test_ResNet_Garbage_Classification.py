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
from tensorflow.keras.utils import to_categorical
import seaborn as sns

tensorflow.random.set_seed(0)
np.random.seed(0)

path = "./dataset/Garbage_Dataset/Garbage_Dataset"

test_datagen = ImageDataGenerator(
        rescale = 1./255
)

img_shape = (224, 224, 3) # default values

val_batch_size = 32


test_generator = test_datagen.flow_from_directory(
            path + '/test',
            target_size = (img_shape[0], img_shape[1]),
            batch_size = val_batch_size,
            class_mode = 'categorical',
            shuffle=False,)

'''Load model in'''
model = keras.models.load_model(".\ResNet_Garbage_Classifier.h5")

"""# Prediction on test set"""

# data = np.load('../input/test-data/test_data.npz')
# x_test, y_test = data['x_test'], data['y_test']
"""# Model evaluation

## Test set accuracy
"""

test_path = './dataset/Garbage_Dataset/Garbage_Dataset/test'

dict = {
        'cardboard' : 0, 
        'glass' : 1,    
        'metal' : 2, 
        'paper' : 3, 
        'plastic' : 4, 
        'trash' : 5
}

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


Y_test = (to_categorical(y_test))
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

from sklearn.metrics import classification_report
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,preds))
print(confusion_matrix(y_test,preds))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,preds)

LABELS=['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
plt.figure(figsize=(10,8))
sns.heatmap(cm, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt = 'd')
plt.title('Confusion Matrix')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.savefig("ResNet_Confusion_Matrix.png")
plt.show()

