import numpy as np
import matplotlib.pyplot as plt, seaborn as sns
# image processing tasks
import cv2
from google.colab.patches import cv2_imshow
# Create the model and train it
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
from keras.datasets import mnist
from tensorflow.math import confusion_matrix


# Loading mnist dataset
(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)
# Show the first 10 images
for i in range(10):
    plt.imshow(x_train[i])
    print("the image shape is:",x_train[i].shape," and the label is:",y_train[i])
    plt.show()
# Show the unique values in y_train,y_test
print(np.unique(y_train))
print(np.unique(y_test))


# Show the first image before scaling
print(x_train[0])
# scaling the values
x_train = x_train/255
x_test = x_test/255
# Show the first image after scaling
print(x_train[0])


# Building convolutional neural network (CNN)
Model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(50,activation='relu'),
    keras.layers.Dense(50,activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')
])
# Compiling the model
Model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# train the model
result = Model.fit(x_train,y_train,validation_split=0.3,epochs=10)
# plot the accuracy with the validation accuracy
plt.figure(figsize=(7,7))
plt.plot(result.history['accuracy'],color='red')
plt.plot(result.history['val_accuracy'],color='blue')
plt.title('model accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()
# Plot the loss with the validation loss
plt.figure(figsize=(7,7))
plt.plot(result.history['loss'],color='red')
plt.plot(result.history['val_loss'],color='blue')
plt.title('model loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

# Avaluation on test data
evaluate = Model.evaluate(x_test,y_test)
print("the loss value is: ",evaluate[0])
print("the accuracy value is: ",evaluate[1])

# Make the model predict on the test input images
predicted_values = Model.predict(x_test)
# Convert the probability predicted values ten digits by argmax into one class label
predicted_values_labels=[np.argmax(value) for value in predicted_values]
print(predicted_values_labels)
print(list(y_test))

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test,predicted_values_labels)
plt.figure(figsize=(5,5))
sns.heatmap(conf_matrix,cbar=True,square=True,fmt='d',annot=True,annot_kws={'size':8},cmap="Blues")
plt.ylabel('true labels')
plt.xlabel('predicted labels')



# Making a predictive system
image_path = input("Enter image path to be predicted: ")
input_image = cv2.imread(image_path)
cv2_imshow(input_image)
# # convert the image to greyscale
greyscale_image = cv2.cvtColor(input_image,cv2.COLOR_RGB2GRAY)
# resize the image into 28*28 
resized_grey_image = cv2.resize(greyscale_image,(28,28))
# scaling the image
resized_grey_image = resized_grey_image/255
# reshape the image
reshaped_image = np.reshape(resized_grey_image,[1,28,28])
# # Make the model predict what is the digit in the image
print("the handwritten digit in the image is:",np.argmax(Model.predict(reshaped_image)))
