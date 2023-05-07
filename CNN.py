import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
from keras import models, layers, losses, regularizers


# Confusion matrix for multi-class classification.
def confusion_matrix(Actual, Predicted, n_classes=10):

  """
  Actual : Actual represents an array showing the true class labels for each sample in the dataset.
  Predicted : Actual represents an array showing the Predicted class labels for each sample in the dataset.
  """

  # Creating a zeros matrix for the confusion matrix with the shape of the (n_classes, n_classes) and making it to a float32 data type.
  c_m = np.zeros((n_classes,n_classes), dtype=np.float32)
  # Iterating through each sample in the dataset.
  for i in range(Actual.shape[0]):
    # Increments the elements with the true and predicted labels.
    c_m[Actual[i], Predicted[i]] += 1
  # Returing the confusion matrix
  return c_m


def train_nn_keras(X_train, Y_train, X_test, Y_test, epochs=1, batch_size=4):
  """
  X_train : The input training data for the train_nn_keras function.
  Y_train : The output training data for the train_nn_keras function.
  X_test : The input test data for the test_nn_keras function.
  Y_test : The output test data for the test_nn_keras function.
  epochs : Number of epochs to train.
  batch_size :  Number of batches to train.
  """
  # Defining the random seed provided along with the template code.
  tf.keras.utils.set_random_seed(5368) # do not remove this line

  # Model Architecture
  # Creating the Sequential object to add layers into the network.
  model = models.Sequential()
  # Adding the Conv2D layer to the model with fileters = 8, kernel size = (3, 3), strides = (1,1), padding='same', activation='relu' and a L2 Regularization of 0.0001.
  model.add(layers.Conv2D(filters = 8, kernel_size = (3,3), strides = (1,1), padding='same', activation='relu', input_shape = X_train.shape[1:], kernel_regularizer = regularizers.l2(0.0001)))
  # Adding the Conv2D layer to the model with filters = 16, kernel_size = (3,3), strides = (1,1), padding='same', activation='relu' and a L2 Regularization of 0.0001.
  model.add(layers.Conv2D(filters = 16, kernel_size = (3,3), strides = (1,1), padding='same', activation='relu', kernel_regularizer = regularizers.l2(0.0001)))
  # Adding the Max Pooling layer with a pool size of (2,2), strides = (2,2).
  model.add(layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)))
  # Adding the Conv2D layer to the model with filters = 32, kernel_size = (3,3), strides = (1,1), padding='same', activation='relu' and a L2 Regularization of 0.0001.
  model.add(layers.Conv2D(filters = 32, kernel_size = (3,3), strides = (1,1), padding='same', activation='relu', kernel_regularizer = regularizers.l2(0.0001)))
  # Adding the Conv2D layer to the model with filters = 64, kernel_size = (3,3), strides = (1,1), padding='same', activation='relu' and a L2 Regularization of 0.0001.
  model.add(layers.Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding='same', activation='relu', kernel_regularizer = regularizers.l2(0.0001)))
  # Adding the Max Pooling layer with a pool size of (2,2), strides = (2,2).
  model.add(layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)))
  # Adding a flatten layer to the model.
  model.add(layers.Flatten())
  # Adding a dense layer to the model with units = 512, activation='relu' and L2 Regularization of 0.0001.
  model.add(layers.Dense(units = 512, activation='relu', kernel_regularizer = regularizers.l2(0.0001)))
  # Adding a dense layer to the model with units = 10, activation='linear' and L2 Regularization of 0.0001.
  model.add(layers.Dense(units = 10, activation='linear', kernel_regularizer = regularizers.l2(0.0001)))
  # Adding a softmax layer to the output layer.
  model.add(layers.Activation('softmax'))
  
  # Compiling the Neural Network model with adam optimizer, loss = losses.categorical_crossentropy and metrics as 'accuracy'.
  model.compile(optimizer = 'adam', loss = losses.categorical_crossentropy, metrics = ['accuracy'])

  # Training the model with a validation split of 0.2 and storing the model in the history object.
  history = model.fit(x = X_train, y = Y_train, epochs = epochs, batch_size = batch_size, validation_split = 0.2)

  # Making predictions using the X_test and applying np.argmax to return the maximum valued index among each column in matrix.
  y_pred = np.argmax(model.predict(X_test), axis=1)

  # Converting the Y_test to the similar shape so that it can be compared with the y_pred.
  y_actual = np.argmax(Y_test, axis=1)

  # Making the confusion matrix by calling the function and storing the values to the c_m.
  c_m = confusion_matrix(y_actual, y_pred)

  # Using the matplotllib.pyplot function to plot the confusion matrix.
  plt.matshow(c_m, cmap='coolwarm')

  # Adding a colorbar along with the confusion matrix.
  plt.colorbar()

  # finding the number of rows and columns of the confusion matrix.
  rows, columns = c_m.shape

  # Displaying values in the confusion matrix.
  for i in range(rows):
      for j in range(columns):
          plt.text(j, i, str(c_m[i][j]), ha='center', va='center')

  # Adding title, xlabel, ylabel and saving the fig to the 'confusion_matrix.png'.
  plt.title('Confusion Matrix')
  plt.xlabel('Predicted Labels')
  plt.ylabel('True labels')
  plt.savefig('confusion_matrix.png')

  # Saving the trained model into 'model.h5'
  model.save('model.h5')

  # Returning the Trained model, history, confusion matrix, and predictions.
  return [model, history, c_m, y_pred]