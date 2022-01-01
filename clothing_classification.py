import tensorflow as tf

import numpy as np 
import matplotlib.pyplot as plt 

# IMPORTING THE DATASET
fashion_mnist = tf.keras.datasets.fashion_mnist 
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# DATA EXPLORATION
# Checking the dimensions of the data and number of datapoints 
train_images.shape
len(train_labels)
train_labels
test_images.shape

# Make a list of Englihs labels in the order for each item that you can refer to with the numerical label in the dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# DATA PRE-PROCESSING
# 1) Visualizing what each image looks like 
plt.imshow(train_images[0])
plt.colorbar()
plt.show()
# 2) Feature scaling - normalizing the magnitude of the data to number between 0 and 1 
train_images = train_images / 250.0 
test_images = test_images / 250.0

plt.figure(figsize = (10, 10)) # Makes the whole figure on which all 25 images (subplots) are plotted scaled to 10 x 10 inches
for i in range(25): 
  plt.subplot(5, 5, i+1) # i+1 NOT i because the figure number has to count up from ONE, not zero
  plt.xticks([]) # the empty brackets [] makes sure that there aren't ticks on the x axis 
  plt.yticks([])
  plt.imshow(train_images[i], cmap = plt.cm.binary) # cmap = plt.cm.binary sets the color of the map (cmap) to black and white coloring
  plt.xlabel(class_names[train_labels[i]])
plt.show()

# CREATING THE ML MODEL
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(10) # What if you put reLU here instead of later converting to a probability model?
])

# Configure the model into the appropriate units/metrics of measuring its performance 
model.compile(optimizer = 'adam', 
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # "from_logits = True" notifies the computer that the original data is NOT a probability model and instead has the raw logit data
              metrics = 'accuracy'
)

# TRAINING THE MODEL

model.fit(train_images, train_labels, epochs = 20)
# Testing performance on the test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2) # a little bit lower accuracy than in training data - overfitting

# TESTING THE TRAINED MODEL

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# Make the model generate predictions for the test dataset
predictions = probability_model.predict(test_images)
# Looking at probability distribution for the first datapoint
predictions[0]
# The prediction from the model (choice with highest probability)
np.argmax(predictions[0])
# Checking answer for first datapoint 
test_labels[0]

# VISUALLY ASSESSING MODEL PERFORMANCE 

# Create functions for displaying the model's prediction, its probability distribution, and the real answer 
def plot_image(i, predictions_array, true_label, images): 
# i: the index for the datapoint being predicted on
# predictions_array: the probability distribution of the model on one given test datapoint (defined by i)
# true_label: the test_labels array (answer keys)
#images: test_images array 

  plt.xticks([])  
  plt.yticks([])
  plt.grid(False)
# Defining prediction of the model
  predicted_label = np.argmax(predictions_array) # the predictions_array would have a certain index [i] when the function is actually called  
# Real answer and image
  true_label, image = true_label[i], images[i]

  plt.imshow(image, cmap = plt.cm.binary) 
  if predicted_label == true_label: # Color code to visually tell if the model got it right/wrong
    color = 'blue'
  else: 
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 
                                       100*np.max(predictions_array), 
                                       class_names[true_label]), color = color) # max instead of maximum, because maximum outputs the max values, one for each multiple arrayS


def plot_value_array(i, predictions_array, true_label): 
# i, predictions_array, true_label - same definitions as before

  true_label = true_label[i]
  predicted_label = np.argmax(predictions_array)

  plt.grid(False)
  plt.xticks(range(10)) # Each of the 9 ticks (1-9) represents each possible clothing item 
  plt.ylim([0, 1])

  thisplot = plt.bar(range(10), predictions_array, color = 'red')
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red') # change the color of the bar when x = predicted label to red
  thisplot[true_label].set_color('blue') # change the color bar when x = right answer to blue (would cover up the red when right answer is predicted)


# Test-displaying the image and probability distribution for the first datapoint
i = 0 

plt.figure(figsize = (6,3))
plt.subplot(1,2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show() 

# Testing on 13th datapoint 
i = 12

plt.figure(figsize = (6,3))
plt.subplot(1,2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

# Creating an array of prediction graphs for a certain number of datasets (starting from the first one)
num_rows = 5
num_cols = 3
num_figs = num_rows * num_cols 
plt.figure(figsize = (2*2*num_cols, 2*num_rows))

for i in range(num_figs): # 2*num_cols (below) because there are twice as many colmuns
  plt.subplot(num_rows, 2*num_cols, 2*i+1) # 2*i + 1 because 2*i ensures that there isn't an overlap between figure numbers (there's an overlap if you use i + 1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()


# USING THE MODEL 

img = test_images[i]
# Adding the new, single image to an empty array (beacuse keras works best with arrays, not single datapoints)
img = (np.expand_dims(img, axis = 0)) # axis = 0 means the expansion of the list would happen column-wise (adding another column)
print(img.shape)
# Making predictions on the test image
prediction_single = probability_model.predict(img)
# Display the prediction it made (in English)
print(class_names[test_labels[1]])
plot_value_array(1, predictions[i], test_labels)

