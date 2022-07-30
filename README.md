# Vegetables Classification
## **Introduction**
<p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The Convolutional Neural Network (CNN or ConvNet) is a subtype of the Neural Networks that is mainly used for applications in image and speech recognition. Its built-in convolutional layer reduces the high dimensionality of images without losing its information. That is why CNNs are especially suited for this use case.
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Generally, this project aims to use Convolutional Neural Network algorithms to classify different types of vegetables.
</p>

## **Convolutional Neural Network**
![Convolutional Neural Network Layers](../master/assets/cnn_layers.png)
<p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Convolutional Neural Network is a deep learning algorithm that is generally used in image processing and takes images as input. This algorithm, which captures and classifies features in images with different operations, consists of different layers. The visual, which passes through these layers, which are Convolutional Layer, Pooling and Fully Connected, is subjected to different processes and reaches the consistency to enter the deep learning model.
</p>

### **Convolutional Layer**
<p>
&nbsp;&nbsp;&nbsp;Convolutional is the first layer that handles the image in CNN algorithms. As it is known, images are actually matrices consisting of pixels with certain values in them. In the convolution layer, a filter smaller than the original image size hovers over the image and tries to capture certain features from these images.
</p>

### **Stride**
<p>
&nbsp;&nbsp;&nbsp;The stride value is a value that can be changed as a parameter in Convolutional Neural Network models. This value determines how many pixels the filter will slide over the main image.
</p>

### **Padding**
<p>
&nbsp;&nbsp;&nbsp;When we apply the filter to an image, the output will be smaller than the original image due to dimensions. The method we can use to prevent this is padding. During the filling process, zeros are added to the image on all four sides as if it were a frame. Depending on the size of the filter, these zero-added layers can be increased.
</p>

### **ReLU**
<p>
&nbsp;&nbsp;&nbsp;ReLU (Rectified Linear Unit) is a nonlinear function that operates as f(x) = max(0,x). ReLU, whose main purpose is to get rid of negative values, has a very important position in CNNs.
</p>

### **Pooling**
<p>
&nbsp;&nbsp;&nbsp;Like the convolutional layer, the pooling layer is also intended to reduce dimensionality. In this way, both the required processing power is reduced and the unnecessary features that are caught are ignored and more important features are focused on.
</p>

### **Fully Connected Layer**
<p>
&nbsp;&nbsp;&nbsp;In the Fully Connected layer, our matrix image, which passes through the convolutional layer and the pooling layer several times, is turned into a flat vector.
</p>

## **Code Overview**
### **Implementation**
&nbsp;&nbsp;&nbsp;Firstly, I added the necessary libraries.

![Code Implementation](../master/assets/code_overview/implementation.png)

### **Assigning Directories**
&nbsp;&nbsp;&nbsp;I then assigned values to the files of the images, specified the file path, and showed how many classes it contains.

![Assigning Directories](../master/assets/code_overview/assigning_directories-1.png)

&nbsp;&nbsp;&nbsp;As you can see, it found 15,000 files belonging to 15 classes in the train file. It also found 3,000 files belonging to 15 classes in the validation file.

![Assigning Directories](../master/assets/code_overview/assigning_directories-2.png)

&nbsp;&nbsp;&nbsp;As you can see here, we assign the train dataset to class_names and show how many classes it has.

### **Building Model**
&nbsp;&nbsp;&nbsp;Now we are ready to build our model. The model type I use in this project is the sequential model type. Because sequential model is the easiest way to create a model in Keras. This allows us to build a model layer by layer.

![Building Model](../master/assets/code_overview/building_model-1.png)

&nbsp;&nbsp;&nbsp;In this model, we use the "add()" function when creating layers.

&nbsp;&nbsp;&nbsp;The first layer I will explain is the Conv2D layer.This type of layer is the convolution layers that will deal with our input images seen as 2D matrices. The number 32 in this layer is the number of nodes in the layer. This number can be set higher or lower depending on the size of the dataset. The kernel size is the size of the filter matrix for our convolution. So 3 core sizes means we will have a 3x3 filter matrix.

&nbsp;&nbsp;&nbsp;Activation is the activation function for the layer. The activation function we will be using for our first layer is the ReLU, or Rectified Linear Activation. This activation function has been proven to work well in neural networks.

&nbsp;&nbsp;&nbsp;The second layer I will explain is the MaxPooling2D layer. Downsamples the input along its spatial dimensions (height and width) by taking the maximum value over an input window (of size defined by pool_size) for each channel of the input. The window is shifted by strides along each dimension.

&nbsp;&nbsp;&nbsp;The third layer I will explain is BatchNormalization. Thanks to this layer, we enable simultaneous learning. It accelerates our education.

&nbsp;&nbsp;&nbsp;The fourth layer I will explain is the Dropout layer. This layer randomly sets the input units to 0 with a rate frequency at each step during the training period, which helps prevent overfitting. Inputs not set to 0 are magnified by 1/(1 - ratio) so that the sum of all inputs does not change.

&nbsp;&nbsp;&nbsp;The sixth layer I will explain is Flatten. This layer is used to flatten data in matrix form.

&nbsp;&nbsp;&nbsp;Our seventh layer is Dense. With this layer, we provide the transition of neurons or nodes between layers. It allows neurons from one layer to be connected to the next layer as input.

&nbsp;&nbsp;&nbsp;After these layer operations, we need to compile our model.

![Building Model](../master/assets/code_overview/building_model-2.png)

&nbsp;&nbsp;&nbsp;Compiling the model is done with the model.compile() function. This function takes three parameters: optimizer, loss, and metric.

&nbsp;&nbsp;&nbsp;Here we see the output of the summary of the model.

![Building Model](../master/assets/code_overview/building_model-3.png)

### **Training Model**
&nbsp;&nbsp;&nbsp;Now we will train our model. We do this with the model.fit() function. where x represents the training data. Epochs, on the other hand, indicates how many times the data set will be trained by going over the model. Validation_data already represents the validation set as you can tell from its name.

![Training Model](../master/assets/code_overview/training_model-1.png)

&nbsp;&nbsp;&nbsp;Here's a graph showing the validation loss of the model we're training:

![Training Model](../master/assets/code_overview/training_model-2.png)

&nbsp;&nbsp;&nbsp;Here is the graph showing the accuracy rate of the model:

![Training Model](../master/assets/code_overview/training_model-3.png)

&nbsp;&nbsp;&nbsp;Now let's test the model's prediction.

![Training Model](../master/assets/code_overview/training_model-4.png)

&nbsp;&nbsp;&nbsp;As you can see, he guessed our first picture correctly.

> **NOTE:**<br>Problem Prediction image has found from Google Images not from dataset which CNN learned.

## **Conclusion**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Convolutional neural networks are used in image and speech processing and are based on the structure of the human visual cortex. They consist of a convolution layer, a flattening layer. a pooling layer and a fully connected layer. Convolutional neural networks divide the image into smaller areas in order to view them separately for the first time. It is important to adjust the arrangement of the convolutional and max-pooling layers to each different use case.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In this paper, we studied to classify vegetables using Convolutional Neural Network. The program has successfully achieved its purpose. By training the program 20 times, we increased the correct prediction rate. We proved this by testing with a photograph of a vegetable that he had not seen before.