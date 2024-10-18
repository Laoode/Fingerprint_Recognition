<h1 align="center"><a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Chakra+Petch&weight=500&size=34&duration=1&pause=1000&color=EEEEEE&background=90EE90C7&vCenter=true&repeat=false&width=800&lines=Biometrics+Recognition++Reconstructing+Fingerprint" alt="Typing SVG" /></a></h1>

<img src="https://github.com/Laoode/Fingerprint_Recognition/blob/main/Images/Fingepint-animation-reduce.gif" alt="logo">
This repository contains a project for Biometrics Recognition Reconstructing Fingerprint Images Using Deep Learning (Convolutional Autoencoder). Systems for recognizing fingerprints have been used extensively to implement precise and trustworthy biometric identification between people. Pattern recognition in computer vision has greatly benefited from deep learning, particularly Convolutional Neural Networks (CNN). In this work, fingerprint images have been reconstructed using a convolutional neural network autoencoder. An autoencoder is a method that can reproduce information in pictures. The advantage of convolutional neural networks makes it suitable for feature extraction.

> [!NOTE]
> 
> This project utilizes deep learning techniques for fingerprint image reconstruction. The code has been executed using an NVIDIA L40S GPU with 48 gigabytes (GB) of memory capacity, which provides the necessary computational power for training the convolutional autoencoder model. Ensure your system has appropriate GPU capabilities to run this project efficiently.

```python
torch.cuda.get_device_name()
```

```python
'NVIDIA L40S'
```

# Table of Contents

1. [Data Exploration](#data-exploration)
2. [The Convolutional Autoencoder](#the-convolutional-autoencoder)
3. [Save Model](#save-model)
4. [Going Further](#going-further)
   
## Data Exploration
<div align="center">
  <img src="https://github.com/Laoode/Fingerprint_Recognition/blob/main/Images/Data-sets.png" alt="logo">
</div>

The datasets used in this project are from a collection of human fingerprint datasets suitable for research and evaluation of fingerprint recognition algorithms. While there are several datasets available from various sensors, this project utilizes the following:
- Public rectangular datasets
    - [FVC2000 DB1 B](http://bias.csr.unibo.it/fvc2000/default.asp)
    - [FVC2000 DB2 B](http://bias.csr.unibo.it/fvc2000/default.asp)
    - [FVC2000 DB3 B](http://bias.csr.unibo.it/fvc2000/default.asp)
    - [FVC2000 DB4 B](http://bias.csr.unibo.it/fvc2000/default.asp)
    - [FVC2002 DB1 B](http://bias.csr.unibo.it/fvc2002/)
    - [FVC2002 DB2 B](http://bias.csr.unibo.it/fvc2002/)
    - [FVC2002 DB3 B](http://bias.csr.unibo.it/fvc2002/)
    - [FVC2002 DB4 B](http://bias.csr.unibo.it/fvc2002/)
    - [FVC2004 DB1 B](http://bias.csr.unibo.it/fvc2004/)
    - [FVC2004 DB2 B](http://bias.csr.unibo.it/fvc2004/)
    - [FVC2004 DB3 B](http://bias.csr.unibo.it/fvc2004/)
    - [FVC2004 DB4 B](http://bias.csr.unibo.it/fvc2004/)
    - [Neurotechnology CrossMatch](https://www.neurotechnology.com/download.html#databases)
    - [Neurotechnology UareU](https://www.neurotechnology.com/download.html#databases)
    - [FVS dataset](https://fvs.sourceforge.net/about.html)

- Licensed rectangular datasets
    - [FVC2002 DB3 A](http://bias.csr.unibo.it/fvc2002/)

The data is split into three sets: training, validation, and testing. For training and validation, I use the FVC and Neurotechnology datasets. The test data, which is never seen during training, comes from the FVS dataset. This ensures a fair evaluation of the model's performance on unseen data.
If you want to access the complete collection of datasets used in this project, please visit my [**fingerprint-datasets**](https://github.com/Laoode/fingerprint-datasets) repository.

```python
# Shapes of training set
print("Dataset (images) shape: {shape}".format(shape=images_arr.shape))
```

````python
# Output
Dataset (images) shape: (2688, 224, 224)
````
As we can see, we have a total of 2688 images for training and validation, excluding the test set. The images in our dataset are grayscale with pixel values ranging from 0 to 255 and dimensions of 224 x 224. Before feeding the data into the model, we reshape each image into a matrix of size 224 x 224 x 1:

```python
images_arr = images_arr.reshape(-1, 224, 224, 1)
```

````python
# Output
New shape: (2688, 224, 224, 1)
````
We ensure the data type of the NumPy array is float32 and rescale the pixel values to the range 0-1 inclusive. Next, in data partitioning we split the data into 80% for training and 20% for validation.

```python
from sklearn.model_selection import train_test_split
train_X,valid_X,train_ground,valid_ground = train_test_split(images_arr,
                                                             images_arr,
                                                             test_size=0.2,
                                                             random_state=13)
```

> [!IMPORTANT]
> For this autoencoder task, we don't need separate labels. The training images serve as both input and ground truth, similar to the labels in a classification task.

## The Convolutional Autoencoder
<img src="https://github.com/Laoode/Fingerprint_Recognition/blob/main/Images/autoencoder.png" alt="logo">

An encoder and a decoder are the two components that make up the autoencoder, as you may already be aware.

- The encoder
    - The first layer will have 32 filters of size 3 x 3, which will be followed by a downsampling (max-pooling) layer.
    - The second layer will have 64 filters of size 3 x 3, and still another downsampling layer.
    - The final layer of encoder will have 128 filters of size 3 x 3.
- The decoder
    - The first layer will have 128 filters of size 3 x 3 followed by an upsampling layer.
    - The second layer will have 64 filters of size 3 x 3 followed by another upsampling layer.
    - The final layer of encoder will have one filter of size 3 x 3.
      
In this model, we use a batch size of 128. Depending on your system's capabilities, using a higher batch size of 256 or 512 may also be preferable. The batch size significantly contributes to determining the learning parameters and affects prediction accuracy and we train the model for 200 epochs.

```python
batch_size = 128
epochs = 200
inChannel = 1
x, y = 224, 224
input_img = Input(shape = (x, y, inChannel))
```

> [!NOTE]
> The number of layers, the number of filters, the filter size, and the number of epochs you train your model are all hyperparameters that should be chosen based on your own intuition. You can experiment with these values to see how your model performs. You will gradually master the art of deep learning in this way!

```python
def autoencoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)

    #decoder
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded
```
After creating the model, I compile it using the RMSprop optimizer. The loss parameter is set to 'mean_squared_error', as the loss after each batch will be calculated using mean squared error pixel by pixel between the batch of predicted output and the ground truth:

```python
autoencoder_model = Model(input_img, autoencoder(input_img))
autoencoder_model.compile(loss='mean_squared_error', optimizer=RMSprop())
```

Here's a summary of the model architecture:

```python
Model: "functional_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_1 (InputLayer)      │ (None, 224, 224, 1)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_6 (Conv2D)               │ (None, 224, 224, 32)   │           320 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 112, 112, 32)   │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_7 (Conv2D)               │ (None, 112, 112, 64)   │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_3 (MaxPooling2D)  │ (None, 56, 56, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_8 (Conv2D)               │ (None, 56, 56, 128)    │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_9 (Conv2D)               │ (None, 56, 56, 128)    │       147,584 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ up_sampling2d_2 (UpSampling2D)  │ (None, 112, 112, 128)  │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_10 (Conv2D)              │ (None, 112, 112, 64)   │        73,792 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ up_sampling2d_3 (UpSampling2D)  │ (None, 224, 224, 64)   │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_11 (Conv2D)              │ (None, 224, 224, 1)    │           577 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 314,625 (1.20 MB)
 Trainable params: 314,625 (1.20 MB)
 Non-trainable params: 0 (0.00 B)
```
Finally, it's time to use Keras' fit() function to train the model! Before that, I have trained the model using 200 epochs, and the validation loss and training loss are both in sync. This shows that my model is not overfitting: the validation loss is decreasing and not increasing. Then, I trained for another 200 epochs using the previously trained model, which means I tried to train for a total of 400 epochs.
```python
autoencoder_train = autoencoder_model.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))
```
Training output:
```python
Epoch 1/200
17/17 ━━━━━━━━━━━━━━━━━━━━ 4s 179ms/step - loss: 0.0080 - val_loss: 0.0070
Epoch 2/200
17/17 ━━━━━━━━━━━━━━━━━━━━ 2s 104ms/step - loss: 0.0069 - val_loss: 0.0062
Epoch 3/200
17/17 ━━━━━━━━━━━━━━━━━━━━ 2s 104ms/step - loss: 0.0065 - val_loss: 0.0063
...
Epoch 198/200
17/17 ━━━━━━━━━━━━━━━━━━━━ 2s 104ms/step - loss: 0.0053 - val_loss: 0.0050
Epoch 199/200
17/17 ━━━━━━━━━━━━━━━━━━━━ 2s 105ms/step - loss: 0.0052 - val_loss: 0.0052
Epoch 200/200
17/17 ━━━━━━━━━━━━━━━━━━━━ 2s 105ms/step - loss: 0.0053 - val_loss: 0.0052
```
Now, let's plot the loss plot between training and validation data to visualize the model performance after 400 epochs.
<div align="center">
  <img src="https://github.com/Laoode/Fingerprint_Recognition/blob/main/Images/plot-training-loss-&-validation-loss.png" alt="logo">
</div>

The performance of the model after training is quantified by its mean_squared_error metric:
```python
# Calculate loss on validation data
loss = autoencoder_model.evaluate(valid_X, valid_ground)  # This will return the loss

# If the model was compiled with loss='mean_squared_error', this value is the metric you're looking for.
print(f'Validation Loss: {loss:.4f}')
```

```python
17/17 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 0.0055
Validation Loss: 0.0052
```
> [!NOTE]
> Keep in mind that autoencoders are typically used for reconstruction tasks, and the more commonly used metric is loss (such as Mean Squared Error) rather than accuracy.
The autoencoder model demonstrates effective reconstruction with low loss (0.0055) and a slightly higher validation loss (0.0052), indicating good generalization without significant overfitting. The small difference between the two suggests the model is stable and performs well on unseen data, utilizing RMSprop and Mean Squared Error (MSE) for optimization and error measurement.

Now we validate the model using validation data for predicting on the model that you trained just now.

```python
decoded_imgs = autoencoder_model.predict(valid_X)  # Using validation for prediction
n = 10  # number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original image
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(valid_X[i].reshape(224, 224), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Reconstructed image
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(224, 224), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')
plt.show()
```
<div align="center">
  <img src="https://github.com/Laoode/Fingerprint_Recognition/blob/main/Images/validation-data.png" alt="logo">
</div>
From the above figures, we can see that the model successfully reconstructed the validation images.

So after that, we test the model using test data. For your information, the trained data uses image type .tiff and in test data we can use another sensor and different type which in test data uses image type .bmp.

But before using test data, make sure the image for testing has been preprocessed like the code above. After that:

```python
# Use the trained model to predict
predicted_images = autoencoder_model.predict(new_images_arr)
# Sample a random subset of indices
sample_indices = random.sample(range(len(new_images_arr)), 10)

plt.figure(figsize=(20, 4))
for i, index in enumerate(sample_indices):
    # Original Image
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(new_images_arr[i].reshape(224, 224), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Reconstructed Image
    ax = plt.subplot(2, 10, i + 1 + 10)
    plt.imshow(predicted_images[i].reshape(224, 224), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')
plt.show()
```

<div align="center">
  <img src="https://github.com/Laoode/Fingerprint_Recognition/blob/main/Images/test-data.png" alt="logo">
</div>

From the above figures, we can observe that our model did a great job in reconstructing the Secugen test images that can be predicted using the trained model.

as we all know, to evaluate how well the reconstruction result of the autoencoder model performs, in addition to MSE, I want to use Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize which parts of the input images contribute most to the model's output, even in the context of autoencoders. Although Grad-CAM is typically used for classification models, it can be adapted for use with autoencoders to understand the features that the model has learned and how they relate to the reconstruction process.
How Grad-CAM Works:

- Gradient Computation: Using the feature maps of the last convolutional layer, Grad-CAM calculates the gradients of the class scores. The significance of each feature map for class prediction is indicated by these gradients.
- Weighted Feature Maps: The feature maps are weighted using the gradients to highlight the regions that have a greater impact on the choice.
- Heatmap Generation: The areas of the image that the model concentrated on for its reconstruction are then highlighted in a heatmap.

```python
def generate_gradcam(model, img_array, layer_name, class_index):
    grad_model = tf.keras.models.Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf.expand_dims(img_array, axis=0))
        loss = predictions[0][class_index]  # Change to the appropriate class index
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)

    return heatmap.numpy()
```

```python
layer_name = 'conv2d_5'  # Specify the convolutional layer name
class_index = 0  # Placeholder value since autoencoders don't have classes
heatmap = generate_gradcam(autoencoder_model, new_images_arr[0], layer_name, class_index)

# Display the heatmap
plt.imshow(heatmap, cmap='jet')
plt.colorbar()
plt.show()
```

<div align="center">
  <img src="https://github.com/Laoode/Fingerprint_Recognition/blob/main/Images/Grad-CAM.png" alt="logo">
</div>

In the above Grad-CAM image, the red color indicates areas with high activation, while the blue color indicates areas with low activation. This indicates the parts of the image that contribute most to the model's decision to classify the fingerprint.

## Save Model
So last but not least, let's now save the trained model. It is an important step when you are working with Deep Learning. Since the weights are the heart of the solution to the problem you are tackling at hand!
You can anytime load the saved weights in the same model and train it from where your training stopped. For example, the above model if trained again, the parameters like weights, biases, the loss function, etc. will not start from the beginning and it will no longer be a fresh training.

```python
# Save the complete model with architecture and weights
autoencoder_model.save('autoencoder_complete_model.h5')
```

```python
# Load the complete model from the .h5 file
loaded_model = load_model('autoencoder_complete_model.h5')
```

## Going Further
So this project was a good start to understanding how to read images from scratch, analyze, preprocess and feed them into the model using a fingerprint dataset. It showed you one of the nice applications of autoencoders practically. If you were able to follow along easily or even with a little more effort, well done!
If this project is useful, star the repo and don't forget to follow me to look forward to the next projects.



<i>Don't forget to support me 🥺👉👈:</i>

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/H2H714OX25)

