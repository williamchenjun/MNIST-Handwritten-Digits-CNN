# Convolutional Neural Network (CNN) – Deep Learning

## Showcase

The model is trained to recognise different stroke widths and handwriting styles.

<div align="center">
<video src="https://user-images.githubusercontent.com/79821802/221979894-b297bc4b-6763-421c-9141-33fd76d44a08.mov"></video>
</div>
<br>
The model accuracy is 99.47% on the MNIST test dataset. You can see that it is also pretty accurate on unseen data as well.

You can adjust the brush stroke width [**Bug**] either through the input field (and pressing the enter/return key on your keyboard to confirm) or you can use the two buttons to increment the width by 1. Moreover, you can import your own trained neural network models (only supports `h5` format). You can change models at any time you want.

## MNIST Dataset

You can find the dataset [here](https://yann.lecun.com/exdb/mnist/). The website describes how the images have been preprocessed as well.

If you don't want to manually download the dataset, you can just use `keras.datasets.mnist`.

----

You can also use `MNIST_CNN` class to quickly work with the convolutional network:

### Make a prediction using the MNIST test dataset
**Script**:
```python
network = MNIST_CNN()
network.load(<h5ModelFilePath>)

network.predict(use_mnist = True, samples = 10, plot = True)
```
**Output**:

You will see a list of the 10 predictions compared to their true label
```
1/1 [==============================] - 0s 326ms/step

Prediction: 1 – True Label: 1
Prediction: 3 – True Label: 3
Prediction: 4 – True Label: 4
Prediction: 0 – True Label: 0
Prediction: 6 – True Label: 6
Prediction: 6 – True Label: 6
Prediction: 2 – True Label: 2
Prediction: 1 – True Label: 1
Prediction: 7 – True Label: 7
Prediction: 1 – True Label: 1
```
and a plot will be save in the `Plots` folder

<div align="center">
<img src="https://user-images.githubusercontent.com/79821802/221983510-2fe57c55-f6a3-451a-9e8f-27b25a590791.png" width=800/>
</div>

**Note**: The original data actually has a black background and white handwriting.

If instead we wrote

```python
network.predict(use_mnist = True, samples = 10, rank = True)
```
we get the following output

```
1/1 [==============================] - 0s 153ms/step

Prediction: 8 – True Label: 8
Guess: 8 – Confidence: 100.00%
Guess: 9 – Confidence: 0.00%
Guess: 4 – Confidence: 0.00%
Guess: 2 – Confidence: 0.00%
Guess: 5 – Confidence: 0.00%
Guess: 3 – Confidence: 0.00%
Guess: 6 – Confidence: 0.00%
Guess: 7 – Confidence: 0.00%
Guess: 1 – Confidence: 0.00%
Guess: 0 – Confidence: 0.00%

Prediction: 5 – True Label: 5
Guess: 5 – Confidence: 100.00%
Guess: 3 – Confidence: 0.00%
Guess: 6 – Confidence: 0.00%
Guess: 9 – Confidence: 0.00%
Guess: 0 – Confidence: 0.00%
Guess: 8 – Confidence: 0.00%
Guess: 4 – Confidence: 0.00%
Guess: 1 – Confidence: 0.00%
Guess: 7 – Confidence: 0.00%
Guess: 2 – Confidence: 0.00%

Prediction: 8 – True Label: 8
Guess: 8 – Confidence: 100.00%
Guess: 4 – Confidence: 0.00%
Guess: 9 – Confidence: 0.00%
Guess: 2 – Confidence: 0.00%
Guess: 6 – Confidence: 0.00%
Guess: 3 – Confidence: 0.00%
Guess: 1 – Confidence: 0.00%
Guess: 5 – Confidence: 0.00%
Guess: 7 – Confidence: 0.00%
Guess: 0 – Confidence: 0.00%

Prediction: 2 – True Label: 2
Guess: 2 – Confidence: 100.00%
Guess: 7 – Confidence: 0.00%
Guess: 1 – Confidence: 0.00%
Guess: 0 – Confidence: 0.00%
Guess: 3 – Confidence: 0.00%
Guess: 8 – Confidence: 0.00%
Guess: 6 – Confidence: 0.00%
Guess: 5 – Confidence: 0.00%
Guess: 9 – Confidence: 0.00%
Guess: 4 – Confidence: 0.00%

Prediction: 4 – True Label: 4
Guess: 4 – Confidence: 100.00%
Guess: 9 – Confidence: 0.00%
Guess: 6 – Confidence: 0.00%
Guess: 2 – Confidence: 0.00%
Guess: 8 – Confidence: 0.00%
Guess: 7 – Confidence: 0.00%
Guess: 1 – Confidence: 0.00%
Guess: 0 – Confidence: 0.00%
Guess: 5 – Confidence: 0.00%
Guess: 3 – Confidence: 0.00%

Prediction: 2 – True Label: 2
Guess: 2 – Confidence: 100.00%
Guess: 7 – Confidence: 0.00%
Guess: 1 – Confidence: 0.00%
Guess: 3 – Confidence: 0.00%
Guess: 0 – Confidence: 0.00%
Guess: 6 – Confidence: 0.00%
Guess: 4 – Confidence: 0.00%
Guess: 5 – Confidence: 0.00%
Guess: 8 – Confidence: 0.00%
Guess: 9 – Confidence: 0.00%

Prediction: 9 – True Label: 9
Guess: 9 – Confidence: 100.00%
Guess: 4 – Confidence: 0.00%
Guess: 7 – Confidence: 0.00%
Guess: 0 – Confidence: 0.00%
Guess: 2 – Confidence: 0.00%
Guess: 3 – Confidence: 0.00%
Guess: 1 – Confidence: 0.00%
Guess: 5 – Confidence: 0.00%
Guess: 8 – Confidence: 0.00%
Guess: 6 – Confidence: 0.00%

Prediction: 5 – True Label: 5
Guess: 5 – Confidence: 100.00%
Guess: 3 – Confidence: 0.00%
Guess: 9 – Confidence: 0.00%
Guess: 7 – Confidence: 0.00%
Guess: 6 – Confidence: 0.00%
Guess: 4 – Confidence: 0.00%
Guess: 8 – Confidence: 0.00%
Guess: 0 – Confidence: 0.00%
Guess: 2 – Confidence: 0.00%
Guess: 1 – Confidence: 0.00%

Prediction: 9 – True Label: 9
Guess: 9 – Confidence: 100.00%
Guess: 2 – Confidence: 0.00%
Guess: 7 – Confidence: 0.00%
Guess: 8 – Confidence: 0.00%
Guess: 4 – Confidence: 0.00%
Guess: 3 – Confidence: 0.00%
Guess: 1 – Confidence: 0.00%
Guess: 0 – Confidence: 0.00%
Guess: 5 – Confidence: 0.00%
Guess: 6 – Confidence: 0.00%

Prediction: 3 – True Label: 3
Guess: 3 – Confidence: 100.00%
Guess: 5 – Confidence: 0.00%
Guess: 2 – Confidence: 0.00%
Guess: 7 – Confidence: 0.00%
Guess: 1 – Confidence: 0.00%
Guess: 8 – Confidence: 0.00%
Guess: 0 – Confidence: 0.00%
Guess: 9 – Confidence: 0.00%
Guess: 6 – Confidence: 0.00%
Guess: 4 – Confidence: 0.00%
```
which show you the prediction confidence rank.

**Note**: The method takes random samples from the dataset. Thus, if you want the same outputs every time you run the script, you need to fix a seed.

### Make a prediction using your own image

```python
network.predict(use_mnist = False, filename = "yourimage.png", file_true_label = <TrueLabel>)
```
where `<TrueLabel>` is the true label of the digit you're trying to predict. 

There also, two additional arguments that you can specify

```python
network.predict(use_mnist = False, filename = "zero.png", file_true_label = 0, invert_color = True, center = True)
```
This model is trained on images with black background and white digits. It won't be able to recognise your image if it's the other way around. You can use the `invert_color` argument to invert your image's color. `center` will center your digit at the center of the image like the MNIST data. It might help with getting a more accurate prediction in case the confidence is quite low.

### Make a prediction using your own dataset

You can load you own dataset and use it to get predictions. Your folder needs to be structured as follows:
```
+ Project Folder
  CNN.py
+ Plots
+ YourDataset
|– 0
   |– zero.png
   |– etc...
|– 1
|– 2
|– 3
|– 4
|– 5
|– 6
|– 7
|– 8
|– 9
```

Thus, in your main working directory, you should have a dataset folder inside of which there are 10 folders named after each digit. Then in each digit folder you should put your images for that digit.

```python
x_test, y_test = network.unseen_sample(pathname, categories = 10, limit = 10, samples = 10, center = True)
network.predict_unseen(x_test, y_test, plot = True)
```
The `network.unseen_sample` imports your dataset and prepares it for you. You can choose to limit how many images to include for each digit. Again, you have the option to `invert_color` or `center` you images. Afterwards, you can either use `network.predict_unseen` (made specifically for this) or `network.predict`. Note that if you use `network.predict`, you will have to generate a single plot for each digit. There isn't support for arrays of images for that method yet. Whereas, `network.predict_unseen` can predict all of your data at once.

```
1/1 [==============================] - 0s 155ms/step


Prediction: 1 – True Label: 1
Guess: 1 – Confidence: 100.00%
Guess: 7 – Confidence: 0.00%
Guess: 2 – Confidence: 0.00%
Guess: 3 – Confidence: 0.00%
Guess: 8 – Confidence: 0.00%
Guess: 4 – Confidence: 0.00%
Guess: 0 – Confidence: 0.00%
Guess: 5 – Confidence: 0.00%
Guess: 9 – Confidence: 0.00%
Guess: 6 – Confidence: 0.00%


Prediction: 1 – True Label: 1
Guess: 1 – Confidence: 100.00%
Guess: 6 – Confidence: 0.00%
Guess: 2 – Confidence: 0.00%
Guess: 7 – Confidence: 0.00%
Guess: 0 – Confidence: 0.00%
Guess: 3 – Confidence: 0.00%
Guess: 5 – Confidence: 0.00%
Guess: 9 – Confidence: 0.00%
Guess: 4 – Confidence: 0.00%
Guess: 8 – Confidence: 0.00%


Prediction: 9 – True Label: 6
Guess: 9 – Confidence: 44.71%
Guess: 6 – Confidence: 24.52%
Guess: 5 – Confidence: 23.59%
Guess: 0 – Confidence: 3.71%
Guess: 8 – Confidence: 3.04%
Guess: 3 – Confidence: 0.23%
Guess: 4 – Confidence: 0.12%
Guess: 1 – Confidence: 0.05%
Guess: 2 – Confidence: 0.03%
Guess: 7 – Confidence: 0.00%


Prediction: 9 – True Label: 9
Guess: 9 – Confidence: 100.00%
Guess: 4 – Confidence: 0.00%
Guess: 0 – Confidence: 0.00%
Guess: 7 – Confidence: 0.00%
Guess: 8 – Confidence: 0.00%
Guess: 2 – Confidence: 0.00%
Guess: 1 – Confidence: 0.00%
Guess: 3 – Confidence: 0.00%
Guess: 5 – Confidence: 0.00%
Guess: 6 – Confidence: 0.00%


Prediction: 1 – True Label: 1
Guess: 1 – Confidence: 100.00%
Guess: 7 – Confidence: 0.00%
Guess: 2 – Confidence: 0.00%
Guess: 3 – Confidence: 0.00%
Guess: 4 – Confidence: 0.00%
Guess: 0 – Confidence: 0.00%
Guess: 8 – Confidence: 0.00%
Guess: 5 – Confidence: 0.00%
Guess: 6 – Confidence: 0.00%
Guess: 9 – Confidence: 0.00%


Guessed Correctly: 4/5
```

### Plot the loss and accuracy graphs for the model

<div align="center">
<img src="https://user-images.githubusercontent.com/79821802/221990103-70f1e65d-f002-49c2-a81c-338f9f63bc60.png" width=400/>
</div>

You can easily plot the 
