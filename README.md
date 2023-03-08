# Convolutional Neural Network (CNN) – Deep Learning

## Showcase

The model is trained to recognise different stroke widths and handwriting styles. If you run `main.py` you can draw your own digits

<div align="center">
<video src="https://user-images.githubusercontent.com/79821802/222105686-db9a13fa-97ab-4ed5-a288-3f825c66b8cd.mov"></video>
</div>
<br>

If you run `Webcam.py`, you can use your camera or webcam to recognise digits

<br><div align="center">
<video src="https://user-images.githubusercontent.com/79821802/223602197-855ab21e-ac90-4e88-a4f1-38851d82b43d.MOV"></video>
</div><br>

**Remark**: Digits must be on a black background written with a white stroke.

<br>
The model accuracy is 99.47% on the MNIST test dataset (i.e. 9,947/10,000). You can see that it is also pretty accurate on unseen data outside of the MNIST dataset as well.

### How to Use 
The instructions are shown on the applet as well. You can adjust the brush size by clicking the + or - buttons. You are able to test your own neural network model by indicating the name of the .h5 file (or the path to that file). Once you have written the name of the model file that you wish to use, press enter/return to confirm your selection. Now you can start writing on the black canvas. Press `command (⌘) + R` to clear the canvas. You can also save the digit you wrote as an image in case you want to use it somewhere else.

To start the applet, just make sure you have all the [required modules](https://github.com/williamchenjun/MNIST-Handwritten-Digits-CNN/blob/main/Requirements/README.md) installed, and run the `main.py` script

```
python3 main.py
```

**Remark**: Make sure that you are running the command from the directory containing the `main.py` script.

**Update**: You can now adjust the canvas pixel size in the `main.py` script. Just set `self.__PixelSize` in the constructor to any positive integer. However, note that if you decrease the size, it will become quite laggy.

## MNIST Dataset

You can find the dataset [here](https://yann.lecun.com/exdb/mnist/). The website describes how the images have been preprocessed as well. You will find a python script in this repository called `utils.py` which will contain some useful image preprocessing tools e.g. to perform data augmentation.

If you don't want to manually download the dataset, you can just use `keras.datasets.mnist` from the Keras module.

----

You can also use `MNIST_CNN` class from my `CNN.py` script to quickly work with the convolutional network I trained.

### Make a prediction using the MNIST test dataset
**Script**:
```python
network = MNIST_CNN()
network.load(<h5ModelFilePath>)

network.predict(use_mnist = True, samples = 10, plot = True)
```
**Output**:

It will output a list of 10 predictions of 10 images chosen at random compared to their true label
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
and a plot will be saved in the `Plots` folder:

<div align="center">
<img src="https://user-images.githubusercontent.com/79821802/221983510-2fe57c55-f6a3-451a-9e8f-27b25a590791.png" width=800/>
</div>

**Note**: The original data actually has a black background and white handwriting.

**Remark**: The folder that contains the `CNN.py` file should have a `Plots` folder.

If you don't want an image generated, you can get a ranking instead:
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
which shows you the confidence rank for each image prediction. That is, how confident the network is that the image is a certain number.

**Note**: The method takes random samples from the dataset. Thus, if you want the same outputs every time you run the script, you need to fix a seed using `random.seed` or `np.random.seed`.

### Make a prediction using your own image
You can import your image and use it to get a prediction:
```python
network.predict(use_mnist = False, filename = "yourimage.png", file_true_label = <TrueLabel>)
```
where `<TrueLabel>` is the true label of the digit you're trying to predict. 

There are also two additional arguments that you can specify: `invert_color` and `center`.
```python
network.predict(use_mnist = False, filename = "zero.png", file_true_label = 0, invert_color = True, center = True)
```
This model is trained on images with a black background and white digits. It won't be able to recognise your image if the color scheme is the other way around. You can use the `invert_color` argument to invert your image's color. `center` will center your digit at the center of the image like the MNIST data (you will be able to find the code used to do this in my `utils.py` script). It might help with getting a more accurate prediction in case the confidence is quite low.

### Make a prediction using your own dataset
You can load your own dataset and use it to get predictions. Your folder needs to be structured as follows:
```
Project Folder
│   CNN.py
│
└───Plots
│
└───YourDataset
     │
     └───0
     │   │   yourdata.png
     │   │   ...
     │
     └───1
     │   │   moredata.png
     │   │   ...
     │
     └───2
     │
     └───3
     │
     └───4
     │
     └───5
     │
     └───6
     │
     └───7
     │
     └───8
     │
     └───9
```
Thus, in your main working directory, you should have a dataset folder inside of which there are 10 folders named after each digit. Then, in each digit folder you should put your images for that digit.
```python
x_test, y_test = network.unseen_sample(pathname, categories = 10, limit = 10, samples = 10, center = True)
network.predict_unseen(x_test, y_test, plot = True)
```
The `network.unseen_sample` method imports your dataset and prepares it for predictions. You can choose to limit how many images to include for each digit. Again, you have the option to `invert_color` or `center` the images of your dataset. Afterwards, you can either use `network.predict_unseen` (made specifically for this) or `network.predict`. Note that if you use `network.predict`, you will have to generate a prediction one image at a time. There isn't support for arrays of images for that method yet. Whereas, `network.predict_unseen` can predict all of your data at once.

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
You can easily plot the accuracy and loss graphs for the model:

```python
network.plot_loss_accuracy()
```

<div align="center">
<img src="https://user-images.githubusercontent.com/79821802/221990103-70f1e65d-f002-49c2-a81c-338f9f63bc60.png" width=400/>
</div>

### Train and save your model

If you want to change the model structure and/or train the model again, you can do that quite simply

```python
network.train()
network.save(<h5ModelDestinationPath>)
```
The model's history will be saved automatically as well. You can change the optimizer and loss functions in the train method by specifying `optimizer` and `loss`. You can either pass the name as a string or use functions from `keras.optimizers` and `keras.losses`. Lastly, `network.train` has a `verbose` argument. You can set it to `0` to not get any progress information in your shell/terminal.
