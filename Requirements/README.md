## Requirements to run the applet

To run the script you will need:
- Python >= 3.9.5
- Tensorflow
- Keras
- NumPy
- Matplotlib
- Tkinter
- Pillow
- Scikit-image
- SciPy

**Note**: There are known problems of compatibility with Tensorflow and Keras. You need to make sure that your Python version is compatible with your Tensorflow and Keras versions. You can read more [here](https://keras.io/about) and [here](https://www.tensorflow.org/install).

I will include a `requirements.txt` file in this repository. You can use that to install all necessary `pip` packages

```
pip install -r requirements.txt
```

### Apple Silicon Mac OS Installation Fixes

If you are having trouble installing tensorflow and/or keras on a M1 mac, or any apple laptop, you can follow this tutorial.

**Disclaimer**: I am not an expert so take these suggestions with a grain of salt and at your own discretion.

#### Installation with Anaconda (Quick Fix)

The quickest method to solve the issues is to use Anaconda. The installation is pretty simple. You can follow a tutorial [here](https://www.anaconda.com/).

Make sure that you have XCode command tools installed:

```
xcode-select --install
```

Then, create a virtual environment

```
conda create --name <EnvironmentName> python=<PythonVersion>
```

**Remark**: Substitute `<EnvironmentName>` and `<PythonVersion>` with the corresponding values.

You can activate your Anaconda environment by simply running

```
conda activate <EnvironmentName>
```

After setting up and activating the virtual environment, you can install tensorflow using [Apple's guidelines](https://developer.apple.com/metal/tensorflow-plugin/):

```
conda install -c apple tensorflow-deps
python -m pip install tensorflow-macos
```

and if you also want the Tensorflow Metal plug-in (in order to run Tensorflow on GPU) 

```
python -m pip install tensorflow-metal
```

If everything worked out, you should be able to import tensorflow in your Python script

```python
import tensorflow as tf

print(tf.__version__)
```

#### Alternative Fixes

Similarly, you can fix the issue by using `pyenv` or by installing an appropriate version of Python on your machine.

To use `pyenv`, install it with [homebrew](https://brew.sh/)

```
brew install pyenv
```

Then, install your desired Python version

```
pyenv install <PythonVersion>
```

Finally, you can either use that version of Python in your shell, locally or globally

```
pyenv shell <PythonVersion>
pyenv local <PythonVersion>
pyenv global <PythonVersion>
```

In this case, you should just move to your project directory and set a local version of Python. Then, you can simply install tensorflow like in the Anaconda case using `pip`.

Finally, if you don't want to install neither Anaconda nor pyenv, you can just go to Python's official website and install the appropriate version of Python on your laptop. If you already have Python installed, you will need to delete it. Only one version can run on your local machine.

