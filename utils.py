def Debug(*obj, **kwargs):
    print(f"Debug: {obj}")
    if len(list(kwargs.keys())):
        for key, val in kwargs.items():
            print(f"{key} : {val}\n")

def add_noise(input, intensity : float = 0.01, spread : float = 0.1):
    try: import numpy as np
    except: raise Exception('Failed to import numpy or the module is missing.')

    assert intensity <= 1 and spread <= 1

    copy_input = np.copy(input)
    shape = copy_input.shape
    copy_input = copy_input.ravel()
    for _ in range(int(spread*copy_input.size)):
        copy_input[np.random.randint(0, copy_input.size)] = np.random.randint(0, 255)*intensity/255
    return copy_input.reshape(shape).astype('float32')

def gauss_noise(image, intensity: float = 0.1, *, max_var: int = 30):
    try: import numpy as np
    except: raise Exception('Failed to import numpy or the module is missing.')

    assert 0.0 <= intensity <= 1.0 and 0 <= max_var <= 255

    if len(image.shape) == 3: row,col,ch= image.shape
    else: row, col = image.shape

    mean = 0
    var = np.random.randint(0, max_var)
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,((row,col,ch) if len(image.shape) == 3 else (row,col)))
    gauss = gauss.reshape(*((row,col,ch) if len(image.shape) == 3 else (row,col)))
    noisy = image + gauss*intensity
    return noisy

def salt_and_pepper_noise(image, intensity : float = 0.1, sparseness : float = 0.1):
    try: import numpy as np
    except: raise Exception("Failed to import numpy or module is missing.")
    for _ in range(int(sparseness*image.shape[0]*image.shape[1])):
        image[int(np.random.randint(0,len(image))), int(np.random.randint(0,len(image[0])))] += np.random.uniform(-255.0,255.0)*intensity
    return image

def rand_zoom(image, variance : int = 3):
    try: 
        import numpy as np
        from skimage.transform import resize
    except: raise Exception('Failed to import numpy or skimage, or module(s) is missing.')

    assert 0 < variance <= 5
    assert type(image) == np.ndarray and len(image) > 0

    image_shape = image.shape
    x = np.random.randint(0, variance)
    y = len(image) - x
    if variance == 0:
        return image
    return resize(image[x:y, x:y], image_shape)

def augment(images, labels, noise_intensity : float = 0.2, max_rotation : int = 10, max_shift : int = 10, max_zoom : int = 3, outputs : int = 5, *, fill : int = 0, invert_colors : bool = False):
    """
    Generate augmented data.

    Parameters
    ----------
    - `images` : `NDArray`
    - `labels` : `NDArray | list`
    - `noise_intensity` : `float` (Default `0.2`)
    - `max_rotation` : `int` (Default `10`)
    - `max_shift` : `int` (Default `10`)
    - `max_zoom` : `int` (Default 3)
    - `outputs` : `int` (Default 5)
        - This parameter indicates the number of augmented outputs per image.

    Returns
    -------
    `tuple[NDArray, NDArray] | tuple[NDArray, list]`
        Returns a tuple containing the augmented images and the associated labels.
    """
    try: 
        from scipy.ndimage import rotate
        import numpy as np
    except: raise Exception('Failed to import scipy or numpy, or the module(s) is missing.')

    assert -360 <= max_rotation <= 360 and -14 <= max_shift <= 14 and 0 <= max_zoom <= 5 and 0 <= noise_intensity <= 1, "Out of bounds."

    results = []

    # A single image's shape is (28, 28). If it comes in an array of length N, it will be (N, 28, 28)
    if len(images.shape) < 3:
        for _ in range(outputs):
            x, y = np.random.randint(-max_shift, max_shift, size = (1,2)).flatten()
            shifted = shift(images, int(x), int(y), fill_value=255-fill)
            zoomed = rand_zoom(shifted, np.random.randint(1, max_zoom))
            rotated = rotate(zoomed, np.random.randint(-max_rotation, max_rotation), reshape = False, cval=255-fill)
            noisy = salt_and_pepper_noise(rotated, noise_intensity, sparseness = 0.2)
            if invert_colors: noisy = invert(noisy)
            results.append(noisy)
        
        return np.array(results), labels

    else:
        assert len(images) == len(labels), "Must have the same amount of images and labels."
        label_list = []
        for image, label in zip(images, labels):
            temp = []
            for _ in range(outputs):
                x, y = np.random.randint(-max_shift, max_shift, size = (1,2)).flatten()
                shifted = shift(image, int(x), int(y), fill_value=255-fill)
                zoomed = rand_zoom(shifted, np.random.randint(1, max_zoom))
                rotated = rotate(zoomed, np.random.randint(-max_rotation, max_rotation), reshape = False, cval=255-fill)
                noisy = salt_and_pepper_noise(rotated, noise_intensity, sparseness = 0.2)
                if invert_colors: noisy = invert(noisy)
                temp.append(noisy)
                label_list.append(label)
            results.extend(temp)
        
        assert len(results) == len(label_list)

        return np.array(results), np.array(label_list)
    
def add_white_bg(filename : str, new_filename : str, *, output_path: str = None):
    try: 
        from subprocess import run
        import os
    except: raise Exception('Failed to import subprocess, or module(s) is missing.')
    if output_path is not None: 
        run("convert {} -background white -alpha remove -alpha off {}/{}".format(filename, output_path, new_filename).split(" "))
    else:
        run("convert {} -background white -alpha remove -alpha off {}".format(filename, new_filename).split(" "))

def shift(image, x : int = 0, y : int = 0, *, output_shape : tuple = (28, 28), fill_value : float = 255.0, dtype : str = "float32"):
    """
    Parameters
    ----------

    - `image` : `NDArray`
    - `x` : `int`
        - Positive `x` shifts to the right.
        - Negative `x` shifts to the left.
    - `y` : `int`
        - Positive `y` shifts to the bottom.
        - Negative `y` shifts to the top.
    - `output_shape` : `tuple[int, int]`
    - `fill_value` : `float`
    - `dtype` : `str`

    Returns
    -------
    Shifted image as a `NDArray`.
    """

    try: import numpy as np
    except: raise Exception("Failed to import numpy or the module is missing.")

    assert len(image.shape) >= 2 and type(x) == int and type(y) == int and type(output_shape) == tuple

    image_shape = image.shape[:2]
    res = image
    col = np.array([[fill_value]]*image_shape[0])
    row = np.array([fill_value]*image_shape[1])

    if x < 0:
        for _ in range(np.abs(x)):
            res = np.hstack((col, res))
            res = res[:, :-1]
    elif x > 0:
        for _ in range(x):
            res = np.hstack((res, col))
            res = res[:, 1:]
    
    if y < 0:
        for _ in range(np.abs(y)):
            res = np.vstack((row, res))
            res = res[:-1, :]
    elif y > 0:
        for _ in range(y):
            res = np.vstack((res, row))
            res = res[1:, :]
    
    return res.reshape(output_shape).astype(dtype)

def find_bounding_box(image):
    try: import numpy as np
    except: raise Exception("Failed to import numpy or the module is missing.")

    white = 255
    max_row_index = 0
    max_col_index = 0
    min_row_index = len(image)
    min_col_index = len(image[0])
    
    transposed_image = image.T

    for row in image:
        (non_white, ) = np.where(row != white)
        if len(non_white) > 0:
            if min(min(non_white), min_row_index) == min(non_white):
                min_row_index = min(non_white)
            if max(max(non_white), max_row_index) == max(non_white):
                max_row_index = max(non_white)
        continue

    for row in transposed_image:
        (non_white, ) = np.where(row != white)
        if len(non_white) > 0:
            if min(min(non_white), min_col_index) == min(non_white):
                min_col_index = min(non_white)
            if max(max(non_white), max_col_index) == max(non_white):
                max_col_index = max(non_white)
        continue

    return min_row_index, min_col_index, max_row_index, max_col_index, 

def center_image(image, approx = "ceil"):
    """
    Center a black and white image to the overall image center point.

    Parameters
    ----------

    - `image` : `NDArray`
    - `approx` : `str`
        - Approximation type: Either `ceil` or `floor`.
    
    Returns
    -------

    The centered image as an `NDArray`.
    """
    try: import numpy as np
    except: raise Exception("Failed to import numpy or the module is missing.")

    assert type(image) == np.ndarray and approx.lower() in ("ceil", "floor")

    image_width = len(image[0])
    image_height = len(image)
    approx = np.ceil if approx.lower() == "ceil" else np.floor
    xmin, ymin, xmax, ymax = find_bounding_box(image.reshape(28, 28))
    dx, dy = ((xmax + xmin - image_width)/2), ((ymax + ymin - image_height)/2)
    shifted_img = shift(image.reshape(28, 28), int(approx(dx)), int(approx(dy)))

    return shifted_img

def load_image(filename : str, output_shape : tuple = (28, 28)):
    try: from keras.utils import load_img, img_to_array
    except: raise Exception('Failed to import keras or module is missing.')

    img = load_img(filename, color_mode="grayscale", target_size=(28, 28))
    img = img_to_array(img)
    img = img.reshape(output_shape).astype('float32')
    return img

def invert(image):
    try: import numpy as np
    except: raise Exception("Failed to import numpy or module is missing.")

    temp = np.full(image.shape, 255)
    inverted = temp - image

    return inverted.astype("float32")