from keras.models import load_model
import cv2
import tkinter as tk
from PIL import Image, ImageTk, ImageOps
import numpy as np

wx, wy = 400, 400

def Webcam(delay : int = 1, unit : str = "ms"):
    """
    Parameters
    ----------
    `delay` : int
        Delay between each frame.
    `unit` : `str`
        Unit of measure for the delay. Default is `ms`. Conversion from `ms` to `s` is done by multiplicating 10e-3.

    Bugs
    ----
    - In the presence of reflections, the model might not be able to predict your digit.
        - Quick Fix: Use paper or a non-reflective surface. Note that the background needs to be black and the stroke needs to be white.

    Example
    -------
    >> Webcam(10) # Refresh frame every 10 milliseconds.
    >> Webcam(10, 's') # Refresh frame every 10 seconds.
    """
    ret, frame = cam.read()

    assert delay >= 0 and unit in ("s", "ms")

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(gray).resize((28, 28))
        predict = model.predict(np.asarray(img).reshape(1, 28, 28) / 255.0, verbose = 0)
        img = ImageOps.contain(img, (wx, wy))
        photo = ImageTk.PhotoImage(image=img)
        webcam_image.photo = photo
        webcam_image.configure(image = photo)
        prediction = np.argmax(predict)
        conf = np.max(predict)
        predicted.text = prediction
        predicted.configure(text = prediction)
        confidence.text = confidence
        confidence.configure(text = f"{conf * 100.0 : .2f}%")

    total_delay = delay*1000 if unit == "s" else delay
    root.after(total_delay, Webcam)

root = tk.Tk()
cam = cv2.VideoCapture(0)
model = load_model("network_for_mnist.h5")

root.geometry(f"{wx}x{wy + 300}")
root.title("Digits Recognition")
root.eval("tk::PlaceWindow . center")

title = tk.Label(root, text = "Handwritten Digits Live Recognition", font = ("Helvetica Bold", 16), justify=tk.CENTER)
description = tk.Label(root, text="Write a digit on a piece of paper or draw it on your device and put it close to you webcam. \nMake sure that the stroke is thick enough to be seen and keep it steady in frame.\nBackground needs to be BLACK and the stroke color needs to be WHITE.", justify=tk.CENTER, wraplength=300)
webcam_image = tk.Label(root)
prediction_label = tk.Label(root, text = "Prediction: ", width=10, justify=tk.CENTER)
predicted = tk.Label(root, width = 5, font = ("Helvetica Bold", 20), justify=tk.CENTER)
confidence_label = tk.Label(root, text="Confidence: ", width=10, justify=tk.CENTER)
confidence = tk.Label(root, width = 5, justify=tk.CENTER)

title.grid(row = 0, column = 0, columnspan = 2, pady=10)
description.grid(row = 1, column=0, columnspan=2, pady=10)
webcam_image.grid(row = 2, column = 0, columnspan=2)
prediction_label.grid(row = 3, column = 0, sticky="E", pady=10)
predicted.grid(row = 3, column = 1, sticky="W", pady=10)
confidence_label.grid(row = 4, column = 0, sticky="E")
confidence.grid(row = 4, column = 1, sticky="W")

if __name__ == "__main__":
    Webcam(20, "ms")
    root.mainloop()
    cam.release()