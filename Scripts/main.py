class DrawDigit:

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    def __init__(self) -> None:

        from utils import Debug

        # Canvas Variables
        self.__CanvasWidth : int = 280
        self.__CanvasHeight : int = 280
        self.__PixelSize : int = 7
        self.__ImageShape : tuple = (1, 28, 28)
        self.__WindowDims : tuple = (550, 650)
        self.__Drawing : dict = {}
        self.__BrushWidth : int = 1
        self.__FirstInit : bool = True

        self.__Root = None
        self.__Canvas = None
        self.__BrushWidthEntry = None

        self.__Debug = Debug
    
    @property
    def __BrushWidth__(self):
        return self.__BrushWidth

    @__BrushWidth__.setter
    def __BrushWidth__(self, width : int):
        self.__BrushWidth = width

    def __CreateWindow__(self) -> None:
        """
        Create a Tkinter window.
        """
        try: 
            import tkinter as tk
            from tkinter import ttk
            from functools import partial
        except: raise Exception("You don't have tkinter installed. Please run 'pip install tkinter' to install the module.")

        if self.__Root is None: self.__Root = tk.Tk()
        self.__Root.geometry(f"{self.__WindowDims[0]}x{self.__WindowDims[1]}")
        self.__Root.eval("tk::PlaceWindow . center")
        self.__Root.title("MNIST Digit Recognition")
        self.__Root.resizable(False, False)

        # Initialise window components.
        self.__Frame = tk.Frame(self.__Root)
        self.__CanvasTitle = ttk.Label(self.__Frame, text = "MNIST Digit Recognition", font = ("Helvetica Bold", 20))
        self.__Description = ttk.Label(self.__Frame, text = "This applet can recognise handwritten digits. \nPress + or - to adjust the brush size. \nLoad your own .h5 model, press Enter/Return to confirm.")
        self.__BrushWidthEntryLabel = ttk.Label(self.__Frame, text = "Brush Stroke Width")
        self.__BrushWidthEntry = ttk.Entry(self.__Frame)
        self.__IncreaseBrushStrokeButton = ttk.Button(self.__Frame, text = "+", width = 1, command = partial(self.__ChangeBrushWidth__, "+"))
        self.__DecreaseBrushStrokeButton = ttk.Button(self.__Frame, text = "-", width = 1, command = partial(self.__ChangeBrushWidth__, "-"))
        self.__ModelFileEntryLabel = ttk.Label(self.__Frame, text = "Model File Name (.h5)")
        self.__ModelFileEntry = ttk.Entry(self.__Frame)
        self.__Canvas = tk.Canvas(self.__Frame, width = self.__CanvasWidth, height = self.__CanvasHeight, bg = "black")
        self.__GuessLabel = ttk.Label(self.__Frame, text = "Guess:", width = 5)
        self.__GuessDigit = ttk.Label(self.__Frame, font = ("Helvetica Bold", 23), width = 2)
        self.__ConfidenceLabel = ttk.Label(self.__Frame, text = "Confidence:", width = 10)
        self.__ConfidencePercentage = ttk.Label(self.__Frame, width = 10)
        self.__SaveImageButton = ttk.Button(self.__Frame, text = "Save Image", command = self.__SaveImage__)

        # Default text.
        self.__BrushWidthEntry.insert(0, str(self.__BrushWidth))
        self.__ModelFileEntry.insert(0, "network_for_mnist.h5")

        # Render components in the window.
        self.__Frame.pack(pady = 20, padx = 60, fill = "both")
        self.__CanvasTitle.grid(row = 0, column = 0, columnspan = 4, pady = 15)

        self.__Description.grid(row = 1, column = 0, columnspan = 4, pady = 5)
        self.__Description.configure(anchor = "center", justify = "center")

        self.__BrushWidthEntryLabel.grid(row = 2, column = 0)
        self.__BrushWidthEntry.grid(row = 2, column = 1)
        self.__IncreaseBrushStrokeButton.grid(row = 2, column = 2)
        self.__DecreaseBrushStrokeButton.grid(row = 2, column = 3)

        self.__ModelFileEntryLabel.grid(row = 3, column = 0)
        self.__ModelFileEntry.grid(row = 3, column = 1, columnspan = 3, sticky="nsew")

        self.__Canvas.grid(row = 4, column = 0, columnspan = 4, pady = 10)

        self.__GuessLabel.grid(row = 5, column = 0, columnspan = 2, pady = 20)
        self.__GuessDigit.grid(row = 5, column = 1, columnspan = 2, pady = 20)

        self.__ConfidenceLabel.grid(row = 6, column = 0, columnspan = 2)
        self.__ConfidencePercentage.grid(row = 6, column = 1, columnspan = 2)

        self.__SaveImageButton.grid(row = 7, column = 0, columnspan = 4, pady = 10)
    
    def __CreateCanvasGrid__(self, event = None) -> None:
        """
        Create a pixelated canvas.
        """
        __item_id : int = 0

        for x in range(0, self.__CanvasWidth, self.__PixelSize):
            for y in range(0, self.__CanvasHeight, self.__PixelSize):

                __item_id = self.__Canvas.create_rectangle(
                    x, y, x + self.__PixelSize, y + self.__PixelSize,
                    fill = "black", outline = "black"
                )

                self.__Canvas.tag_bind(__item_id)
                self.__Drawing.update({(int((__item_id - 1) // (self.__CanvasWidth / self.__PixelSize)), 
                                        int((__item_id - 1)  % (self.__CanvasWidth / self.__PixelSize))) : 0})
    
    def __ClearCanvas__(self, event = None) -> None:
        for px in self.__Drawing.keys():
                self.__Drawing.update({px : 0})

        cw, ch = int(self.__CanvasWidth / self.__PixelSize), int(self.__CanvasHeight / self.__PixelSize)

        for item_id in range(1, cw*ch + 1):
            self.__Canvas.itemconfig(item_id, {"fill" : "black"})

    
    def __GetItemIds__(self, event = None, *, dx : int = 1, dy : int = 1) -> tuple:
        """
        Get the item id of the rectangle you click on.
        """
        mx, my = event.x, event.y
        return self.__Canvas.find_overlapping(mx, my, mx + dx, my + dy)
    
    def __ColorPixel__(self, event = None, *, brushWidth : int = 1) -> None:
        """
        Color the pixels you click.
        """
        if self.__FirstInit:
            self.__BrushWidth__ = brushWidth
            self.__FirstInit = False

        Pixels = self.__GetItemIds__(event, dx = self.__BrushWidth__, dy = self.__BrushWidth__)

        for px in Pixels:
            self.__Canvas.itemconfig(px, {"fill" : "white"})
            self.__Drawing.update({(int((px - 1) // (self.__CanvasWidth / self.__PixelSize)),
                                    int((px - 1)  % (self.__CanvasWidth / self.__PixelSize))) : 255})
        
    
    def __ExtractImage__(self):
        try: 
            import numpy as np
            from PIL import Image
        except: raise Exception("You don't have the numpy module installed.")

        image = np.zeros((int(self.__CanvasWidth / self.__PixelSize), 
                          int(self.__CanvasHeight / self.__PixelSize)))
        
        for (px, py), pval in self.__Drawing.items():
            image[(px, py)] = pval

        image = np.array([np.uint8(image).T]) / 255.0
        image = Image.fromarray(image[0]).resize((28, 28))
        image = np.asarray(image).reshape(1, 28, 28)

        return image
    
    def __LoadModel__(self, event = None):
        try: from keras.models import load_model
        except: raise Exception("You don't have the keras module installed.")
        model_filename = self.__ModelFileEntry.get()
        self.__model = load_model(model_filename)
    
    def __Predict__(self, event = None) -> None:
        try: 
            import numpy as np
            import matplotlib.pyplot as plt
        except: raise Exception("You don't have the numpy module installed.")

        drawing = self.__ExtractImage__()
        prediction = self.__model.predict(drawing, verbose = 0)
        self.__GuessDigit.config(text = str(np.argmax(prediction)))
        prediction_confidence = max(list(zip(range(0, len(prediction.flatten())), prediction.flatten())), key = lambda x : x[1])
        self.__ConfidencePercentage.config(text = f"{prediction_confidence[1] * 100.0 : .2f}%")

    def __ChangeBrushWidth__(self, action : str, *, max_width : int = 20) -> None:

        if self.__FirstInit:
            self.__FirstInit = False

        if action == "+":
            if self.__BrushWidth__ < max_width:
                self.__BrushWidth__ += 1
        elif action == "-":
            if self.__BrushWidth__ > 0:
                self.__BrushWidth__ -= 1
        else:
            self.__Debug("You can only increase (+) or decrease (-) the brush stroke.")

        self.__BrushWidthEntry.delete(0, len(self.__BrushWidthEntry.get()))
        self.__BrushWidthEntry.insert(0, self.__BrushWidth__ )
    
    def __SaveImage__(self, event = None):
        try: 
            from PIL import Image
            import numpy as np
        except: raise Exception("You don't have the Pillow module installed.")

        image = np.zeros((int(self.__CanvasWidth / self.__PixelSize), 
                          int(self.__CanvasHeight / self.__PixelSize)))
        
        for (px, py), pval in self.__Drawing.items():
            image[(px, py)] = pval

        image = Image.fromarray(np.uint8(image).T)
        image = image.resize((28, 28))
        image.save("Drawing.png", format = "png")

    def Display(self) -> None:
        """
        Display window.
        """
        self.__CreateWindow__()
        self.__CreateCanvasGrid__()
        self.__LoadModel__()
        
        self.__Canvas.bind("<B1-Motion>", self.__ColorPixel__, add = "+")
        self.__Canvas.bind("<B1-Motion>", self.__Predict__, add = "+")
        self.__Root.bind("<Command-r>", self.__ClearCanvas__)
        self.__ModelFileEntry.bind("<Return>", self.__LoadModel__)

        self.__Root.mainloop()

if __name__ == "__main__":
    Window = DrawDigit()
    Window.Display()