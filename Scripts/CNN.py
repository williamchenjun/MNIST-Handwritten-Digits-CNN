class MNIST_CNN:
    def __init__(self):
        self.model = None
        self.history = None

    def __build_network__(self):
        try: 
            from keras.models import Sequential
            from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, BatchNormalization, Flatten
        except:
            raise Exception("Failed to import Keras or module is missing.")

        model = Sequential()

        model.add(Conv2D(32, kernel_size = (5, 5), activation = 'relu', input_shape = (28, 28, 1)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size = (5, 5), activation = 'relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2, 2), 1))
        model.add(Dropout(.25))

        model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2, 2), 2))
        model.add(Dropout(.25))

        model.add(Flatten())

        model.add(Dense(256, activation = 'relu'))
        model.add(Dense(128, activation = 'relu'))
        model.add(BatchNormalization())
        model.add(Dropout(.5))
        model.add(Dense(10, activation = 'softmax'))

        if self.model is None:
            self.model = model

        return self.model
    
    def __process_data__(self, to_cat: bool = True, reshape: bool = True):
        try: 
            from keras.datasets import mnist
            from keras.utils import to_categorical
        except: raise Exception("Failed to import Keras or module is missing.")

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if reshape:
            x_train = x_train.reshape(len(x_train), 28, 28, 1).astype("float32") / 255.0
        if to_cat:
            y_train = to_categorical(y_train, 10)
        
        x_test = x_test.reshape(len(x_test), 28, 28, 1).astype("float32") / 255.0
        y_test = to_categorical(y_test, 10)
        

        return x_train, y_train, x_test, y_test
    
    def train(self, optimizer = "adam", loss_func = "categorical_crossentropy", verbose : int = 1):
        try: 
            from keras.callbacks import ReduceLROnPlateau, EarlyStopping
            from utils import augment
            from keras.utils import to_categorical
        except: raise Exception("Failed to import Keras or module is missing.")

        if self.model is None:
            self.model = self.__build_network__()
        
        x_train, y_train, x_test, y_test = self.__process_data__(False, False)
        x_train, y_train = augment(x_train, y_train, max_shift = 6, max_rotation = 20, max_zoom = 2, noise_intensity = .1, outputs = 5, fill = 255)
        x_train = x_train.astype("float32") / 255.0
        y_train = to_categorical(y_train, 10)
        reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = .2, patience = 7, min_lr = 1e-5, verbose = 0)
        early_stop = EarlyStopping(monitor = "val_loss", patience = 7)
        self.model.compile(optimizer = optimizer, loss = loss_func, metrics = ["accuracy"])

        self.history = self.model.fit(
            x_train.reshape(len(x_train), 28, 28, 1), y_train,
            batch_size = 256,
            epochs = 30,
            validation_split = 0.2,
            callbacks = [reduce_lr, early_stop],
            verbose = verbose
        )

        loss, acc = self.model.evaluate(x_test, y_test)

        print(f"\nTest Loss: {loss : .2f} – Test Accuracy: {acc*100.0 : .2f}%\n")
    
    def plot_loss_accuracy(self):
        try: import matplotlib.pyplot as plt
        except: raise Exception("Failed to import matplotlib or module is missing.")

        training_loss = self.history.history.get("loss")
        val_loss = self.history.history.get("val_loss")
        training_acc = self.history.history.get("accuracy")
        val_acc = self.history.history.get("val_accuracy")

        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        ax = ax.ravel()

        ax[0].plot(training_loss, label = "Training Loss")
        ax[0].plot(val_loss, label = "Validation Loss")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Loss")
        ax[0].legend()

        ax[1].plot(training_acc, label = "Training Accuracy")
        ax[1].plot(val_acc, label = "Validation Accuracy")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()

        fig.suptitle("Loss and Accuracy Plots")

        import os
        count = 0
        for file in os.listdir("Plots"):
            if file.startswith("Loss_and_Accuracy") and file.endswith(".png"):
                count += 1

        plt.savefig(f"Plots/Loss_and_Accuracy_{count}.png", format = "png", dpi = 300)
    
    def save(self, filename = "network_for_mnist.h5"):
        try:import numpy as np
        except: raise Exception("Failed to import numpy or module is missing.")

        self.model.save(filename)
        np.save("history_for_mnist.npy", self.history)
    
    def load(self, filename = "network_for_mnist.h5"):
        try: 
            from keras.models import load_model
            import numpy as np
            import os
        except: raise Exception("Failed to import keras or module is missing.")
        
        for file in os.listdir(os.getcwd()):
            if filename in file:
                self.model = load_model(filename)
                self.history : dict = np.load('history_for_mnist.npy', allow_pickle='TRUE').item()
                return self.model
        print(f"The file '{filename}' does not exist or is not in the current directory: {os.getcwd()}.")
    
    def __load_image__(self, filename : str):
        try: from keras.utils import load_img, img_to_array
        except: raise Exception('Failed to import keras or module is missing.')

        img = load_img(filename, color_mode="grayscale", target_size=(28, 28))
        img = img_to_array(img)
        img = img.reshape(1, 28, 28).astype('float32') / 255.0
        return img
    
    def __labelled_sample__(self, data, labels, size : int = 1000):
        try: import numpy as np
        except: raise Exception("Failed to import numpy or module is missing.")

        assert len(data) == len(labels) and size > 0

        result = []
        lbls = []
        k = 0
        
        while k < size:
            num = np.random.randint(0, len(data))
            choice = data[num]
            choice_label = labels[num]
            if id(choice) in map(id, result):
                continue
            result.append(choice)
            lbls.append(choice_label)
            k += 1

        return np.array(result), np.array(lbls)

    def predict(self, use_mnist : bool = True, samples : int = None, filename : str = None, file_true_label : int = None, plot : bool = False, rank : bool = False, *, invert_color : bool = False, center: bool = False):

        try: 
            import numpy as np
            from utils import invert
        except: raise Exception("Failed to import numpy or module is missing.")
        
        assert (use_mnist and samples is not None and filename is None) or (filename is not None and not use_mnist and samples is None)

        if use_mnist:
            _, _, x_test, y_test = self.__process_data__()
            x_test, y_test = self.__labelled_sample__(x_test, y_test, samples)
            if invert_color: x_test = np.array(list(map(invert, x_test)))
                    
            predictions = self.model.predict(x_test)

            if rank:
                for prediction, true_label in zip(predictions, y_test):
                    print(f"Prediction: {np.argmax(prediction)} – True Label: {np.where(true_label == 1)[0][0]}")
                
                    probabilities = prediction.flatten()
                    ranking = sorted(list(zip(np.arange(0, 10), probabilities)), 
                                    reverse=True, 
                                    key = lambda x: x[1])
                                    
                    for guess, confidence in ranking:
                        print(f"Guess: {guess} – Confidence: {confidence*100.0 :.2f}%")

            else:
                for prediction, true_label in zip(predictions, y_test):
                    print(f"Prediction: {np.argmax(prediction)} – True Label: {np.where(true_label == 1)[0][0]}")

        else:
            try: from utils import center_image, load_image
            except: raise Exception("Failed to import from utils.")

            image = load_image(filename)
            if center: image = center_image(image)
            if invert_color: image = np.array([invert(image)])
            prediction = self.model.predict(image.reshape(1, 28, 28, 1))

            if rank:
                probabilities = prediction.flatten()
                ranking = sorted(list(zip(np.arange(0, 10), probabilities)), 
                                reverse=True, 
                                key = lambda x: x[1])
                                
                for guess, confidence in ranking:
                    print(f"Guess: {guess} – Confidence: {confidence*100.0 :.2f}%")
            else:
                print(f"Prediction: {np.argmax(prediction)} – True Label: {file_true_label}")
      
        
        if plot:
            try: import matplotlib.pyplot as plt
            except: raise Exception("Failed to import matplotlib or module is missing.")

            if use_mnist:
                fig, ax = plt.subplots(int(np.ceil(samples/5)), 5, figsize = (6, 10))
                axes = ax.ravel()
                i = 0

                for prediction, image, true_label, ax in zip(predictions, x_test, y_test, axes):
                    if invert_color: image = invert(image)
                    ax.imshow(image.reshape(28, 28), cmap = "gray_r")
                    ax.set_title(f"Prediction: {np.argmax(prediction)} – Confidence: {np.max(prediction)*100.0 : .2f} – True Label: {np.where(true_label == 1)[0][0]}", size = 3)
                    i += 1
                
                for ax in axes[i:]:
                    fig.delaxes(ax)

                import os
                count = 0
                for file in os.listdir("Plots"):
                    if file.startswith("Predictions") and file.endswith(".png"):
                        count += 1

                plt.tight_layout()
                if samples > 5: plt.subplots_adjust(wspace = 1, hspace = -0.82)
                else: plt.subplots_adjust(wspace = 1)
                plt.savefig(f"Plots/Predictions_{count}.png", format = "png", dpi = 300, bbox_inches='tight')
            else:
                if invert_color: image = invert(image)
                plt.imshow(image.reshape(28, 28), cmap = "gray")

                import os
                count = 0
                for file in os.listdir("Plots"):
                    if file.startswith("Predictions") and file.endswith(".png"):
                        count += 1

                plt.title(f"Prediction: {np.argmax(prediction)} – Confidence: {np.max(prediction)*100.0 : .2f} – True Label: {file_true_label}")
                plt.savefig(f"Plots/Predictions_{count}.png", format = "png", dpi = 300, bbox_inches='tight')
    
    def __prepare_unseen_dataset__(self, pathname: str , categories: int, limit: int, center: bool):
        try: 
            from utils import load_image, center_image
            import os
        except: raise Exception("Failed to import numpy or utils, or module(s) is missing.")

        folder = os.listdir(f"{os.getcwd()}/{pathname}")
        folder = [file for file in folder if not file.startswith(".DS_Store")]
        digits = {i : [] for i in range(0, categories)} 
        
        for label, digit_folder in enumerate(sorted(folder)):
            for count, digit in enumerate(os.listdir(f"{pathname}/{digit_folder}")):
                if digit.startswith(".DS_Store"): continue
                if limit is not None and count > limit: break
                image = load_image(f"{pathname}/{digit_folder}/{digit}")
                if center: image = center_image(image)
                digits[label].append(image)
        
        return digits
    
    def unseen_sample(self, pathname: str = "new_unseen_data", categories: int = 10, limit: int = 100, samples: int = 5, *, invert_color : bool = False, center : bool = False):
        try: 
            import numpy as np
            from utils import invert
        except: raise Exception('Failed to import numpy or module is missing.')
        dataset = self.__prepare_unseen_dataset__(pathname, categories, limit, center)
        sample = []
        categories = len(dataset.keys())

        for _ in range(samples):
            label = np.random.randint(0, categories)
            rand_list = dataset[label]
            digit = rand_list[np.random.randint(0, len(rand_list))]
            if id(digit) in map(id, sample):
                continue
            if invert_color: digit = invert(digit)
            sample.append((digit, label))

        x_test, y_test = zip(*sample)
        x_test = np.array(list(x_test)).astype("float32") / 255.0
        y_test = np.array(list(y_test))

        return x_test, y_test

    def predict_unseen(self, x_test, y_test, plot: bool = False, rank: bool = False):
        try: 
            import numpy as np
            import os
        except: raise Exception("Failed to import numpy or utils, or module(s) is missing.")
        
        predictions = self.model.predict(x_test.reshape(len(x_test), 28, 28, 1))

        if rank:
            correct = 0
            print("\n")
            for prediction, true_label in zip(predictions, y_test):
                if int(np.argmax(prediction)) == int(true_label): correct += 1
                print(f"Prediction: {np.argmax(prediction)} – True Label: {true_label}")
            
                probabilities = prediction.flatten()
                ranking = sorted(list(zip(np.arange(0, 10), probabilities)), 
                                reverse=True, 
                                key = lambda x: x[1])
                                
                for guess, confidence in ranking:
                    print(f"Guess: {guess} – Confidence: {confidence*100.0 :.2f}%")
                print("\n")
        else:
            correct = 0
            print("\n")
            for prediction, true_label in zip(predictions, y_test):
                if int(np.argmax(prediction)) == int(true_label): correct += 1
                print(f"Prediction: {np.argmax(prediction)} – True Label: {true_label}")
            print("\n")
        
        print(f"Guessed Correctly: {correct}/{len(x_test)}\n")
        
        if plot:
            try: import matplotlib.pyplot as plt
            except: raise Exception("Failed to import matplotlib or module is missing.")

            fig, ax = plt.subplots(int(np.ceil(len(x_test)/5)), 5)
            axes = ax.ravel()
            i = 0

            for prediction, image, true_label, ax in zip(predictions, x_test, y_test, axes):
                ax.imshow(image.reshape(28, 28), cmap = "gray")
                ax.set_title(f"Prediction: {np.argmax(prediction)} – Confidence: {np.max(prediction)*100.0 : .2f} – True Label: {true_label}", size = 3)
                i += 1
            
            for ax in axes[i:]:
                fig.delaxes(ax)

            import os
            count = 0
            for file in os.listdir("Plots"):
                if file.startswith("Unseen_Predictions") and file.endswith(".png"):
                    count += 1

            plt.tight_layout()
            if len(x_test) > 5: plt.subplots_adjust(wspace = 1, hspace = -0.5)
            fig.suptitle("Prediction of Unseen Data")
            plt.savefig(f"Plots/Unseen_Predictions_{count}.png", format = "png", dpi = 300, bbox_inches='tight')