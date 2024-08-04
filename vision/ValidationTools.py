"""Module contains """

# handling data
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# import torch
import torch
from torchvision.io import read_image

# custom modules
from plotting.customPlotting import PiePlot
from .Utils import printNumOfLearnableWeights, printModelSize

# pylint: disable=W0401
# pylint: disable=W0614
from .Definitions import *

torch.manual_seed(0)


class ConfusionMatrix:
    """
    This class is to calculate and plot the confusion matrix.
    """

    def __init__(self, y, y_pred, labels: dict[int, str], title: str):
        """
        Calculate the confusion matrix.

        Parameters
        ----------
        y: np.array
            The true labels.
        y_pred: np.array
            The predicted labels.
        labels: dict[int,str]
            A dictionary with the label names e.g. {0 : "Dog", 1 : "Cat"}
        title: str
            A title for the confusion matrix.
        """
        self.y = y
        self.y_pred = y_pred
        self.title = title
        self.accuracy: float = 0
        self.dicLabels = labels
        # create dictionary to count true and false classified
        # samples from a confusion matrix e.g. self.resultsConfusionMatrix =
        # {"trueDog": 0, "falseDog": 0, "trueCat": 0, "falseCat": 0}
        self.resultsConfusionMatrix: dict[str, int] = {}

        # calculate the confusion matrix
        confMatrix = confusion_matrix(
            self.y, self.y_pred, labels=list(self.dicLabels.keys())
        )
        print(f"Confusion Matrix: {confMatrix}")
        # The cells of a confusion matrix (actual values vs predicted values)
        # can be interpreted as following example for 3 classes.
        #
        #                         PREDICTED CLASS
        #         -----------------------------------------
        # Class0 | trueClass0  | falseClass1 | falseClass2 |
        # Class1 | falseClass0 | trueClass1  | falseClass2 |  ACTUAL CLASS
        # Class2 | falseClass0 | falseClass1 | trueClass2  |
        #         -----------------------------------------
        #             Class0       Class1        Class2

        # calculate the number of true and false classified samples for
        # every class and fill the dictionary e.g.
        # self.resultsConfusionMatrix:
        # {'trueClass0': 0, 'trueClass1': 1, 'falseClass0': 59, 'falseClass1': 0}
        for i, row in enumerate(confMatrix):
            for j, _ in enumerate(row):
                # print(f"i,j: {i},{j} = {el}")
                key: str = ""
                if i == j:
                    # get the true classified classes
                    key = f"true{self.dicLabels[i]}"
                    # print(f"key: {key}")
                    self.resultsConfusionMatrix[key] = confMatrix[i, j]
                else:
                    # sum up the false classified classes
                    key = f"false{self.dicLabels[j]}"
                    # print(f"key: {key}")
                    if key in list(self.resultsConfusionMatrix.keys()):
                        self.resultsConfusionMatrix[key] += confMatrix[i, j]
                    else:
                        self.resultsConfusionMatrix[key] = confMatrix[i, j]

        # print the true and false classified classes
        width: int = 10
        for key in list(self.resultsConfusionMatrix.keys()):
            print(
                f"Found {str(self.resultsConfusionMatrix[key]).center(width,' ')} "
                f"images classified as {key.ljust(width,' ')} in {len(self.y)} images."
            )

        # calc the accuracy from the confusionmatrix
        # np.trace return the sum along diagonals of the array.
        self.accuracy = np.trace(confMatrix) / np.sum(confMatrix)

    def plot(self, title: str, path: str):
        """
        Plot the confusion matrix.

        Parameters
        ----------
        title: str
            The titel for the plot.
        path: str
            The path to save the plot.
        """

        colors: list[tuple[float, float, float]] = [
            tab10Colors["tab:olive"],
            tab10Colors["tab:cyan"],
        ]

        # create a colormap with N bins
        cmapOliveCyan: LinearSegmentedColormap = LinearSegmentedColormap.from_list(
            "olive_cyan", colors, N=10
        )

        # plot a confusion matrix
        disp: ConfusionMatrixDisplay = ConfusionMatrixDisplay.from_predictions(
            y_true=self.y,
            y_pred=self.y_pred,
            labels=list(self.dicLabels.keys()),
            display_labels=list(self.dicLabels.values()),
            cmap=cmapOliveCyan,
            colorbar=False,
            text_kw={"color": "black", "fontsize": "20"},
        )
        # plot the title + the accuracy in %
        disp.ax_.set_title(
            f"{title}:\n Accuracy {100*self.accuracy:.2f}%",
            fontdict={"size": fontSizesPlot["title"] - 2},
        )
        # setting the fontsize for the confusion matrix is not so easy
        # i have to overrite the x and y labels to set the fontsize
        disp.ax_.set_xlabel(
            xlabel=disp.ax_.get_xlabel(),
            fontdict={"size": fontSizesPlot["xy_labels"] - 1},
        )
        disp.ax_.set_ylabel(
            ylabel=disp.ax_.get_ylabel(),
            fontdict={"size": fontSizesPlot["xy_labels"] - 1},
        )
        # remove whitespace and linebreak for filenames
        path = path.replace(" ", "").replace("\n", "")
        disp.figure_.savefig(
            f"{path}.svg",
            bbox_inches="tight",  # ensure that the whole legend is visible
            # pad_inches=0.1,  # ensure that the whole legend is visible
            dpi="figure",
            format="svg",
        )

    def plotPie(self, title: str, path: str):
        """
        Plot a pie graph for a confusion matrix
        when we have only samples for one class e.g cats.
        # e.g. confMatrix = [[0, 0][45, 16]]

        Parameters
        ----------
        title: str
            The title for the pie chart.
        path:  str
            The path to save the plot.
        """

        # calculate the confusion matrix
        confMatrix = confusion_matrix(
            self.y, self.y_pred, labels=list(self.dicLabels.keys())
        )

        # plot only when we have only samples for one class e.g cats
        # confMatrix = [[0, 0][45, 16]]
        # np.sum(confMatrix, axis = 1) leads to [0, 61]
        if np.sum(confMatrix, axis=1)[0] == 0:
            piePlot = PiePlot(
                # remove whitespace and linebreak for filenames
                title=f"{title}: Accuracy {100*self.accuracy:.2f}%",
                path=path.replace(" ", "").replace("\n", ""),
            )
            piePlot.plot(confMatrix[1], labels=list(self.dicLabels.values()))


class MyValidator:
    """
    This class is to analyse the prediction
    of a trained model.
    False and True classified predictions
    are counted, printed and visualized by
    a confusion matrix.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        checkPointPath: str,
        dataset: torchDataSet,
        dicLabels: dict[int, str],
        title: str,
    ):
        """
        Parameters
        ----------
        model: torch.nn.Module
            The model to analyse.
        checkPointPath: str
            The checkpoint to load by the model.
        dataset: torchDataSet
            The dataset to make predictions with the model.
        titel: str
            The title for the confusion matrix.
        """
        self.model = model
        self.dataSet = dataset
        self.title = title
        self.numImages: int = self.dataSet.__len__()
        # create a list for indices of true and false predicted images
        self.falsePred: list[int] = []
        self.truePred: list[int] = []
        # load the checkpoint
        print(f"\nLoading checkpoint: {checkPointPath}")
        checkpoint: Any = torch.load(checkPointPath)
        # load the model parameters from the checkpoint
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.dicLabels = dicLabels
        self.confMatrix: ConfusionMatrix = None
        self.predict(self.dicLabels)

    def predict(self, dicLabels: dict[int, str]):
        """
        Make predictions with and count true and false classified.

        Parameters
        ----------
        dicLabels: dic[int,str]
            The dictionary with the labels as
            int and str e.g. {0: "Dog", 1: "Cat"}
        """
        # create dictionary to count true and false classified samples from a confusion matrix e.g.
        # self.resultsConfusionMatrix = {"trueDog": 0, "falseDog": 0, "trueCat": 0, "falseCat": 0}
        self.resultsConfusionMatrix: dict[str, int] = {}
        self.dicLabels = dicLabels
        correct: int = 0
        notCorrect: int = 0
        im: torch.Tensor = torch.Tensor()
        label: int = 0
        y_pred = np.empty(shape=[self.numImages], dtype=int)
        y = np.empty(shape=[self.numImages], dtype=int)

        for index in range(self.numImages):
            # normalise the image for prediction because we trained it normalised
            (im, label), _ = self.dataSet.getImage(index, normalise=True)
            # add one dimension cause the model expects a batch, a 4D input
            # im.shape: torch.Size([1, 100, 100]) -> im.shape: torch.Size([1, 1, 100, 100])
            z: torch.Tensor = self.model(im[None, :])

            yhat = self.__getPredictedClass__(z)

            y_pred[index] = yhat
            y[index] = label

            # calc true and false classified images
            correct += yhat == label
            notCorrect += yhat != label

            # store the index for true and false predicted image
            if yhat != label:
                self.falsePred.append(index)
            else:
                self.truePred.append(index)

        assert self.numImages == (notCorrect + correct)
        width: int = 10
        print(
            f"Found {str(correct).center(width,' ')} "
            f"true classified images in {self.numImages} images."
        )
        print(
            f"Found {str(notCorrect).center(width,' ')} "
            f"false classified images in {self.numImages} images.\n"
        )

        # calc the confusion matrix to plot it later
        self.confMatrix = ConfusionMatrix(y, y_pred, self.dicLabels, self.title)

    def plotConfusionMatrix(self, title: str, path: str):
        """
        Plot a confusion matrix.

        Parameters
        ----------
        title: str
            The title for the pie chart.
        path:  str
            The path to save the image.
        """
        self.confMatrix.plot(title, path)

    def plotPie(self, title: str, path: str):
        """
        Plot a pie graph for a confusion matrix
        when we have only samples for one class e.g cats.
        # e.g. confMatrix = [[0, 0][45, 16]]

        Parameters
        ----------
        title: str
            The title for the pie chart.
        path:  str
            The path to save the image.
        """
        self.confMatrix.plotPie(title, path)

    def plotClassifiedImages(
        self, classType: str = "False", number: int = 0, resize: bool = False
    ):
        """
        Plot the false or true classified images.

        Parameters
        ----------
        classType: str
            The class to plot, 'True' or 'False'.
        number: int
            The number of images to plot.
        resize: bool
            Parameter to resize the image for plotting.
        """
        assert number > 0, "'number' of Images to plot must be > 0"

        predictions: list[int] = []
        if classType == "False":
            predictions = self.falsePred
        elif classType == "True":
            predictions = self.truePred
        else:
            assert False, "Only 'False' and 'True' are valid values for classType!"

        im: torch.Tensor = torch.Tensor()
        label: int = 0
        for index in predictions[0:number:1]:
            # normalise the image because we trained it normalised
            (im, label), _ = self.dataSet.getImage(index, normalise=True)

            # add one dimension cause the model expects a batch, a 4D input
            # im.shape: torch.Size([1, 100, 100]) -> im.shape: torch.Size([1, 1, 100, 100])
            z: torch.Tensor = self.model(im[None, :])

            yhat = self.__getPredictedClass__(z)
            # yhat: int = int((z.detach().numpy()[0] > 0.5))

            print(f"Real: {labelDic[label]} -- Classified: {labelDic[yhat]}")
            print(f"Label: {label} -- Yhat: {yhat}")

            # dont normalise the image for ploting
            self.dataSet.plotImages([index], normalise=False, resize=resize)

    def __getPredictedClass__(self, z: torch.Tensor) -> int:
        """
        Calc and return the predicted class from a 1 or 2
        dimensional tensor.

        Parameters
        ----------
        z : torch.Tensor
            The probability for the predicted classes
            with shape [1, numLabels] or
            with shape [1,1] for binary classification.

        Returns:
        -------
            int: The predicted class.
        """

        yhat: int | None = None

        # This is a hack!!!
        # Because pretrained models from pytorch dont have the member criterion
        # we will always use "CE" as the criterion for standard models
        if not hasattr(self.model, "criterion"):
            # the class with the highest value is what we choose as prediction
            _, yhat = torch.max(z.detach(), 1)
            yhat = int(yhat)
            return yhat

        if self.model.criterion == "CE":
            # the class with the highest value is what we choose as prediction
            _, yhat = torch.max(z.detach(), 1)
            yhat = int(yhat)
        elif self.model.criterion == "BCE":
            # map float torch Tensor to 0 or 1
            yhat = int((z.detach().numpy()[0] > 0.5))
        else:
            assert False, "No known criterion definied!"

        return yhat


class ValidatePretrainedModel:
    """
    A generic class to predict labels for images with
    pretrained models trained on IMAGENET1K_V1 dataset.
    See https://pytorch.org/vision/stable/models.html.
    Only binary prediction is possible.
    """

    def __init__(
        self,
        model: torchVisionModels,
        weights: torchVisionWeights,
        category: list[str],
    ):
        """
        Init the model with pretrained weights.

        Parameters
        ----------
        model: torchVisionModels
            A torchvision.model.
        weights: torchVisionWeights
            Pretrained weights for torch vision models.
        category: list[str]
            A list with category names according to IMAGENET1K_V1
            e.g. ["Siamese cat", "Egyptian cat"]. These list
            defines one class. All other categories will be
            handled as the second class.
            For all available categories see
            https://www.image-net.org/challenges/LSVRC/2017/browse-synsets.php

        """
        self.weights = weights
        # Step 1: Initialize model with the best available weights
        self.model = model(self.weights)

        self.model.eval()
        # define the subcategories
        self.category = category
        self.confMatrix: ConfusionMatrix | None = None
        # print learnable weights
        printNumOfLearnableWeights(self.model)
        printModelSize(self.model)
        self.dicLabels: dict[int, str] | None = None

    def predict(self, dataFrame: pd.DataFrame, dicLabels: dict[int, str]):
        """
        Calculate a 2x2 confusion matrix for images in
        a DataFrame based on the prediction of a pretrained model.
        All categories in self.category will be treated as one class.
        And all other categories will be treated as another class.

        Parameters
        ----------
        dataFrame: pd.DataFrame
            A DataFrame with the path to an image and the label for the image.
        dicLabels: dict[int,str]
            A dictionary with the label names e.g. {0 : "Dog", 1 : "Cat"}
        """
        self.dicLabels = dicLabels
        # Step 2: Initialize the inference transforms
        preprocess = self.weights.transforms()
        # The preprocessing steps are:
        # preprocess: ImageClassification(
        #    crop_size=[224]
        #    resize_size=[232]
        #    mean=[0.485, 0.456, 0.406]
        #    std=[0.229, 0.224, 0.225]
        #    interpolation=InterpolationMode.BILINEAR
        # )

        # create arrays for predicted and true labels
        y_pred = np.empty(shape=[len(dataFrame)], dtype=int)
        y = np.empty(shape=[len(dataFrame)], dtype=int)

        for i in range(len(dataFrame)):
            # Get the true label
            y[i] = dataFrame.iloc[i]["label"]
            # Step 3: Apply inference preprocessing transforms
            # see 'Using the pre-trained models' on https://pytorch.org/vision/stable/models.html
            img: torch.Tensor = preprocess(
                read_image(dataFrame.iloc[i]["fileName"])
            ).unsqueeze(0)

            # Step 4: Use the model and print the predicted category
            prediction: torch.Tensor = self.model(img).squeeze(0).softmax(0)
            classId: int = prediction.argmax().item()
            categoryName: str = self.weights.meta["categories"][classId]

            # print the category and the propability
            # cat = str(categoryName in self.category)
            # score: torch.Tensor = prediction[classId].item()
            # print(f"{cat.ljust(10,' ')}{categoryName.ljust(30,' ')} {100 * score:.1f}%")

            # this is a specific implementation for binary classification
            # every category which is in self.category will be handled as the first class
            # any other category will be handled as the second class
            if categoryName in self.category:
                y_pred[i] = 1
            else:
                y_pred[i] = 0

        # calc the confusion matrix to plot it later
        self.confMatrix = ConfusionMatrix(y, y_pred, self.dicLabels, "Test")

    def plotConfusionMatrix(self, title: str, path: str):
        """
        Plot a confusion matrix.

        Parameters
        ----------
        title: str
            The title for the pie chart.
        path:  str
            The path to save the image.
        """
        self.confMatrix.plot(title, path)

    def plotPie(self, title: str, path: str):
        """
        Plot a pie graph for a confusion matrix
        when we have only samples for one class e.g cats.
        # e.g. confMatrix = [[0, 0][45, 16]]

        Parameters
        ----------
        title: str
            The title for the pie chart.
        path:  str
            The path to save the image.
        """
        self.confMatrix.plotPie(title, path)
