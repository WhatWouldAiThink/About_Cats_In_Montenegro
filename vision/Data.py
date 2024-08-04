"""
This Module contains a subclassed Dataset.
"""

# import basics
from pathlib import Path

# handling data
import pandas as pd
import numpy as np

# for plotting and colormaps
import matplotlib.pyplot as plt

# import torch
import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset

# pylint: disable=W0401
# pylint: disable=W0614
from .Definitions import *

torch.manual_seed(0)


# Define the data class used for training and plotting
class Data(Dataset):
    """
    Custom Dataset for loading image samples and
    their corresponding labels from a DataFrame.
    """

    def __init__(self, dataFrame: pd.DataFrame):
        """
        Parameters
        ----------
        name : dataFrame
            The DataFrame to load.
        """
        self.len = dataFrame.shape[0]
        self.dataFrame = dataFrame

    def __getitem__(self, index: int) -> tuple[torch.Tensor, bool]:
        """
        Return an image as tensor. The images is normalised
        and resized to MIN_SIZE_IMAGE.

        Parameters
        ----------
        index : int
            The index of the image to load

        Returns
        -------
        tuple[torch.Tensor, bool]
            The image and the label of the image
        """
        return self.__getImage__(index, normalise=True, resize=False)[0]

    # Get Length
    def __len__(self) -> int:
        """
        Return the number of images.

        Returns
        -------
        int
            The number of images in the DataFrame
        """
        return self.len

    def __getImage__(
        self, index: int, normalise: bool, resize: bool
    ) -> tuple[tuple[torch.Tensor, bool], str]:
        """
        Return an image as tensor.
        The images can be normalised
        to a predefinied mean and std
        and resized to MIN_SIZE_IMAGE.

        Parameters
        ----------
        index : int
            The index of the image to load
        normalise : bool
            A flag to normalise the image
        resize : bool
            A flag to resize the image

        Returns
        -------
        tuple[torch.Tensor, bool]
            The image and the label of the image
        str
            The filename of the image
        """

        # Normalize 1 or 3 channels depending on the shape
        shapeImage: tuple[int, int, int] = read_image(
            self.dataFrame["fileName"][index]
        ).shape
        # print(f"shape: {shapeImage}")
        if shapeImage[0] == 3:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif shapeImage[0] == 1:
            mean = [0.5]
            std = [0.5]
        else:
            assert False, f"No normalisation for image shape {shapeImage} defined"

        listTransform: list[Any] = []
        listTransform.append(transforms.ToTensor())

        if resize:
            listTransform.append(
                transforms.Resize(MIN_SIZE_IMAGE, transforms.InterpolationMode.BILINEAR)
            )
        if normalise:
            listTransform.append(transforms.Normalize(mean, std))

        composed: Any = transforms.Compose(listTransform)

        image: torch.Tensor = torch.Tensor()
        # read image and convert to numpy to apply ToTensor() and Normalise
        image = read_image(self.dataFrame["fileName"][index]).permute(1, 2, 0).numpy()

        return (
            composed(image),
            self.dataFrame["label"][index],
        ), self.dataFrame[
            "fileName"
        ][index]

    def getImage(
        self, index: int, normalise: bool = False, resize: bool = False
    ) -> tuple[tuple[torch.Tensor, bool], str]:
        """
        Return an image as tensor. The images can be normalised
        and resized to MIN_SIZE_IMAGE. Use this method to plot
        an image.

        Parameters
        ----------
        index : int
            The index of the image to load
        normalise : bool
            A flag to normalise the image
        resize : bool
            A flag to resize the image

        Returns
        -------
        tuple[torch.Tensor, bool]
            The image and the label of the image
        str
            The filename of the image
        """
        return self.__getImage__(index, normalise, resize)

    def plotImages(
        self, indices: list[int], normalise: bool = False, resize: bool = False
    ):
        """
        Plot a image or list of images

        Parameters
        ----------
        indices : list[int]
            The indices of the images to plot
        normalise : bool
            A flag to normalise the image
        resize : bool
            A flag to resize the image
        """

        _ = plt.figure(figsize=(figSize[0] * len(indices), figSize[1] * len(indices)))

        for count, index in enumerate(indices):
            # never normalise the image, they are allready
            # shifted to a range of [0,1] by ToTensor()
            (image, label), fileName = self.getImage(index, False, resize)

            # remove the prefix from file name e.g. dataset/train_set/cats/cat.1880.jpg
            # to cat.1880.jpg
            title = "/".join(Path(fileName).parts[3::])

            axis = plt.subplot(1, len(indices), count + 1)
            axis.set_title(f"{title} : {labelDic[label]}")
            axis.imshow(image.permute(1, 2, 0).numpy())

        plt.show()

    def plotHistogram(
        self, indices: list[int], normalise: bool = False, resize: bool = False
    ):
        """
        Plot a histogram for list of images

        Parameters
        ----------
        indices : list[int]
            The indices of the images to plot
        normalise : bool
            A flag to normalise the image
        resize : bool
            A flag to resize the image
        """

        _ = plt.figure(figsize=(figSize[0] * len(indices), figSize[1]))
        # myDpi=96
        # _ = plt.figure(figsize=(224/myDpi, 224/myDpi), dpi=myDpi)
        yMax = 0
        bins = 100

        # calc the maximum counts from histogram to scale the y axis
        # equal for all histograms
        for count, index in enumerate(indices):
            # never normalise the image, they are allready
            # shifted to a range of [0,1] by ToTensor()
            (image, _), fileName = self.getImage(index, False, resize)
            counts, _ = np.histogram(image.flatten().numpy(), bins=bins)
            if yMax < np.amax(counts):
                yMax = np.amax(counts)

        for count, index in enumerate(indices):
            # never normalise the image, they are allready
            # shifted to a range of [0,1] by ToTensor()
            (image, _), fileName = self.getImage(index, False, resize)
            # calc the histogram
            counts, _ = np.histogram(image.flatten().numpy(), bins=bins)
            # remove the prefix from file name e.g. dataset/train_set/cats/cat.1880.jpg
            # to cat.1880.jpg
            title = "/".join(Path(fileName).parts[3::])

            axis = plt.subplot(1, len(indices), count + 1)
            axis.hist(image.flatten().numpy(), bins=bins)
            axis.set_ylim(0, yMax + 500)
            axis.set_xlim(0, 1.0)
            # plot histogram with precalculated counts and bins
            axis.set_title(f"Histogram for\n {title}", y=0.82, fontsize=20)
            axis.tick_params(axis="x", labelsize=16)
            axis.tick_params(axis="y", labelsize=16)

        plt.tight_layout()
