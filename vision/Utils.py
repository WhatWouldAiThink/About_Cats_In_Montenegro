"""
Module has some useful helper functions
to create directories, create a dataframe,
print a dataframe.
"""

# import basics
import os
from math import floor

# handling data
import pandas as pd

# import opencv
import cv2 as cv

# import torch
import torch

# pylint: disable=W0401
# pylint: disable=W0614
from .Definitions import *


def info(dataFrame: pd.DataFrame):
    """
    Print information about a dataframe.

    Parameters
    ----------
    dataFrame: pd.DataFrame
        The dataframe.
    """
    width: int = 79
    print("info start".center(width, "="))
    print()
    print("dataFrame.info()")
    dataFrame.info()
    print()
    print("dataFrame.head()")
    print(dataFrame.head())
    print()
    print("dataFrame.describe()")
    print(dataFrame.describe())
    print()
    print("info end".center(width, "="))


def printModelSize(model: torch.nn.Module):
    """
    Calc the size of the model according to
    https://discuss.pytorch.org/t/finding-model-size/130275

    Parameters
    ----------
    model : A model to train.
    """

    paramSize: float = 0.0
    bufferSize: float = 0.0

    for param in model.parameters():
        paramSize += param.nelement() * param.element_size()
    for buffer in model.buffers():
        bufferSize += buffer.nelement() * buffer.element_size()

    modelSize = (paramSize + bufferSize) / 1024**2
    print(f"model size: {modelSize:.3f}MB")


def printNumOfLearnableWeights(model: torch.nn.Module):
    """
    Print the number of trainable
    weights for a model.

    Parameters
    ----------
    model : A model to train.
    """
    trainableWeights = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable weights: {trainableWeights}")


def createDirectories(rootPath: str, folders: list[str]) -> bool:
    """
    For every folder in folders the following
    directory structure will be created.
    --rootPath
      |--folder
      |  |--cats
      |  |--dogs

    Parameters
    ----------
    rootPath : str
        The top level folder.
    folders : list[str]
        A list with folder names.

    Returns
    -------
    Bool
        False if at least on directory exists
        otherwise True.
    """
    # create a new directory with subfolders
    for subfolder1 in folders:
        for subfolder2 in ["cats", "dogs"]:
            path = f"{rootPath}/{subfolder1}/{subfolder2}"
            # create new path
            if os.path.exists(path):
                print(f"directory: {path} already exists")
                print("Stop creating any directory")
                return False

            os.makedirs(path)
            print(f"created directory: {path}")
    return True


def readDataToDataFrame(path: str) -> pd.DataFrame:
    """
    Create a DataFrame with a column for filename,
    label, width and height of the image.

    Parameters
    ----------
    path : str
        The path to the DataFrame

    Returns
    -------
    DataFrame
        The created DataFrame
    """
    fileNames: list[str] = []
    labels: list[int] = []
    width: list[int] = []
    height: list[int] = []
    for root, dirs, files in os.walk(path):
        # print(f"root: {root}")
        # print(f"dirs: {dirs}")
        # print(f"files: {files}")

        # print the last level of folder structure where the files are located
        if root and not dirs and files:
            print(f"Reading files from folder: {root}")
        for name in files:
            fileName = os.path.join(root, name)
            fileNames.append(fileName)

            if name.startswith("cat"):
                labels.append(1)
            elif name.startswith("dog"):
                labels.append(0)
            else:
                print("Error dog or cat found in path")

            # read the height, width of the image
            imShape: tuple[int, int] = cv.imread(fileName, cv.IMREAD_UNCHANGED).shape
            height.append(imShape[0])
            width.append(imShape[1])

    data = {
        "fileName": fileNames,
        "label": labels,
        "width": width,
        "height": height,
    }
    dataFrame = pd.DataFrame(
        {
            "fileName": pd.Series(dtype="str"),
            "label": pd.Series(dtype="bool"),
            "width": pd.Series(dtype="int"),
            "height": pd.Series(dtype="int"),
        }
    )
    dataFrame["fileName"].astype("str")
    dataFrame["label"].astype("bool")
    return pd.DataFrame(data)


def calcSizeOut(
    sizeIn: tuple[int, int],
    kernelSize: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> tuple[int, int]:
    """
    Calc the size of an image after convolution or max pooling.

    Parameters
    ----------
    sizeIn : tuple[int,int]
        The size of the image
    kernelSize: int
        The size of the kernel
    stride: int
        The stride
    padding: int
        The padding
    dilation: int
        The dilation

    Returns
    -------
    tuple[int,int]
        The ouput size of the image
    """
    sizeOut: list[int] = [0, 0]

    for index, _ in enumerate(sizeIn):
        sizeOut[index] = floor(
            (
                (sizeIn[index] + (2 * padding) - (dilation * (kernelSize - 1)) - 1)
                / stride
            )
            + 1
        )

    return sizeOut[0], sizeOut[1]
