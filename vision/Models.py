"""
Module contains a Convolutional Neural Net.
"""

# import torch
import torch
from torch import nn

# pylint: disable=W0401
# pylint: disable=W0614
from .Definitions import *

# custom modules
from .Utils import calcSizeOut

torch.manual_seed(0)


class CNN(torch.nn.Module):
    """
    A generic class to create deep convolutional neural network.

    A variable number of convolution layers can be defined.
    Each convolution layer consists of convolution,
    batch normalisation, relu activation and max pooling.

    After the last convolution layer the model will be flattened.

    A variable number of fully connected layers can be defined.
    Each fully connected layer consists of drop out and a batch
    normalisation.

    The activation function for the last fully connected layer is
    a Sigmoid function when the last fully connected layer has only 1
    neuron otherwise a Softmax function is used.

    """

    def __init__(
        self,
        imageSize: tuple[int, int],
        convLayers: list[dict[str, int]],
        fullConLayers: list[int],
        dropOut: float = 0.5,
        batchNormConv: bool = True,
        batchNormFullCon: bool = True,
    ):
        """

        Parameters
        ----------
        imageSize: tuple[int,int]
            The input size of the image
        convLayers : list[Dict]
            A list of dictionarys. A dictionary defines the number of
            input and output channels, the size for the kernel, the stride
            and the padding for a convolution layer e.g.

            conv1 = {"inChannels" : 3, "outChannels" : 4,
                     "cKernel" : 5, "cStride" : 1, "cPadding" : 2,
                     "mKernel" : 2, "mStride" : 2, "mPadding" : 0}

        fullConLayers : list[int]
            A list to define the number of neurons for the fully connected layers
        dropOut: float
            The drop out for all fully connected layers.
        batchNormConv: bool
            Flag to add a batch normalisation to all convolution layers.
        batchNormFullCon: bool
            Flag to add a batch normalisation to all fully connected layers.
        """

        super(CNN, self).__init__()
        # criterion will be set to BCE or CE according to the last output layer
        self.criterion = ""
        self.dropOut = dropOut
        self.batchNormConv = batchNormConv
        self.batchNormFullCon = batchNormFullCon
        self.imageSize = imageSize
        self.model = nn.Sequential()
        # default parameters for convolution and max pooling
        defaultParam: dict[str, int] = {
            "cKernel": 5,
            "cStride": 1,
            "cPadding": 2,
            "mKernel": 2,
            "mStride": 2,
            "mPadding": 0,
        }

        # create layers with convolution, maxpooling, batchnorm and activation
        sizeImageOutMaxPool: tuple[int, int] = (0, 0)
        sizeImageOutConv: tuple[int, int] = (0, 0)
        print("Creating a CNN with the following architecture:")
        for index, layer in zip(range(len(convLayers)), convLayers):
            # read the info about the layer from the dictionary
            inChannels: int = layer["inChannels"]
            outChannels: int = layer["outChannels"]

            # print some info about the actual layer
            print(f"\nConvolution Layer: {index}")
            print(f"Channels in: {inChannels}")
            print(f"Channels out: {outChannels}")

            # print(f"layer.keys(): {layer.keys()}")
            # print(f"list(layer.keys()): {list(layer.keys())}")

            cKernel: int = 0
            cStride: int = 0
            cPadding: int = 0
            mKernel: int = 0
            mStride: int = 0
            mPadding: int = 0

            # read all default params for convolution if one parameter is not defined
            for param in ["cKernel", "cStride", "cPadding"]:
                if param in list(layer.keys()):
                    cKernel = layer["cKernel"]
                    cStride = layer["cStride"]
                    cPadding = layer["cPadding"]
                else:
                    print("Reading default parameters for convolution")
                    cKernel = defaultParam["cKernel"]
                    cStride = defaultParam["cStride"]
                    cPadding = defaultParam["cPadding"]
                    break

            # read all default params for max pooling if one parameter is not defined
            for param in ["mKernel", "mStride", "mPadding"]:
                if param in list(layer.keys()):
                    mKernel = layer["mKernel"]
                    mStride = layer["mStride"]
                    mPadding = layer["mPadding"]
                else:
                    print("Reading default parameters for max pooling")
                    mKernel = defaultParam["mKernel"]
                    mStride = defaultParam["mStride"]
                    mPadding = defaultParam["mPadding"]
                    break

            self.model.append(
                nn.Conv2d(
                    in_channels=inChannels,
                    out_channels=outChannels,
                    kernel_size=cKernel,
                    stride=cStride,
                    padding=cPadding,
                )
            )
            # append batch normalisation after the convolution layer
            if self.batchNormConv is True:
                self.model.append(nn.BatchNorm2d(outChannels))

            # append relu
            self.model.append(nn.ReLU())

            # append max pooling
            self.model.append(
                nn.MaxPool2d(
                    kernel_size=mKernel,
                    stride=mStride,
                    padding=mPadding,
                )
            )

            # calc the output image size after convolution
            if index == 0:
                # special handling to calc the input size of the image after convolution
                # needed for the first convolution layer
                sizeImage = self.imageSize
            else:
                sizeImage = sizeImageOutMaxPool

            # calc size after convolution
            sizeImageOutConv = calcSizeOut(
                sizeImage, kernelSize=cKernel, stride=cStride, padding=cPadding
            )
            print(f"Size Image out Conv: {sizeImageOutConv}")

            # calc size after max pooling
            sizeImageOutMaxPool = calcSizeOut(
                sizeImageOutConv,
                kernelSize=mKernel,
                stride=mStride,
                padding=mPadding,
            )
            print(f"Size Image out MaxPool: {sizeImageOutMaxPool}")

        # flatten to add fully connected layers
        self.model.append(nn.Flatten(1, -1))

        # create fully connected layers
        neuronsPerLayer: list[int] = []
        for fullLayer, numNeurons in zip(range(len(fullConLayers)), fullConLayers):
            print(f"\nFully connected layer: {fullLayer}")
            print(f"Number of Neurons: {numNeurons}")
            neuronsPerLayer.append(numNeurons)

            # calc the number of input features
            inFeatures: int = 0
            if fullLayer == 0:
                # special handling to calc the number of input features needed
                # for the first full connected layer
                inFeatures = (
                    outChannels * sizeImageOutMaxPool[0] * sizeImageOutMaxPool[1]
                )
            else:
                inFeatures = neuronsPerLayer[fullLayer - 1]

            self.model.append(nn.Linear(inFeatures, numNeurons))
            # initialise the weights for the last added layer
            torch.nn.init.xavier_uniform_(self.model[-1].weight)
            self.model.append(nn.Dropout(self.dropOut))
            if self.batchNormFullCon is True:
                self.model.append(nn.BatchNorm1d(numNeurons))

        if fullConLayers[-1] == 1:
            # Use Sigmoid to use BinaryCrossEntropyLoss() when the
            # last fully connected layer has only one neuron
            self.model.append(nn.Sigmoid())
            self.criterion = "BCE"
            print("\nSetting criterion to BinaryCrossEntropyLoss!")
        else:
            self.model.append(nn.Softmax(dim=1))
            self.criterion = "CE"
            print("\nSetting criterion to CrossEntropyLoss!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create the output of the neural net
        for a given input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output of the neural net.
        """
        # if you want to know the shape of x for every
        # layer go to the implementation of
        # nn.Sequential() and add
        # print(f"shape.input: {input.shape}")
        # in the forward() method
        return self.model(x)
