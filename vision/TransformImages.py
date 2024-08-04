"""Module contains a class for image transformations"""

# for type hints
from typing import Generator

# import basics
import os
from pathlib import Path

# import opencv
import cv2 as cv

# handling data
import numpy as np

# import torch
import torch
from torchvision import transforms
from torchvision.io import read_image, write_jpeg

torch.manual_seed(0)

# pylint: disable=W0401
# pylint: disable=W0614
from .Definitions import *


class TransformImages:
    """
    Class for transforming e.g. rotate, resize
    the images in a folder and saving the
    transformed images to a new folder.
    """

    def __init__(
        self,
        rootPathIn: str,
        rootPathOut: str,
        resize: bool = False,
        size: tuple[int, int] = (0, 0),
    ):
        """
        Parameters
        ----------
        rootPathIn: str
            The path of the folder with images to transform
        rootPathOut: str
            The path of the folder for the transformed images
        resize: bool
            If True all images will be resized to size after a transformation.
        size: tuple(int,int)
            The new size after a transformation.
        """
        self.rootPathIn = rootPathIn
        self.rootPathOut = rootPathOut
        self.postProcessing = self.ForwardOrResizeImage(resize, size)
        # len of input and output path must be equal to transform and
        # save a image from e.g
        # rootIn/folder1In/f../image1.jpg to
        # rootOut/folder1Out/../image1.jpg
        assert len(Path(self.rootPathIn).parts) == len(Path(self.rootPathOut).parts)
        # print(f"self.rootPathIn: {self.rootPathIn}")
        # print(f"self.rootPathOut: {self.rootPathOut}")

    class ForwardOrResizeImage:
        """
        A helper Class to be used as a Functor for
        postprocessing a transformed image.
        If resize is False the image will be only
        forwarded.
        """

        def __init__(self, resize: bool = False, size: tuple[int, int] = (0, 0)):
            """
            Parameters
            ----------
            resize: bool
                If True all images will be resized to size after a transformation
            size: tuple(int,int)
                The new size after a transformation
            """
            self.resize = resize
            self.size = size
            if self.resize is True:
                assert (self.size[0] > 0) and (self.size[1] > 0)

        def __call__(self, im: torch.Tensor) -> torch.Tensor:
            """
            Call to resize or forward a image.
            If the image is smaller than self.size
            the image will be upscaled!

            Parameters
            ----------
            im: torch.Tensor
                The image to be resized or forwarded

            Returns
            -------
            torch.Tensor
                The resized or forwarded image
            """
            if self.resize is True:
                return transforms.functional.resize(
                    im, self.size, transforms.InterpolationMode.BILINEAR
                )

            return im

    def getFilePaths(
        self, transformType: str = ""
    ) -> Generator[tuple[str, str], None, None]:
        """
        Helper method to get the path of all images
        in a folder defined by self.rootPathIn.
        Creates the output path for all images
        and return the input path and output path.
        This method is used by all methods transforming
        an image.

        Parameters
        ----------
        im: torch.Tensor
            The image to be resized or forwarded
        """
        for root, _, files in os.walk(self.rootPathIn):
            for name in files:
                filePathIn: str = os.path.join(root, name)

                # split into the path and the extension
                # because we want to modify the file name
                # _filePathIn is a tuple e.g. ('dataset', 'test_set', 'dogs', 'dog.4155')
                _filePathIn, fileExtension = os.path.splitext(filePathIn)

                # replace the root name and add the type of transformation e.g.
                # filePathIn: dataset/test_set/dogs/dog.4155.jpg
                # filePathOut: dataset50x50/test_set/dogs/dog.4155_transform.jpg

                # calc the levels of the rootPathIn e.g. dataset/test_set -> 2
                # to skip these levels when contructing the filePathOut
                skipRootPathIn = len(Path(self.rootPathIn).parts)

                # loop over the input path except the file name
                pathOut = self.rootPathOut
                for folder in Path(_filePathIn).parts[skipRootPathIn:-1]:
                    pathOut = os.path.join(pathOut, folder)

                # add the type of transformation and the extension to the file name
                filePathOut: str = (
                    Path(_filePathIn).parts[-1] + "_" + transformType + fileExtension
                )
                filePathOut = os.path.join(pathOut, filePathOut)

                # print(f"Generator filePathIn: {filePathIn}")
                # print(f"Generator filePathOut: {filePathOut}")

                yield filePathIn, filePathOut

    def calcEdges(self, channels: int = 3):
        """
        Compute the edges of the images.

        Parameters
        ----------
        channels: int
            The number of channels for the output image.
        """
        for fileNameIn, fileNameOut in self.getFilePaths(transformType="ED"):
            # print(f"{fileNameIn} : {fileNameOut}")
            # read image and convert to 1 channel gray image
            im: torch.Tensor = transforms.functional.rgb_to_grayscale(
                read_image(fileNameIn), 1
            )
            c, w, h = im.shape

            # convert to numpy and change dimension from [c, w, h] to [w, h, c]
            imageGray = np.empty(shape=(w, h, c), dtype=int)
            imageGray = im.numpy().reshape((w, h, c))

            lowerThreshold: int = 150
            upperThreshold: int = 200

            # Apply canny edge detector
            imageEdges = cv.Canny(
                imageGray, lowerThreshold, upperThreshold, apertureSize=3
            )

            # Clone channel 1 to the 3 RGB channels
            imageEdgesOut = np.zeros((w, h, channels))
            for channel in np.arange(channels):
                imageEdgesOut[:, :, channel] = imageEdges

            # change dimension from [w, h, c] to [c, w, h] and convert to tensor
            write_jpeg(
                self.postProcessing(
                    torch.from_numpy(imageEdgesOut)
                    .to(dtype=torch.uint8)
                    .permute(2, 0, 1)
                ),
                fileNameOut,
            )

    def equalize(self):
        """
        Equalize the histogram of an image in order to create
        a uniform distribution of grayscale values in the output.
        """
        for fileNameIn, fileNameOut in self.getFilePaths(transformType="EQ"):
            write_jpeg(
                self.postProcessing(
                    transforms.functional.equalize(read_image(fileNameIn))
                ),
                fileNameOut,
            )

    def downScale(self, step: int = 2):
        """
        Downscale the images by taken only
        every step row and column from the image.

        Parameters
        ----------
        step: int
            Take only every step row and column of the image,
            step=2 means, take the first,third,fifth...
            row and column.
        """
        for fileNameIn, fileNameOut in self.getFilePaths(transformType=""):
            # print(f"downScale shape before: {read_image(fileNameIn).shape}")
            # im = read_image(fileNameIn)[:,::drop,::drop]
            # print(f"downScale shape after: {im.shape}")
            write_jpeg(
                self.postProcessing(read_image(fileNameIn)[:, ::step, ::step]),
                fileNameOut,
            )

    def noTransform(self):
        """
        Move the images from fileNameIn
        to fileNameOut.
        """
        for fileNameIn, fileNameOut in self.getFilePaths(transformType=""):
            write_jpeg(read_image(fileNameIn), fileNameOut)

    def pad(self, mode: str = "S"):
        """
        Open all images from fileNameIn,
        add a padding, do post processing
        and writes the new images to fileNameOut.

        Parameters
        ----------
        mode: str
            The mode of padding, S for small
            M for Medium and B for Big.
        """

        # PD means PAD
        for fileNameIn, fileNameOut in self.getFilePaths(transformType=f"PD_{mode}"):
            im: torch.Tensor = read_image(fileNameIn)
            _, h, w = im.shape
            # define types of padding, medium and big padding
            # resize the image e.g. and a padding so the image size
            # will not change, this leads to a zoom out effect
            padding: tuple[int, int]
            size: tuple[int, int]
            if mode == "B":
                # Big resize
                size = (int(h / 3), int(w / 3))
                padding = (int(w / 3), int(h / 3))
            elif mode == "M":
                # Medium resize
                size = (int(h / 2), int(w / 2))
                padding = (int(w / 4), int(h / 4))
            elif mode == "S":
                # Small resize
                size = (int(6 * h / 10), int(6 * w / 10))
                padding = (int(2 * w / 10), int(2 * h / 10))
            else:
                print(f"No mode definied for {mode}")
                assert False

            im = transforms.functional.resize(
                im, size, transforms.InterpolationMode.BILINEAR
            )
            im = transforms.functional.pad(im, padding=padding, padding_mode="constant")
            write_jpeg(self.postProcessing(im), fileNameOut)

    def toGrayScale(self, channels: int = 3):
        """
        Open all images from fileNameIn,
        convert all channels to gray,
        do post processing
        and writes the new images to fileNameOut.

        Parameters
        ----------
        channels: int
            The number of channels for the output image.
        """
        # GS means GRAYSCALE
        for fileNameIn, fileNameOut in self.getFilePaths(transformType="GS"):
            im: torch.Tensor = read_image(fileNameIn)
            write_jpeg(
                self.postProcessing(
                    transforms.functional.rgb_to_grayscale(im, channels)
                ),
                fileNameOut,
            )

    def invert(self):
        """
        Open all images from fileNameIn,
        invert the colors of all channels,
        do post processing
        and writes the new images to fileNameOut.
        """
        # IN means INVERT
        for fileNameIn, fileNameOut in self.getFilePaths(transformType="IN"):
            im: torch.Tensor = read_image(fileNameIn)
            write_jpeg(
                self.postProcessing(transforms.functional.invert(im)),
                fileNameOut,
            )

    def colorJitter(self, mode: str = "CJ1"):
        """
        Open all images from fileNameIn,
        change saturation or hue,
        do post processing
        and writes the new images to fileNameOut.

        Parameters
        ----------
        mode: str
            The mode of collor jitter,
            CJ1 for changing saturation,
            CJ2 for channging hue.
        """
        # parameters for the original image
        # brightness=1
        # contrast=1
        # saturation=1
        # hue=0

        # CJ means ColorJitter
        for fileNameIn, fileNameOut in self.getFilePaths(transformType=f"{mode}"):
            im: torch.Tensor = read_image(fileNameIn)
            # define types of color jitter
            if mode == "CJ1":
                im = transforms.functional.adjust_saturation(im, 3)
            elif mode == "CJ2":
                im = transforms.functional.adjust_hue(im, 0.5)
            else:
                print(f"No mode definied for {mode}")
                assert False

            write_jpeg(self.postProcessing(im), fileNameOut)

    def posterize(self, bits: int = 8):
        """
        Open all images from fileNameIn,
        apply posterize transform,
        do post processing
        and writes the new images to fileNameOut.

        Parameters
        ----------
        bits: int
            The number of bits to keep for each channel.
        """
        # PO means POSTERIZE
        for fileNameIn, fileNameOut in self.getFilePaths(transformType=f"PO_{bits}"):
            im: torch.Tensor = read_image(fileNameIn)
            write_jpeg(
                self.postProcessing(transforms.functional.posterize(im, bits)),
                fileNameOut,
            )

    def perspective(self, mode: str = "P1"):
        """
        Open all images from fileNameIn,
        apply a perspective transform,
        do post processing
        and writes the new images to fileNameOut.

        Parameters
        ----------
        mode: str
            The mode of four (P1,..,P4) predefined
            perspective transformation.
        """

        # PE means PERSPECTIVE
        for fileNameIn, fileNameOut in self.getFilePaths(transformType=f"PE_{mode}"):
            im: torch.Tensor = read_image(fileNameIn)
            _, h, w = im.shape

            # a point defines [top-left, top-right, bottom-right, bottom-left]
            points: tuple[Any, Any]
            # define types of perspectives
            if mode == "P1":
                # right side is shifted to the back
                points = (
                    [[0, 0], [w, 0], [w, h], [0, h]],
                    [
                        [w * 0.11, 0],
                        [w, h * 0.2],
                        [w * 0.87, h * 0.95],
                        [w * 0.12, h],
                    ],
                )

            elif mode == "P2":
                # left side is shifted to the back
                points = (
                    [
                        [0, 0],
                        [w * 1.27, 0],
                        [w * 1.27, h * 1.3],
                        [0, h * 1.27],
                    ],
                    [
                        [w * 0.04, h * 0.29],
                        [w * 1.27, h * 0.3],
                        [w * 1.05, h * 1.2],
                        [w * 0.12, h * 0.97],
                    ],
                )

            elif mode == "P3":
                # top side is shifted to the back
                points = (
                    [
                        [0, 0],
                        [w * 1.27, 0],
                        [w * 1.27, h * 1.27],
                        [0, h * 1.27],
                    ],
                    [
                        [w * 0.3, h * 0.16],
                        [w * 1.03, h * 0.2],
                        [w * 1.2, h * 1.25],
                        [w * 0.06, h * 1.23],
                    ],
                )

            elif mode == "P4":
                # bottom side is shifted to the back
                points = (
                    [[0, 0], [w, 0], [w, h], [0, h]],
                    [
                        [w * 0.1, h * 0.1],
                        [w, h * 0.16],
                        [w * 0.76, h * 0.85],
                        [w * 0.2, h * 0.76],
                    ],
                )

            else:
                print(f"No mode definied for {mode}")
                assert False

            write_jpeg(
                self.postProcessing(
                    transforms.functional.perspective(im, points[0], points[1])
                ),
                fileNameOut,
            )

    def rotate(self, angle: int):
        """
        Open all images from fileNameIn,
        rotate, do post processing
        and writes the new images to fileNameOut.

        Parameters
        ----------
        angle: int
            The angle to rotate a image clockwise.
        """
        # RO means ROTATE
        for fileNameIn, fileNameOut in self.getFilePaths(transformType=f"RO_{angle}"):
            im: torch.Tensor = read_image(fileNameIn)
            # shift the angle by 360 to have clockwise rotation
            write_jpeg(
                self.postProcessing(transforms.functional.rotate(im, 360 - angle)),
                fileNameOut,
            )

    def centerCrop(self):
        """
        Open all images from fileNameIn,
        centerCrop, do post processing
        and writes the new images to fileNameOut.
        """
        # CC means Center Crop
        for fileNameIn, fileNameOut in self.getFilePaths(transformType="CC"):
            im: torch.Tensor = read_image(fileNameIn)
            _, h, w = im.shape
            # crop the image in the center with a crop box size of (h/2, w/2)
            write_jpeg(
                self.postProcessing(
                    transforms.functional.center_crop(im, (int(h / 2), int(w / 2)))
                ),
                fileNameOut,
            )

    def resize(self):
        """
        Open all images from fileNameIn,
        do post processing
        and writes the new images to fileNameOut.
        This method works only when
        self.resize is True and self.size is set.
        """
        resize = self.postProcessing.resize
        size = self.postProcessing.size
        assert resize is True

        # RE means Resize
        for fileNameIn, fileNameOut in self.getFilePaths(transformType="RE"):
            im: torch.Tensor = read_image(fileNameIn)
            _, h, w = im.shape

            # only down size images
            if h >= size[0] and w >= size[1]:
                write_jpeg(self.postProcessing(im), fileNameOut)
