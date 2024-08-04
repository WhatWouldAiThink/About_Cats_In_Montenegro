"""
Module for common used variables
"""

# for type hints
from typing import Any

# create a colormap for 2 elements
cMap: list[str] = ["green", "red"]
cDic: dict[int, str] = {0: cMap[0], 1: cMap[1]}
labelDic: dict[int, str] = {0: "Dog", 1: "Cat"}

figSize: tuple[int, int] = (5, 5)

MIN_SIZE_IMAGE: tuple[int, int] = (224, 224)

fontSizesPlot: dict[str, int] = {}
fontSizesPlot["title"] = 16
fontSizesPlot["legend"] = 13
fontSizesPlot["xy_labels"] = 15
fontSizesPlot["xy_ticks"] = 12
fontSizesPlot["text"] = 14

# define alias for unkown torch types
torchDataSet = Any
torchDataLoader = Any
torchVisionModels = Any
torchVisionWeights = Any

# added rgb defintion for tab10 colors to be more flexible
_tab10Data = (
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),  # 1f77b4
    (1.0, 0.4980392156862745, 0.054901960784313725),  # ff7f0e
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),  # 2ca02c
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # d62728
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),  # 9467bd
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),  # 8c564b
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),  # e377c2
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),  # 7f7f7f
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),  # bcbd22
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),  # 17becf
)

_tab10Names = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
)

tab10Colors: dict[str, tuple[float, float, float]] = {}

for colorName, colorData in zip(_tab10Names, _tab10Data):
    tab10Colors[colorName] = colorData
