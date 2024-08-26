# Copyright © 2024, whatwouldaithink.com.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Module provide plotting functions."""

import numpy as np

# for plotting and colormaps
import matplotlib.pyplot as plt

from vision.Definitions import tab10Colors, fontSizesPlot, figSize


class PiePlot:
    """
    Class for plotting a pie chart.
    """

    def __init__(self, title: str, path: str) -> None:
        """
        Parameters
        ----------
        title: str
            The title of the pie plot.
        path: str
            The path to save the plot.
        """
        self.title: str = title
        self.path: str = path
        self.colorMap: list[tuple[float, float, float]] = [
            tab10Colors["tab:olive"],
            tab10Colors["tab:cyan"],
        ]

    def plot(self, data: tuple[int, int], labels: list[str] = ["Dog", "Cat"]) -> None:
        """
        Plot and save a bargraph.
        Parameters
        ----------
        data: tuple[int, int]
            The data to plot
        labels: list[str]
            The labels
        """

        assert len(data) == len(labels), "Number of data and labels are not equal!"

        _, ax = plt.subplots(figsize=figSize, subplot_kw={"aspect": "equal"})

        # plt.xticks(fontsize=fontSizesPlot["xy_ticks"])
        # plt.yticks(fontsize=fontSizesPlot["xy_ticks"])
        ax.pie(
            x=data,
            labels=labels,
            autopct=lambda pct: f"{pct:.1f}%",
            colors=self.colorMap,
            # textprops=dict(color="black", size=fontSizesPlot["xy_ticks"] + 2),
            textprops={"color": "black", "size": fontSizesPlot["xy_ticks"] + 2},
            startangle=45,
            counterclock=True,
            radius=1.2,
            pctdistance=0.5,
            labeldistance=1.1,
        )
        plt.title(self.title, fontdict={"fontsize": fontSizesPlot["title"] - 2})
        # plt.tight_layout()
        plt.savefig(f"{self.path}.svg", dpi="figure", format="svg")
        plt.show()


class Plot2D:
    """
    Class for plotting data in a x-y diagram.
    """

    def __init__(
        self,
        xMinMax: tuple[int, int],
        yMinMax: tuple[int, int],
        title: str,
    ):
        """
        Parameters
        ----------
        xMinMax: tuple[int, int]
            The limits for the x-axis.
        yMinMax: tuple[int, int]
            The limits for the y-axis.
        accuracy: float
            The accuracy of the test data.
        title: str
            The title for the diagram.
        """

        self.title = title
        self.xMinMax = xMinMax
        self.yMinMax = yMinMax
        self.figSize = (5, 5)
        self.colorMap: list[str] = [
            "tab:cyan",
            "tab:pink",
            "tab:olive",
            "tab:blue",
            "tab:gray",
            "tab:brown",
            "tab:red",
            "tab:purple",
            "tab:orange",
            "tab:green",
        ]

    def plotLearningProcess(self, dictAccuracies: dict[str, list[float]]):
        """
        Plot the learning curve stored in an dictionary.

        Parameters
        ----------
        dictAccuracies: dict[str,list[float]]
        """

        plt.figure(figsize=self.figSize)
        ax = plt.axes()

        for i, (key, listAccuracy) in enumerate(dictAccuracies.items()):
            # generate the values for the x-axis
            if i == 0:
                x = np.arange(1, len(listAccuracy) + 1, 1)
            # plot the accuracy for train and test set
            if i <= 1:
                ax.plot(
                    x,
                    np.array(listAccuracy) * 100,
                    linestyle="-",
                    color=self.colorMap[i],
                    label=f"{key}",
                )
            # plot the accuracy for validation sets
            else:
                ax.plot(
                    x,
                    np.array(listAccuracy) * 100,
                    linestyle="-",
                    alpha=0.95,
                    color=self.colorMap[i],
                    label=f"{key}",
                )

            # add a dot for the maximum accuracy of every set
            indexMax: np.intp = np.argmax(listAccuracy)
            # indexMax+1 because the list starts at index 0 but the plot at index 1
            ax.plot(
                indexMax + 1,
                np.array(listAccuracy[indexMax]) * 100,
                "o",
                color=self.colorMap[i],
            )

            # add a vertical line at the position of the maximum accuracy for the test set
            if key == "test":
                indexMax: np.intp = np.argmax(dictAccuracies["test"])
                ax.vlines(
                    x=indexMax + 1,
                    ymin=0,
                    ymax=100,
                    colors=tab10Colors["tab:gray"],
                    linestyle="--",
                    alpha=0.85,
                )

        # set xticks depending on the max number of epochs
        if self.xMinMax[1] <= 10:
            plt.xticks(
                ticks=np.arange(1, self.xMinMax[1] + 1, 1),
                fontsize=fontSizesPlot["xy_ticks"],
            )
        else:
            plt.xticks(
                ticks=[1, 10, 20, 30, 40, 50], fontsize=fontSizesPlot["xy_ticks"]
            )
        plt.yticks(
            ticks=[0, 20, 40, 60, 80, 90, 95, 100], fontsize=fontSizesPlot["xy_ticks"]
        )
        plt.xlim(self.xMinMax[0], self.xMinMax[1] + 1)
        plt.ylim(self.yMinMax[0], self.yMinMax[1])
        plt.xlabel("Epoch", fontdict={"size": fontSizesPlot["xy_labels"]})
        plt.ylabel("Accuracy in %", fontdict={"size": fontSizesPlot["xy_labels"]})
        plt.title(self.title, fontdict={"size": fontSizesPlot["title"]})
        plt.grid(visible=True, which="major", axis="y")
        plt.grid(visible=True, which="major", axis="x")
        plt.legend(
            loc="lower center",
            ncols=2,
            bbox_to_anchor=(0.5, -0.35),
            framealpha=0,
            fontsize=fontSizesPlot["legend"],
        )
        plt.text(
            self.xMinMax[1] * 0.3,
            25,
            "WhatWouldAiThink.com",
            fontsize=fontSizesPlot["text"],
            rotation=45,
            color="grey",
        ).set_zorder(1000)

        path: str = f"{self.title}.svg"
        # remove whitespace and linebreak for filenames
        plt.savefig(
            path.replace(" ", "").replace("\n", ""),
            bbox_inches="tight",  # ensure that the whole legend is visible
            pad_inches=0.1,  # ensure that the whole legend is visible
            dpi="figure",
            format="svg",
        )
        plt.show()

    def plotImageSize(self, width: int, height: int):
        """
        Plot the size of an image.

        Parameters
        ----------
        width: int
            The width of the image.
        height: int
            The height of the image.
        """

        plt.figure(figsize=self.figSize)
        ax = plt.axes()
        ax.plot(width, height, "o")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        ax.set_xlim(self.xMinMax[0], self.xMinMax[1])
        ax.set_ylim(self.yMinMax[0], self.yMinMax[1])
        plt.title(self.title, fontdict={"size": fontSizesPlot["title"]})
        plt.text(
            600,
            100,
            "WhatWouldAiThink.com",
            fontsize=10,
            rotation=45,
            color="grey",
        ).set_zorder(1000)
        plt.show()
