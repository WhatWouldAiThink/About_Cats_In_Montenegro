# Copyright © 2024, whatwouldaithink.com.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Module provides a Trainingmanager to resume training"""

# import basics
from time import time, ctime, sleep
from pathlib import Path

# handling data
import numpy as np

np.random.seed(0)

# import torch
import torch
from torch import nn
from torch.utils.data import DataLoader

# for plotting and colormaps
import matplotlib.pyplot as plt

# custom modules
from plotting.customPlotting import Plot2D as cPlot2D

# pylint: disable=W0401
# pylint: disable=W0614
from .Definitions import *

torch.manual_seed(0)


class TrainingManager:
    """
    A class to handle common tasks when training
    neural networks.
    The TrainingManager can save and load checkpoints
    to resume training and plot the training and
    test accuracy.
    """

    def __init__(
        self,
        lr: float,
        model: torch.nn.Module,
        nameCheckpoint: str,
        batchSize: int,
        criterion: str,
        datasets: dict[str, torchDataSet],
        title: str = "No title",
    ):
        """
        Parameters
        ----------
        lr : float
            The learning rate.
        model: torch.nn.Module
            The neural network to train.
        nameCheckpoint: str
            The name of the checkpoint to store.
        batchSize: int
            The batch size.
        criterion: str
            The criterion used to calculate the loss.
        datasets: dict[str,torchDataSet]
            A dictionary with datasets used for train, test, validation.
            The accuracy for the datasets are stored in the checkpoint.
            A least datasets for train and test must be provided and
            named 'train' and 'test' e.g. datasets = {"train": Data(), "test", Data()}
            Unlimited number of datasets for validation can be provided.
        title: str
            The title for plotting the learning curve.
        """
        self.datasets = datasets
        assert (
            len(self.datasets) >= 2
        ), "At least a dataset for train and test must be defined"
        assert (
            "train" in self.datasets.keys()
        ), "At dataset for train named 'train' must be defined"
        assert (
            "test" in self.datasets.keys()
        ), "At dataset for train named 'test' must be defined"
        self.nEpochs: int | None = None
        self.learningRate: float = lr
        self.model: torch.nn.Module = model
        self.nameCheckpoint: str = nameCheckpoint
        self.batchSize: int = batchSize
        self.title: str = title
        self.numWorkers: int = 8
        self.sleepFactorBatch: float = 0  # 0.20 -> 5%
        self.sleepTimePeriodic: int = 30  # 30s
        self.sleepIntervalPeriodic: int = (
            5 * 60
        )  # sleep every 5 minutes for self.sleepTimePeriodic
        self.factorProcessedBatches: int = 1
        self.maxAccuracy: float = 0.0
        self.criterion: str = criterion
        # two critera are defined
        assert self.criterion in [
            "BCE",
            "CE",
        ], f"No operation definied for criterion {self.criterion}"
        self.keysDataSets: list[str] = []
        self.dicDataLoaders: dict[str, torchDataLoader] = {}
        self.dicNumSamples: dict[str, int] = {}
        self.dicAccuracyList: dict[str, list[float]] = {}
        self.checkpoint: dict[str, Any] = {
            "epoch": int,
            "model_state_dict": dict[Any, Any],
            "optimizer_state_dict": dict[Any, Any],
        }
        self.checkpointBest: Any = {
            "accuracy": float,
            "model_state_dict": dict[Any, Any],
            "optimizer_state_dict": dict[Any, Any],
        }
        self.checkpointPath: str = ""
        self.checkpointBestPath: str = ""

        for nameDataSet, dataSet in datasets.items():
            # get the keys, e.g. train, test, validate
            self.keysDataSets.append(f"{nameDataSet}")
            # create dic for accuracy "{"train":[], "test":[]}
            self.dicAccuracyList[f"{nameDataSet}"] = []
            # create a dic for the checkpoint {accuracy_lists: {"train":[], "test":[]}
            self.checkpoint["accuracy_lists"] = {f"{nameDataSet}": []}
            # number of samples
            self.dicNumSamples[f"{nameDataSet}"] = len(dataSet)
            print(f"Number of samples for {nameDataSet}: {len(dataSet)}")
            # create the trainloaders
            self.dicDataLoaders[f"{nameDataSet}"] = DataLoader(
                dataset=dataSet,
                batch_size=self.batchSize,
                num_workers=self.numWorkers,
                shuffle=True,
            )

        # init a the checkpoint
        self.createCheckPoints()
        # create optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learningRate)
        """ 
        Todo. change to this optimizer, usefull for pretrained models
        self.optimizer = torch.optim.Adam(
            [
                parameters
                for parameters in model.parameters()
                if parameters.requires_grad
            ],
            lr=self.learningRate,
        )
        """

    def loadCheckPoint(self, path: str) -> tuple[int, int]:
        """
        Load the last checkpoint to resume training.

        Parameters
        ----------
        path : str
            The path to the checkpoint.
        Returns
        -------
        tuple[int,int]
            The start and end epoch for training.
        """
        print(f"\nLoading checkpoint: {path}")
        checkpoint: Any = torch.load(path)

        print(f"checkpoint[epoch]: {checkpoint['epoch']}")

        # load the parameters from the checkpoint to resume training
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print("\nLoading checkpoint['accuracy_lists']")
        for nameDataSet, accuracyList in checkpoint["accuracy_lists"].items():
            print(f"{nameDataSet} : {accuracyList}")
            self.dicAccuracyList[f"{nameDataSet}"] = accuracyList
            print(f"len = {len(self.dicAccuracyList[f'{nameDataSet}'])}")

        epochStart: int = checkpoint["epoch"] + 1
        epochEnd: int = checkpoint["epoch"] + self.nEpochs
        return epochStart, epochEnd

    def loadBestCheckPoint(self):
        """
        Load the best checkpoint.
        """

        print(f"\nLoading checkpoint: {self.checkpointBestPath}")
        checkpointBest: Any = torch.load(self.checkpointBestPath)

        self.maxAccuracy = checkpointBest["accuracy"]
        print(f"max_accuracy is: {self.maxAccuracy}")
        self.checkpointBest = checkpointBest

    def createCheckPoints(self):
        """
        Create two checkpoint.
        One to save the checkpoint for the best
        accuracy, to deploy the model.
        One to save the last checkpoint to resume
        training.
        """
        self.checkpointPath = f"checkpoint_{self.nameCheckpoint}.pt"
        self.checkpointBestPath = f"checkpoint_{self.nameCheckpoint}_best.pt"

        if (
            Path(self.checkpointPath).is_file()
            or Path(self.checkpointBestPath).is_file()
        ):
            print(
                f"\nFile {self.checkpointPath} or {self.checkpointBestPath} already exist."
            )
            print("ONLY RESUME TRAINING POSSIBLE!!!.\n")
        else:
            print(f"Init new checkpoint: {self.checkpointPath}")
            print(f"Init new checkpointBest: {self.checkpointBestPath}")

    def resume(self, nEpochs: int):
        """
        Resume training with a loaded checkpoint.

        Parameters
        ----------
        nEpochs : int
            The number of epochs for resume training.
        """
        self.nEpochs = nEpochs
        epochStart, epochEnd = self.loadCheckPoint(self.checkpointPath)
        self.loadBestCheckPoint()

        # plot the learning curve at the begining when you resume the training
        self.plotLearningProgress(
            epoch=epochStart - 1,
            epochs=epochStart - 1,
            dicAccuracyList=self.dicAccuracyList,
            title=self.title,
            forcePlot=True,
        )

        # return after the checkpoint (model weights and optimizer paramters) is loaded
        if self.nEpochs <= 0:
            return
        self.trainResumeLoop(epochStart, epochEnd)

    def train(self, nEpochs: int):
        """
        Start a training from scratch.
        A new checkpoint will be created and
        saved after the first epoch.

        Parameters
        ----------
        nEpochs : int
            The number of epochs for resume training.
        """
        self.nEpochs = nEpochs
        assert Path(self.checkpointPath).is_file() is False
        assert Path(self.checkpointBestPath).is_file() is False
        # return after the checkpoint (model weights and optimizer paramters) is loaded
        if self.nEpochs <= 0:
            return
        self.trainResumeLoop(1, self.nEpochs)

    def trainResumeLoop(self, epochStart: int, epochEnd: int):
        """
        Starts the training loop.
        Calculates the training and test accuracy.
        Save checkpoints and plot the accuracy.

        Parameters
        ----------
        epochStart : int
            The start epoch
        epochEnd: int
            The end epoch
        """

        # set learning rate
        for paramGroups in self.optimizer.param_groups:
            paramGroups["lr"] = self.learningRate
            print(f"Start learning with lr = {paramGroups['lr']}")

        for epoch in np.arange(epochStart, epochEnd + 1, 1):
            tic = time()
            print(
                f"\nRunning Epoch: {epoch} from {epochEnd} with lr = {self.optimizer.state_dict()['param_groups'][0]['lr']}"
            )

            accuracyTest: float = 0
            for nameDataSet, dataLoader in self.dicDataLoaders.items():
                if nameDataSet == "train":
                    _, _ = self.__train__(epoch, nameDataSet, dataLoader)
                elif nameDataSet == "test":
                    _, accuracyTest = self.__eval__(epoch, nameDataSet, dataLoader)
                else:
                    _ = self.__eval__(epoch, nameDataSet, dataLoader)

            # save every epoch to resume training
            self.saveCheckpoint(epoch)

            # save only when accuracy increase to have the best trained model
            if epoch == 1 or ((epoch > 1) and (accuracyTest > self.maxAccuracy)):
                self.maxAccuracy = accuracyTest
                self.saveBestCheckpoint(epoch, accuracyTest)

            # plot the progress
            self.plotLearningProgress(
                epoch, epochEnd, self.dicAccuracyList, self.nameCheckpoint
            )

            usedTime = time() - tic
            print(f"Used time for one epoch is {usedTime:.2f} sec")

    def saveCheckpoint(self, epoch: int):
        """
        Write the checkpoint for the last epoch to
        the file self.checkpoint.

        Parameters
        ----------
        epochs : int
            The number of epochs the last epoch.
        """
        # save every epoch to resume training
        self.checkpoint["epoch"] = epoch
        self.checkpoint["model_state_dict"] = self.model.state_dict()
        self.checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

        # write the accuracy list for all dataset to the checkpoint
        for nameDataSet in self.keysDataSets:
            self.checkpoint["accuracy_lists"][f"{nameDataSet}"] = self.dicAccuracyList[
                f"{nameDataSet}"
            ]

        # check that number of epochs and len of accuracy is equal
        for nameDataSet in self.keysDataSets:
            assert epoch == len(self.dicAccuracyList[f"{nameDataSet}"])

        torch.save(self.checkpoint, self.checkpointPath)

    def saveBestCheckpoint(self, epoch: int, accuracy: float):
        """
        Write the best checkpoint to the file self.checkpoint.

        Parameters
        ----------
        epochs : int
            The number of epochs the last epoch.
        accuracy: float
            The accuracy of the test data.
        """

        self.checkpointBest["accuracy"] = accuracy
        self.checkpointBest["model_state_dict"] = self.model.state_dict()
        self.checkpointBest["optimizer_state_dict"] = self.optimizer.state_dict()

        # check that number of epochs and len of accuracy is equal
        for nameDataSet in self.keysDataSets:
            assert epoch == len(self.dicAccuracyList[f"{nameDataSet}"])

        torch.save(self.checkpointBest, self.checkpointBestPath)
        print(
            f"Saving model with test accuracy: {accuracy} to {self.checkpointBestPath}"
        )

    def __train__(
        self, epoch: int, nameDataSet: str, dataLoader: torchDataLoader
    ) -> tuple[torch.Tensor, float]:
        """
        Run one epoch and train the model according to
        mini batch gradient descent and calc the loss,
        the correct classified samples and the
        training accuracy in %.

        Parameters
        ----------
        epoch : int
            The actual epoch.

        Returns
        -------
        tuple[torch.Tensor, float]
            The losses and the accuracy.
        """
        assert nameDataSet == "train"
        self.model.train()
        correct: int = 0
        loss: torch.Tensor = torch.Tensor()
        tic: float = 0
        startTime: float = time()
        numImages: int = len(dataLoader.dataset)
        numBatchTotal: int = len(dataLoader)

        print(f"Start train at: {ctime(startTime)}")
        for currentBatchNumber, (xTrain, yTrain) in enumerate(dataLoader):
            tic = time()
            self.optimizer.zero_grad()
            z = self.model(xTrain)

            # convert the torch.int64 Tensor to torch.float32
            # dtype(z): torch.float32
            # dtype(y): torch.int64

            loss = self.calcLoss(z, yTrain.float())
            correct += self.calcCorrect(z, yTrain.float())
            loss.backward()
            self.optimizer.step()

            # sleep after every batch
            # after every batch sleep x percent of the time used for the last batch
            timeBatch: float = time() - tic
            sleep(timeBatch * self.sleepFactorBatch)
            # print(f"Train time for one batch is: {timeBatch}")
            # print(f"Batch sleeping for: {timeBatch * self.sleepFactorBatch} s")

            # sleep periodically
            if (time() - startTime) > self.sleepIntervalPeriodic:
                # print(f"Periodic Sleeping for: {self.sleepTimePeriodic} s")
                sleep(self.sleepTimePeriodic)
                startTime = time()

            # print after every batch for numBatchTotal < 10
            if int(numBatchTotal / 10) == 0:
                print(
                    f"Processed {self.batchSize*currentBatchNumber} from {numImages} images."
                )
            # print after every 10% of total batches are calculated
            elif (currentBatchNumber % int(numBatchTotal / 10)) == 0:
                print(
                    f"Processed {self.batchSize*currentBatchNumber} from {numImages} images."
                )

        print(f"correct train: {correct}")
        # calc accuracy in %
        accuracy = correct / self.dicNumSamples[f"{nameDataSet}"]
        self.dicAccuracyList[f"{nameDataSet}"].append(accuracy)
        assert epoch == len(self.dicAccuracyList[f"{nameDataSet}"])
        return loss, accuracy

    def __eval__(
        self, epoch: int, nameDataSet: str, dataLoader: torchDataLoader
    ) -> tuple[torch.Tensor, float]:
        """
        Evaluate the model on the test data and
        calc the loss, the correct classified
        samples and the training accuracy in %.

        Parameters
        ----------
        epoch : int
            The actual epoch.

        Returns
        -------
        tuple[torch.Tensor, float]
            The losses and the accuracy.
        """
        self.model.eval()
        correct: int = 0
        loss: torch.Tensor = torch.Tensor()
        tic: float = 0
        startTime: float = time()
        numImages: int = len(dataLoader.dataset)
        numBatchTotal: int = len(dataLoader)

        print(f"\nStart eval at: {ctime(startTime)}")
        print(f"Dataset: {nameDataSet}")
        print(f"numBatch: {numBatchTotal}")
        for currentBatchNumber, (xTest, yTest) in enumerate(dataLoader):
            tic = time()
            z = self.model(xTest)
            loss = self.calcLoss(z, yTest.float())
            correct += self.calcCorrect(z, yTest.float())

            # sleep after every batch
            # after every batch sleep x percent of the time used for the last batch
            timeBatch: float = time() - tic
            sleep(timeBatch * self.sleepFactorBatch)
            # print(f"Eval time for one batch is: {timeBatch}")
            # print(f"Batch sleeping for: {timeBatch * self.sleepFactorBatch} s")

            # sleep periodically
            if (time() - startTime) > self.sleepIntervalPeriodic:
                # print(f"Periodic Sleeping for: {self.sleepTimePeriodic} s")
                sleep(self.sleepTimePeriodic)
                startTime = time()

            # print after every batch for numBatchTotal < 10
            if int(numBatchTotal / 10) == 0:
                print(
                    f"Processed {self.batchSize*currentBatchNumber} from {numImages} images."
                )
            # print after every 10% of total batches are calculated
            elif (currentBatchNumber % int(numBatchTotal / 10)) == 0:
                print(
                    f"Processed {self.batchSize*currentBatchNumber} from {numImages} images."
                )

        print(f"correct {nameDataSet}: {correct}")
        # calc accuracy in %
        accuracy: float = correct / self.dicNumSamples[f"{nameDataSet}"]
        self.dicAccuracyList[f"{nameDataSet}"].append(accuracy)
        assert epoch == len(self.dicAccuracyList[f"{nameDataSet}"])
        return loss, accuracy

    def calcLoss(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calc the losses for back propagation,
        according to a criterion.

        Parameters
        ----------
        z : torch.Tensor
            The predicted lables.
        z : torch.Tensor
            The true labels.
        accuracy: float
            The accuracy of the test data.

        Returns
        -------
        torch.Tensor
            The losses.
        """
        loss: torch.Tensor = torch.Tensor()
        if self.criterion == "BCE":
            critBCE = nn.BCELoss()
            loss = critBCE(z.view(-1), y)
        elif self.criterion == "CE":
            critCE = nn.CrossEntropyLoss()
            loss = critCE(z, y.long())
        else:
            assert False, f"No operation definied for criterion {self.criterion}"
        return loss

    def calcCorrect(self, z: torch.Tensor, y: torch.Tensor) -> int:
        """
        Calc the number of correct classified samples.

        Parameters
        ----------
        z : torch.Tensor
            The predicted lables.
        y : torch.Tensor
            The true labels.

        Returns
        -------
        int
            The number of correct classified samples.
        """

        correct: int = 0
        if self.criterion == "BCE":
            correct = self.calcCorrectForBCELoss(z, y)
        elif self.criterion == "CE":
            correct = self.calcCorrectForCELoss(z, y)
        else:
            assert False, f"No operation definied for criterion {self.criterion}"

        return correct

    def calcCorrectForBCELoss(self, z: torch.Tensor, y: torch.Tensor) -> int:
        """
        Calc the number of correct classified samples for the
        binary cross entropy criterion.

        Parameters
        ----------
        z : torch.Tensor
            The probability for the predicted classes
            with shape [batchSize, 1]
        y : torch.Tensor
            The true labels.

        Returns
        -------
        int
            The number of correct classified samples.
        """
        # For logits use predicted_labels = y_pred > 0.0
        correct: torch.Tensor = torch.zeros([1], dtype=torch.int32)
        correct += ((z.view(-1).detach() > 0.5) == (y > 0.5)).sum()
        return int(correct)

    def calcCorrectForCELoss(self, z: torch.Tensor, y: torch.Tensor) -> int:
        """
        Calc the number of correct classified samples for the
        cross entropy criterion.

        Parameters
        ----------
        z : torch.Tensor
            The probability for the predicted classes
            with shape [batchSize, numLabels]
        y : torch.Tensor
            The true labels with shape [batchSize]

        Returns
        -------
        int
            The number of correct classified samples.
        """
        yhat: torch.Tensor = torch.zeros([1], dtype=torch.int64)
        correct: torch.Tensor = torch.zeros([1], dtype=torch.int64)

        # torch.max return the probability of the predicted class and the predicted class
        # maxbe use probability later
        # pylint: disable=W0612
        probability: torch.Tensor
        probability, yhat = torch.max(z.detach().data, 1)
        # pylint: enable=W0612

        # print(f"probability: {probability}")
        # print(f"yhat: {yhat}")

        correct += (yhat == y).sum().item()
        # print(f"correct: {correct}")
        return int(correct)

    def plotLearningProgress(
        self,
        epoch: int,
        epochs: int,
        dicAccuracyList: dict[str, list[float]],
        title: str,
        forcePlot: bool = False,
    ):
        """
        Plots the training and test accuracy.

        Parameters
        ----------
        epoch : int
            The actual epoch.
        epochs : int
            The number of epochs to train.
        """
        if epoch % 1 == 0 or forcePlot:
            plot2 = cPlot2D((1, epochs), (0, 100), title)
            plot2.plotLearningProcess(dicAccuracyList)
            plt.show()
