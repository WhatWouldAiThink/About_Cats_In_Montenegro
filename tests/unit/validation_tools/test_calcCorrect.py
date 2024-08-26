# Copyright © 2024, whatwouldaithink.com.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This module test the method calcCorrect(yhat, y)
from the class Trainingmanager for
binary cross entropy loss (BCE) and for cross entropy loss (CE).
"""

import torch
from vision.TrainModels import TrainingManager


class FakeTrainingManagerForTesting(TrainingManager):
    """
    Create a kind of mock class by not calling
    super.__init__().
    """

    def __init__(self, **kwargs):
        # pylint: disable=W0231
        # add needed attributes to test class TrainingManager
        for key, value in zip(kwargs.keys(), kwargs.values()):
            # we need only the attribute 'criterion' to test
            # the method calcCorrect(yhat, y) from TrainingManager
            if key == "criterion":
                self.criterion = value
        # print the properties
        # print(f"self.__dict__: {self.__dict__}")


def test_calcCorrectForBCE() -> None:
    """
    A function to test calcCorrect(yhat, y) for BCE by calling
    function helper_calcCorrect(yhat, y, numEqualClasses) with
    different input combinations.
    """

    # -------------------- test against y = 0 --------------------------------

    # compare tensor([0., 0., 0., 0., 0.]) vs. tensor([0., 0., 0., 0., 0.])
    # expect 5 equal
    yhat: torch.Tensor = torch.zeros([5], dtype=torch.float)
    y: torch.Tensor = torch.zeros([5], dtype=torch.float)
    helper_calcCorrectBCE(yhat, y, 5)

    # compare tensor([0.5, 0., 0.5, 0., 0.5]) vs. tensor([0., 0., 0., 0., 0.])
    # expect 5 equal
    yhat: torch.Tensor = torch.zeros([5], dtype=torch.float)
    yhat[0] = 0.5
    yhat[2] = 0.5
    yhat[4] = 0.5
    y: torch.Tensor = torch.zeros([5], dtype=torch.float)
    helper_calcCorrectBCE(yhat, y, 5)

    # compare tensor([0.5, 0.5, 0.5, 0.5, 0.5]) vs. tensor([0., 0., 0., 0., 0.])
    # expect 5 equal
    yhat: torch.Tensor = torch.zeros([5], dtype=torch.float)
    for i, _ in enumerate(yhat):
        yhat[i] = 0.5
    y: torch.Tensor = torch.zeros([5], dtype=torch.float)
    helper_calcCorrectBCE(yhat, y, 5)

    # test against y = 1

    # compare tensor([0.51, 0.51, 0.51, 0.51, 0.51]) vs. tensor([0., 0., 0., 0., 0.])
    # expect 0 equal
    yhat: torch.Tensor = torch.zeros([5], dtype=torch.float)
    for i, _ in enumerate(yhat):
        yhat[i] = 0.51
    y: torch.Tensor = torch.zeros([5], dtype=torch.float)
    helper_calcCorrectBCE(yhat, y, 0)

    # compare tensor([1., 1., 1., 1., 1.]) vs. tensor([0., 0., 0., 0., 0.])
    # expect 0 equal
    yhat: torch.Tensor = torch.ones([5], dtype=torch.float)
    y: torch.Tensor = torch.zeros([5], dtype=torch.float)
    helper_calcCorrectBCE(yhat, y, 0)

    # -------------------- test against y = 1 --------------------------------

    # compare tensor([0., 0., 0., 0., 0.]) vs. tensor([1., 1., 1., 1., 1.])
    # expect 5 equal
    yhat: torch.Tensor = torch.zeros([5], dtype=torch.float)
    y: torch.Tensor = torch.ones([5], dtype=torch.float)
    helper_calcCorrectBCE(yhat, y, 0)

    # compare tensor([0.5, 0., 0.5, 0., 0.5]) vs. tensor([1., 1., 1., 1., 1.])
    # expect 5 equal
    yhat: torch.Tensor = torch.zeros([5], dtype=torch.float)
    yhat[0] = 0.5
    yhat[2] = 0.5
    yhat[4] = 0.5
    y: torch.Tensor = torch.ones([5], dtype=torch.float)
    helper_calcCorrectBCE(yhat, y, 0)

    # compare tensor([0.5, 0.5, 0.5, 0.5, 0.5]) vs. tensor([1., 1., 1., 1., 1.])
    # expect 5 equal
    yhat: torch.Tensor = torch.zeros([5], dtype=torch.float)
    for i, _ in enumerate(yhat):
        yhat[i] = 0.5
    y: torch.Tensor = torch.ones([5], dtype=torch.float)
    helper_calcCorrectBCE(yhat, y, 0)

    # compare tensor([0.51, 0., 0.51, 0., 0.51]) vs. tensor([1., 1., 1., 1., 1.])
    # expect 0 equal
    yhat: torch.Tensor = torch.zeros([5], dtype=torch.float)
    yhat[0] = 0.51
    yhat[2] = 0.51
    yhat[4] = 0.51
    y: torch.Tensor = torch.ones([5], dtype=torch.float)
    helper_calcCorrectBCE(yhat, y, 3)

    # compare tensor([0.51, 0.51, 0.51, 0.51, 0.51]) vs. tensor([1., 1., 1., 1., 1.])
    # expect 0 equal
    yhat: torch.Tensor = torch.zeros([5], dtype=torch.float)
    for i, _ in enumerate(yhat):
        yhat[i] = 0.51
    y: torch.Tensor = torch.ones([5], dtype=torch.float)
    helper_calcCorrectBCE(yhat, y, 5)

    # compare tensor([1., 1., 1., 1., 1.]) vs. tensor([1., 1., 1., 1., 1.])
    # expect 0 equal
    yhat: torch.Tensor = torch.ones([5], dtype=torch.float)
    y: torch.Tensor = torch.ones([5], dtype=torch.float)
    helper_calcCorrectBCE(yhat, y, 5)


def helper_calcCorrectBCE(
    yhat: torch.Tensor, y: torch.Tensor, numEqualClasses: int
) -> None:
    """
    A helper function to test the function calcCorrect(yhat, y)
    for Binary Cross Entropy Loss"

    Args:
        yhat (torch.Tensor): The probability for the predicted labels
            with shape [batchSize, 1]
        y (torch.Tensor): The true labels shape [batchSize]
        numEqualClasses (int): Expected number of equal classes in yhat and y.
    """

    # create mocked class
    tm = FakeTrainingManagerForTesting(criterion="BCE")
    assert y.shape == yhat.shape
    result = tm.calcCorrect(yhat, y)
    assert result == numEqualClasses


def test_calcCorrectForCE() -> None:
    """
    A function to test calcCorrect(yhat, y) for CE by calling
    function helper_calcCorrect(yhat, y, numEqualClasses) with
    different input combinations.
    """

    # -------------------------- test against label 0 -----------------------------

    # test against label 0
    # expect 3 equal
    # compare tensor([[0.5, 0.5],[0.5, 0.5],[0.5, 0.5]]) vs. tensor([0., 0., 0.])
    yhat = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=torch.float)
    y: torch.Tensor = torch.zeros([3], dtype=torch.float)
    helper_calcCorrectCE(yhat, y, 3)

    # test against label 0
    # expect 3 equal
    # compare tensor([[0.51, 0.49],[0.51, 0.49],[0.51, 0.49]]) vs. tensor([0., 0., 0.])
    yhat = torch.tensor([[0.51, 0.49], [0.51, 0.49], [0.51, 0.49]], dtype=torch.float)
    y: torch.Tensor = torch.zeros([3], dtype=torch.float)
    helper_calcCorrectCE(yhat, y, 3)

    # test against label 0
    # expect 3 equal
    # compare tensor([[0.6, 0.4],[0.6, 0.4],[0.6, 0.4]]) vs. tensor([0., 0., 0.])
    yhat = torch.tensor([[0.6, 0.4], [0.6, 0.4], [0.6, 0.4]], dtype=torch.float)
    y: torch.Tensor = torch.zeros([3], dtype=torch.float)
    helper_calcCorrectCE(yhat, y, 3)

    # test against label 0
    # expect 0 equal
    # compare tensor([[0.4, 0.6], [0.4, 0.6], [0.4, 0.6]]) vs. tensor([0., 0., 0.])
    yhat = torch.tensor([[0.4, 0.6], [0.4, 0.6], [0.4, 0.6]], dtype=torch.float)
    y: torch.Tensor = torch.zeros([3], dtype=torch.float)
    helper_calcCorrectCE(yhat, y, 0)

    # test against label 0
    # expect 0 equal
    # compare tensor([[0., 1.], [0., 1.], [0., 1.]]) vs. tensor([0., 0., 0.])
    yhat = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float)
    y: torch.Tensor = torch.zeros([3], dtype=torch.float)
    helper_calcCorrectCE(yhat, y, 0)

    # test against label 0
    # expect 3 equal
    # compare tensor([[1., 0.], [1., 0.], [1., 0.]]) vs. tensor([0., 0., 0.])
    yhat = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=torch.float)
    y: torch.Tensor = torch.zeros([3], dtype=torch.float)
    helper_calcCorrectCE(yhat, y, 3)

    # test against label 0
    # expect 2 equal
    # compare tensor([[1., 0.], [0., 1.], [1., 0.]]) vs. tensor([0., 0., 0.])
    yhat = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float)
    y: torch.Tensor = torch.zeros([3], dtype=torch.float)
    helper_calcCorrectCE(yhat, y, 2)

    # -------------------------- test against label 1 -----------------------------

    # test against label 1
    # expect 0 equal
    # compare tensor([[0.5, 0.5],[0.5, 0.5],[0.5, 0.5]]) vs. tensor([0., 0., 0.])
    yhat = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=torch.float)
    y: torch.Tensor = torch.ones([3], dtype=torch.float)
    helper_calcCorrectCE(yhat, y, 0)

    # test against label 1
    # expect 0 equal
    # compare tensor([[0.51, 0.49],[0.51, 0.49],[0.51, 0.49]]) vs. tensor([0., 0., 0.])
    yhat = torch.tensor([[0.51, 0.49], [0.51, 0.49], [0.51, 0.49]], dtype=torch.float)
    y: torch.Tensor = torch.ones([3], dtype=torch.float)
    helper_calcCorrectCE(yhat, y, 0)

    # test against label 1
    # expect 0 equal
    # compare tensor([[0.6, 0.4],[0.6, 0.4],[0.6, 0.4]]) vs. tensor([0., 0., 0.])
    yhat = torch.tensor([[0.6, 0.4], [0.6, 0.4], [0.6, 0.4]], dtype=torch.float)
    y: torch.Tensor = torch.ones([3], dtype=torch.float)
    helper_calcCorrectCE(yhat, y, 0)

    # test against label 0
    # expect 3 equal
    # compare tensor([[0.4, 0.6], [0.4, 0.6], [0.4, 0.6]]) vs. tensor([0., 0., 0.])
    yhat = torch.tensor([[0.4, 0.6], [0.4, 0.6], [0.4, 0.6]], dtype=torch.float)
    y: torch.Tensor = torch.ones([3], dtype=torch.float)
    helper_calcCorrectCE(yhat, y, 3)

    # test against label 0
    # expect 3 equal
    # compare tensor([[0., 1.], [0., 1.], [0., 1.]]) vs. tensor([0., 0., 0.])
    yhat = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float)
    y: torch.Tensor = torch.ones([3], dtype=torch.float)
    helper_calcCorrectCE(yhat, y, 3)

    # test against label 1
    # expect 0 equal
    # compare tensor([[1., 0.], [1., 0.], [1., 0.]]) vs. tensor([0., 0., 0.])
    yhat = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=torch.float)
    y: torch.Tensor = torch.ones([3], dtype=torch.float)
    helper_calcCorrectCE(yhat, y, 0)

    # test against label 1
    # expect 1 equal
    # compare tensor([[1., 0.], [0., 1.], [1., 0.]]) vs. tensor([0., 0., 0.])
    yhat = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float)
    y: torch.Tensor = torch.ones([3], dtype=torch.float)
    helper_calcCorrectCE(yhat, y, 1)


def helper_calcCorrectCE(
    yhat: torch.Tensor, y: torch.Tensor, numEqualClasses: int
) -> None:
    """
    A helper function to test the function calcCorrect(yhat, y)
    for Cross Entropy Loss"

    Args:
        yhat (torch.Tensor): The probability for the predicted labels
            with shape [batchSize, numLabels]
        y (torch.Tensor): The true labels with shape [batchSize]
        numEqualClasses (int): Expected number of equal classes in yhat and y.
    """

    # create mocked class
    tm = FakeTrainingManagerForTesting(criterion="CE")
    result = tm.calcCorrect(yhat, y)
    assert result == numEqualClasses
