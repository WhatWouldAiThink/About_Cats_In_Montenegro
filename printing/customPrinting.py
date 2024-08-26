# Copyright © 2024, whatwouldaithink.com.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Module provide functions to print basic DataFrame information."""

import pandas as pd
from IPython.display import display


def info(df: pd.DataFrame):
    """
    This function prints basic informations
    about a DataFrame.

    Parameters
    ----------
    df : dataFrame
        The DataFrame to print.
    """
    print("==================== df.info() ====================")
    display(df.info())
    print("==================== df.head() ====================")
    display(df.head())
    print("================== df.describe() ==================")
    display(df.describe())
    print()
