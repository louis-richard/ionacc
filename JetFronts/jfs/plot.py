#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot tools
"""

# Built-in imports
import itertools

from datetime import datetime

# 3rd party imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib import dates as mdates


__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def show_tint(ax, tint, color="k"):

    r"""Add rectangle to panel to show time interval

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis
    tint : list of str
        Time interval.
    color : str
        Color of the rectangle.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis with the rectangle.

    """
    start_time = datetime.fromisoformat(tint[0][:-3])
    stop_time = datetime.fromisoformat(tint[1][:-3])

    start, stop = [mdates.date2num(start_time), mdates.date2num(stop_time)]

    width = stop - start

    ymin, ymax = ax.get_ylim()

    height = 0.05 * (ymax - ymin)

    rect = plt.Rectangle((start, ymax - height), width, height, color=color)
    ax.add_patch(rect)

    return ax


def plot_tetrahedron(ax, r_gsm_sep):
    r"""Plot MMS tetrahedron configuration in 3D

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis
    r_gsm_sep : list of array_like
        Spacecraft locations with respect to the center of mass of the
        tetrahedron.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis

    """
    for i in range(0, 4):
        ax.plot3D(r_gsm_sep[i][0], r_gsm_sep[i][1], r_gsm_sep[i][2],
                  marker="d", linestyle="none", markersize=8.5,
                  label=f"MMS {i + 1}")
        ax.view_init(elev=20., azim=-70)

    for i, j in itertools.combinations([0, 1, 2, 3], 2):
        x_ = [r_gsm_sep[i][0], r_gsm_sep[j][0]]
        y_ = [r_gsm_sep[i][1], r_gsm_sep[j][1]]
        z_ = [r_gsm_sep[i][2], r_gsm_sep[j][2]]
        ax.plot3D(x_, y_, z_, "k-", linewidth=1)

    return ax


def add_eis_charge_state(ax, energy_0, energy_1, charge, **kwargs):
    r"""Add boxes indicating the integer charge states.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis
    energy_0 : array_like
        Energy of the first specie (y-axis).
    energy_1 : array_like
        Energy of the second specie (x-axis).
    charge : list of ints
        Charge states to mark.

    Other Parameters
    ----------------
    kwargs : dict, Optional
        Keyword arguments for patches.Rectangle

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis

    """

    energy_0 = np.hstack([energy_0[0] - np.diff(energy_0)[0], energy_0,
                          energy_0[-1] + np.diff(energy_0)[-1]])

    for c in charge:
        for i, e_1 in enumerate(energy_1):
            j = np.argmin(np.abs(c * energy_0 - e_1)) - 1
            rect = patches.Rectangle((i - .5, j - .5), 1, 1, **kwargs)

            # Add the patch to the Axes
            ax.add_patch(rect)

    return ax
