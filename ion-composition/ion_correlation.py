#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import argparse

# 3rd party imports
import yaml
import xarray as xr
import matplotlib.pyplot as plt

from pyrfu.plot import plot_heatmap, annotate_heatmap

# Local imports
from jfs.plot import add_eis_charge_state

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2021"
__license__ = "Apache 2.0"

plt.style.use("scientific")


def main(args):
    # Read time intervals
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    correlation = xr.load_dataset(cfg["file_path"])
    corr_hplus_henplus = correlation["correlation_hplus_henplus"]

    f, ax = plt.subplots(1, figsize=(14, 10))
    f.subplots_adjust(top=0.94, bottom=0.06, left=0.05, right=0.95)
    im, cbar = plot_heatmap(
        ax,
        corr_hplus_henplus,
        corr_hplus_henplus.energy_00.data,
        corr_hplus_henplus.energy_01.data,
        cbarlabel="Correlation",
        cmap="RdBu",
        vmin=0,
        vmax=1,
    )

    annotate_heatmap(im, textcolors=("white", "black"), threshold=0.12)

    add_eis_charge_state(
        ax,
        corr_hplus_henplus.energy_00.data,
        corr_hplus_henplus.energy_01.data,
        charge=[1, 2],
        linewidth=3,
        edgecolor="y",
        facecolor="none",
        zorder=3,
    )

    cbar.ax.set_ylabel("Correlation")
    ax.set_ylabel("$K_{H^{+}}$ [keV]")
    ax.set_xlabel("$K_{He^{n+}}$ [keV]")
    ax.xaxis.set_label_position("top")

    plt.savefig("./figures/figure_3.pdf")
    plt.savefig("./figures/figure_3.png", dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Path to the configuration file (.yml)",
        required=True,
        type=str,
    )
    main(parser.parse_args())
