#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import argparse
import itertools

# 3rd party imports
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pyrfu.pyrf import cotrans, cross
from pyrfu.plot import plot_line, span_tint, make_labels, plot_clines

# Local imports
from jfs.utils import combine_flux_instruments
from jfs.load import (load_fpi_moments_mmsx, load_hpca_moments,
                      load_eb_mmsx, load_eis_omni_mmsx,
                      load_fpi_dpf_omni_mmsx, load_hpca_flux,
                      load_feeps_dpf_omni_mmsx)

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2021"
__license__ = "Apache 2.0"

plt.style.use("scientific")

def main(args):
    # Read time intervals
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    tint = cfg["tint"]

    b_gse, e_gse = load_eb_mmsx(tint, cfg)
    b_gsm, e_gsm = [cotrans(field, "gse>gsm") for field in [b_gse, e_gse]]

    # %% Load ion and electron moments from FPI-DIS and FPI-DES
    moments_i, _ = load_fpi_moments_mmsx(tint, cfg)
    _, v_gse_i, _, _ = moments_i

    # Transform ion bulk velocity to GSM coordinates
    v_gsm_i = cotrans(v_gse_i, "gse>gsm")

    # Compute the motionnal electric field
    evxb_gsm_i = -1e-3 * cross(cotrans(v_gse_i, "gse>gsm"), b_gsm)

    # %% Load H+ and He++ moments from HPCA
    # moments_p, _ = load_hpca_moments_mmsx(tint, cfg)
    moments_p, _ = load_hpca_moments(tint, 2, cfg)
    _, v_gsm_p, _, _ = moments_p

    evxb_gsm_p = -1e-3 * cross(v_gsm_p, b_gsm)

    # %% Load FPI-DIS and FPI-DES omni-directional differential particle flux.
    dpf_fpi_omni_i, dpf_fpi_omni_e = load_fpi_dpf_omni_mmsx(tint, cfg)

    # %% Load EIS omni-directional proton differential particle flux.
    # Protons (H+)
    dpf_eis_omni_p, _ = load_eis_omni_mmsx("proton", tint, cfg)
    # Alphas (Hen+)
    dpf_eis_omni_a, _ = load_eis_omni_mmsx("alpha", tint, cfg)
    # Electrons (e)
    # dpf_eis_omni_e, _ = load_eis_omni_mmsx("electron", tint, cfg)

    # Load FEEPS electron flux
    _, dpf_omni_feeps_e = load_feeps_dpf_omni_mmsx(tint, cfg)

    # %% Load alpha particles omni-directional differential particle flux.
    # _, dpf_hpca_omni_a = load_hpca_flux_mmsx(tint, cfg)
    dpf_hpca_omni_p, dpf_hpca_omni_a = load_hpca_flux(tint, 2, cfg)
    dpf_hpca_omni_p.data *= 1e3
    dpf_hpca_omni_a.data *= 1e3

    # %% Combine FPI-DIS & EIS H+, HPCA He++ & EIS Hen+ and FPI-DES & EIS e
    # dpf_omni_p = combine_flux_instruments(dpf_fpi_omni_i, dpf_eis_omni_p)
    dpf_omni_p = combine_flux_instruments(dpf_hpca_omni_p, dpf_eis_omni_p)
    dpf_omni_a = combine_flux_instruments(dpf_hpca_omni_a, dpf_eis_omni_a)
    dpf_omni_e = combine_flux_instruments(dpf_fpi_omni_e, dpf_omni_feeps_e)

    t_idx = [503, 531, 608, 682, 797, 927]

    # %% Plot
    fig, axs = plt.subplots(6, sharex="all", figsize=(10.2, 14.4))
    fig.subplots_adjust(top=.95, bottom=.05, left=.11, right=.89, hspace=0)

    plot_line(axs[0], b_gsm)
    axs[0].legend(["$B_{x}$", "$B_{y}$", "$B_{z}$"], frameon=True,
                  loc="upper right", ncol=3)
    axs[0].set_ylabel("$B$" + "\n" + "[nT]")

    plot_line(axs[1], e_gsm[:, 1], color="tab:green", label="$E_y$")
    plot_line(axs[1], evxb_gsm_i[:, 1], color="tab:blue",
              label="$(-V_i \\times B)_y$")
    plot_line(axs[1], evxb_gsm_p[:, 1], color="tab:cyan",
              label="$(-V_{H^+} \\times B)_y$")
    axs[1].set_ylim([-12, 12])
    axs[1].legend(frameon=True, loc="upper right", ncol=3)
    axs[1].set_ylabel("$E_y$" + "\n" + "[mV m$^{-1}$]")

    comp_ = ["x", "y", "z"]
    colors_i = ["tab:blue", "tab:green", "tab:red"]
    colors_p = ["tab:cyan", "tab:olive", "tab:pink"]
    for i, c_i, c_p in zip(range(3), colors_i, colors_p):
        plot_line(axs[2], v_gsm_i[:, i], color=c_i,
                  label=f"$V_{{i{comp_[i]}}}$")
        plot_line(axs[2], v_gsm_p[:, i], color=c_p,
                  label=f"$V_{{H^+{comp_[i]}}}$")

    axs[2].set_ylim([-1100, 2200])
    axs[2].legend(frameon=True, loc="upper right", ncol=3)
    axs[2].set_ylabel("$V_p$" + "\n" + "[km s$^{-1}$]")

    idx = np.logical_and(dpf_omni_p.energy.data > .1,
                         dpf_omni_p.energy.data < 200)
    axs[3], caxs3 = plot_clines(axs[3], dpf_omni_p[:, idx], yscale="log",
                                cscale="log", cmap="viridis")
    axs[3].set_ylim([1e-1, 7e7])
    axs[3].set_ylabel("Diff. Flux H$^{+}$" + "\n"
                      + "[(cm$^{2}$ s sr keV)$^{-1}$]")
    caxs3.set_ylabel("$K_{H^{+}}$ [keV]")

    idx = np.logical_and(dpf_omni_a.energy.data > .1,
                         dpf_omni_a.energy.data < 200)
    axs[4], caxs4 = plot_clines(axs[4], dpf_omni_a[:, idx], yscale="log",
                                cscale="log", cmap="viridis")
    axs[4].set_ylim([1e-1, 7e7])
    axs[4].set_ylabel("Diff. Flux He$^{n+}$" + "\n"
                      + "[(cm$^{2}$ s sr keV)$^{-1}$]")
    caxs4.set_ylabel("$K_{He^{n+}}$ [keV]")

    idx = np.logical_and(dpf_omni_e.energy.data > .1,
                         dpf_omni_e.energy.data < 200)
    axs[5], caxs5 = plot_clines(axs[5], dpf_omni_e[:, idx], yscale="log",
                                cscale="log", cmap="viridis")
    axs[5].set_ylim([1e-1, 7e7])
    axs[5].set_ylabel("Diff. Flux e" + "\n" + "[(cm$^{2}$ s sr keV)$^{-1}$]")
    caxs5.set_ylabel("$K_{e}$ [keV]")

    axs[-1].get_shared_x_axes().join(*axs)

    fig.align_ylabels(axs)
    axs[-1].set_xlim(mdates.datestr2num(tint))

    make_labels(axs, [.01, .9])
    span_tint(axs, cfg["tints"][0], ec="none", fc="tab:purple", alpha=.2)
    span_tint(axs, cfg["tints"][1], ec="none", fc="tab:purple", alpha=.2)

    fpi_time = dpf_fpi_omni_i.time.data
    for (i, t_), ax in itertools.product(enumerate(t_idx), axs):
        ax.axvline(fpi_time[t_], linestyle=":", color="k", linewidth=1.2)

    plt.savefig("./figures/figure_2.pdf")
    plt.savefig("./figures/figure_2.png", dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        help="Path to the configuration file (.yml)",
                        required=True, type=str)
    main(parser.parse_args())
