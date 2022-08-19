#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import argparse

# 3rd party imports
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# from scipy import optimize
from pyrfu.mms import rotate_tensor
from pyrfu.pyrf import cotrans, norm, resample, trace
from pyrfu.plot import (
    make_labels,
    plot_line,
    plot_magnetosphere,
    plot_spectr,
    span_tint,
)
from scipy import constants

# Local imports
from jfs.utils import find_feeps_clusters
from jfs.plot import show_tint, plot_tetrahedron
from jfs.load import (
    load_r_mmsx,
    load_eb_mmsx,
    load_fpi_def_omni_mmsx,
    load_fpi_moments_mmsx,
    load_hpca_moments,
    load_feeps_dpf_omni_mmsx,
)

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2021"
__license__ = "Apache 2.0"

plt.style.use("scientific")


def residual(a, n_h, n_he):
    return np.sum((n_h.data - a * n_he.data) ** 2) / len(n_h.data)


def main(args):
    # Read time intervals
    with open(args.config) as fs:
        cfg = yaml.load(fs, Loader=yaml.FullLoader)

    tint = cfg["tints"]["overview"]

    # Load spacecraft location
    _, r_gsm_avg, r_gsm_sep = load_r_mmsx(tint, cfg)
    print(r_gsm_avg)

    # %%
    b_gse, _ = load_eb_mmsx(tint, cfg)
    b_gsm = cotrans(b_gse, "gse>gsm")

    # %% Load ion and electron differential energy flux from FPI-DIS and
    # FPI-DES
    def_omni_fpi_i, def_omni_fpi_e = load_fpi_def_omni_mmsx(tint, cfg)

    # %% Load ion and electron moments from FPI-DIS and FPI-DES
    moments_i, _ = load_fpi_moments_mmsx(tint, cfg)
    n_i, v_gse_i, t_gse_i, _ = moments_i

    # Transform ion bulk velocity to GSM coordinates and compute scalar
    # temperature from temperature tensor
    v_gsm_i = cotrans(v_gse_i, "gse>gsm")
    t_fac_i = rotate_tensor(t_gse_i, "fac", b_gse)
    t_i = trace(t_fac_i) / 3

    # %% Load H+ and He++ moments from HPCA
    # moments_p, moments_a = load_hpca_moments_mmsx(tint, cfg)
    moments_p, moments_a = load_hpca_moments(tint, 2, cfg)
    _, _, _, p_p = moments_p
    _, _, _, p_a = moments_a

    # %% Compute plasma beta
    p_tot = 1e-9 * p_p + 1e-9 * resample(p_a, p_p)  # Plasma press.
    p_mag = 1e-18 * norm(b_gsm) ** 2 / (2 * constants.mu_0)  # Mag. press.
    beta_ = p_tot / resample(p_mag, p_p)  # plasma beta

    # %% Load high energy ion and electron differential particle flux from
    # FEEPS
    dpf_omni_feeps_i, dpf_omni_feeps_e = load_feeps_dpf_omni_mmsx(tint, cfg)

    # %%
    times, tints = find_feeps_clusters(dpf_omni_feeps_i)
    t_idx = [503, 531, 608, 682, 797, 927]
    n_tid = len(t_idx) + 1

    # %%
    fig = plt.figure(figsize=(12, 17.2))
    gsp1 = fig.add_gridspec(
        20, 1, top=0.95, bottom=0.05, left=0.1, right=0.9, hspace=0.1
    )

    gsp10 = gsp1[:3].subgridspec(1, 3, hspace=0)
    gsp11 = gsp1[4:].subgridspec(8, 1, hspace=0)

    # Create axes in the grid spec
    axs10 = [fig.add_subplot(gsp10[i]) for i in range(2)]
    axs01 = fig.add_subplot(gsp10[2], projection="3d")
    axs11 = [fig.add_subplot(gsp11[i]) for i in range(8)]

    # Plot MMS tetrahedron configuration
    axs01 = plot_tetrahedron(axs01, r_gsm_sep)
    axs01.set_xlabel("$X_{GSM}$ [km]")
    axs01.set_ylabel("$Y_{GSM}$ [km]")
    axs01.set_zlabel("$Z_{GSM}$ [km]")

    axs01.legend(
        loc="upper right",
        ncol=2,
        bbox_to_anchor=(0, 1.1, 1, 0.2),
        bbox_transform=axs01.transAxes,
    )

    field_lines = [False, True]
    for i, y_axis in zip(range(2), ["$Y_{GSM}$ [$R_E$]", "$Z_{GSM}$ [$R_E$]"]):
        plot_magnetosphere(axs10[i], tint, field_lines=field_lines[i])
        axs10[i].invert_xaxis()
        axs10[i].plot(
            r_gsm_avg[0] / 6371,
            r_gsm_avg[i + 1] / 6371,
            marker="^",
            color="tab:red",
            linestyle="",
            label="MMS",
        )
        axs10[i].set_xlim([-30, 15])
        axs10[i].set_ylim([-20, 20])
        axs10[i].set_aspect("equal")
        axs10[i].set_xlabel("$X_{GSM}$ [$R_E$]")
        axs10[i].set_ylabel(y_axis)
        axs10[i].invert_xaxis()

    # Plot magnetic field in GSM coordinates
    plot_line(axs11[0], b_gsm, zorder=n_tid)
    axs11[0].legend(
        ["$B_x$", "$B_y$", "$B_z$", "|B|"], frameon=True, ncol=3, loc="upper right"
    )
    axs11[0].set_ylabel("$B$" + "\n" + "[nT]")
    axs11[0].set_ylim([-22, 15])

    for t_ in times:
        show_tint(axs11[0], t_, "tab:purple")

    # Plot the FPI-DIS and HPCA H+ bulk velocity in GSM coordinates
    comps_ = ["x", "y", "z"]
    colors_i = ["tab:blue", "tab:green", "tab:red"]
    for i, c_fpi in zip(range(3), colors_i):
        plot_line(
            axs11[1],
            v_gsm_i[:, i],
            zorder=n_tid + i,
            color=c_fpi,
            label=f"$V_{{i{comps_[i]}}}$",
        )

    axs11[1].legend(ncol=3, frameon=True, loc="upper right")
    axs11[1].set_ylim([-800, 2200])
    axs11[1].set_ylabel("$V_i$" + "\n" + "[km s$^{-1}$]")

    # Plot FPI-DIS, HPCA H+ and scaled HPCA He++ number densities
    plot_line(axs11[2], n_i, zorder=n_tid + 0, label="$n_i$")
    axs11[2].set_ylabel("$n_i$" + "\n" + "[cm$^{-3}$]")

    # Plot plasma beta
    plot_line(axs11[3], beta_, color="tab:blue")
    axs11[3].set_yscale("log")
    axs11[3].set_ylim([2e-2, 1.3e3])
    axs11[3].set_ylabel("$\\beta_i$" + "\n" + " ")
    axs11[3].axhspan(0.02, 0.1, color="black", alpha=0.2)
    axs11[3].axhspan(0.1, 0.7, color="tab:red", alpha=0.2)
    axs11[3].axhspan(0.7, 1.3e3, color="tab:green", alpha=0.2)
    axs11[3].text(0.93, 0.85, "CPS", color="tab:green", transform=axs11[3].transAxes)
    axs11[3].text(0.93, 0.2, "PSBL", color="tab:red", transform=axs11[3].transAxes)
    axs11[3].text(0.93, 0.03, "Lobe", color="k", transform=axs11[3].transAxes)

    axs11[4], caxs4 = plot_spectr(
        axs11[4],
        dpf_omni_feeps_i[:, 2:],
        yscale="log",
        cscale="log",
        clim=[2e-1, 2e2],
        cmap="Spectral_r",
    )
    axs11[4].set_ylabel("$K_i$" + "\n" + "[keV]")
    caxs4.set_ylabel("Diff. Flux" + "\n" + "[(cm$^{2}$ s sr keV)$^{-1}$]")

    axs11[5], caxs5 = plot_spectr(
        axs11[5],
        def_omni_fpi_i[:, 13:],
        yscale="log",
        cscale="log",
        clim=[1e3, 1e6],
        cmap="Spectral_r",
    )
    plot_line(axs11[5], t_i, zorder=n_tid + 0, label="$T_i$")
    axs11[5].set_ylabel("$K_i$" + "\n" + "[eV]")
    axs11[5].legend(loc="lower right", frameon=True, ncol=3)
    axs11[5].grid(visible=False, which="major")
    caxs5.set_ylabel("DEF" + "\n" + "[(cm$^{2}$ s sr)$^{-1}$]")

    axs11[6], caxs6 = plot_spectr(
        axs11[6],
        dpf_omni_feeps_e[:, 1:11],
        yscale="log",
        cscale="log",
        clim=[2e0, 2e3],
        cmap="Spectral_r",
    )
    axs11[6].set_ylabel("$K_e$" + "\n" + "[keV]")
    caxs6.set_ylabel("Diff. Flux" + "\n" + "[(cm$^{2}$ s sr keV)$^{-1}$]")

    axs11[7], caxs7 = plot_spectr(
        axs11[7],
        def_omni_fpi_e[:, 9:],
        yscale="log",
        cscale="log",
        clim=[42.8e3, 42.8e6],
        cmap="Spectral_r",
    )
    # axs11[4].axhline(def_omni_fpi_e.energy.data[7])
    axs11[7].set_ylabel("$K_e$" + "\n" + "[eV]")
    caxs7.set_ylabel("DEF" + "\n" + "[(cm$^{2}$ s sr)$^{-1}$]")

    fpi_time = def_omni_fpi_i.time.data
    for i, t_ in enumerate(t_idx):
        for ax in axs11[:4]:
            ax.axvline(
                fpi_time[t_], linestyle=":", color="k", zorder=i + 1, linewidth=1.2
            )

        for ax in axs11[4:]:
            ax.axvline(fpi_time[t_], linestyle=":", color="k", linewidth=1.2)

    axs11[-1].get_shared_x_axes().join(*axs11)

    fig.align_ylabels(axs11)

    for ax in axs11[:-1]:
        ax.xaxis.set_ticklabels([])

    axs11[-1].set_xlim(mdates.datestr2num(tint))

    make_labels(axs10, [0.028, 0.9], pad=0)
    make_labels([axs01], [0.028, 0.9], pad=2)
    make_labels(axs11, [0.008, 0.86], pad=3)

    for t_ in tints:
        span_tint(axs11, t_, ec="k", fc="tab:purple", alpha=0.2)

    plt.savefig("./figures/figure_1.pdf")
    plt.savefig("./figures/figure_1.png", dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Path to the configuration file (.yml)",
        required=True,
        type=str,
    )
    main(parser.parse_args())
