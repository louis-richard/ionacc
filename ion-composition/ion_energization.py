#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import argparse
import warnings

# 3rd party imports
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerTuple, HandlerErrorbar
from uncertainties import unumpy, nominal_value, std_dev
from pyrfu.plot import plot_line, plot_spectr, span_tint, make_labels
from pyrfu.mms import (vdf_omni, eis_skymap, eis_skymap_combine_sc,
                       eis_ang_ang)
from pyrfu.pyrf import (cotrans, calc_dt, datetime642iso8601, time_clip,
                        iso86012datetime64, norm)

# Local imports
from jfs.load import (load_eb_mmsx, load_eis_omni_mmsx, load_eis_allt_mmsx,
                      load_hpca_flux)

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2021"
__license__ = "Apache 2.0"

plt.style.use("scientific")


def calc_ratio(dpf_sour, dpf_peak, cts_sour, cts_peak):
    dpf_sour.data[dpf_sour.data == 0] = np.nan
    dpf_peak.data[dpf_peak.data == 0] = np.nan
    cts_sour.data[cts_sour.data == 0] = np.nan
    cts_peak.data[cts_peak.data == 0] = np.nan

    flux_sour = unumpy.uarray(dpf_sour.data,
                              dpf_sour.data / np.sqrt(cts_sour.data))
    flux_peak = unumpy.uarray(dpf_peak.data,
                              dpf_peak.data / np.sqrt(cts_peak.data))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dpf_ratio = np.array(list(map(nominal_value, flux_peak / flux_sour)))
        err_ratio = np.array(list(map(std_dev, flux_peak / flux_sour)))

    return dpf_ratio, err_ratio


def main(args):
    # Read time intervals
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # %% Define data path and time interval
    tint = cfg["tints"][int(args.t_id)]

    # %% Load magnetic and electric field
    b_gse, e_gse = load_eb_mmsx(tint, cfg)
    b_gsm, e_gsm = [cotrans(field, "gse>gsm") for field in [b_gse, e_gse]]

    # %% Load EIS proton omni-directional differential particle flux.
    _, cts_eis_omni_p = load_eis_omni_mmsx("proton", tint, cfg)
    dpf_allt_p, _ = load_eis_allt_mmsx("proton", tint, cfg)

    # %% Load EIS alphas omni-directional differential particle flux.
    _, cts_eis_omni_a = load_eis_omni_mmsx("alpha", tint, cfg)
    dpf_allt_a, _ = load_eis_allt_mmsx("alpha", tint, cfg)

    # %%
    dpf_skymap_p_mms = [None for _ in range(3)]
    dpf_skymap_a_mms = [None for _ in range(3)]

    for (i, dpf_p), dpf_a in zip(enumerate(dpf_allt_p), dpf_allt_a):
        dpf_skymap_p_mms[i] = eis_skymap(eis_ang_ang(dpf_p), to_psd=False)
        dpf_skymap_a_mms[i] = eis_skymap(eis_ang_ang(dpf_a), to_psd=False)

    dpf_eis_p = eis_skymap_combine_sc(dpf_skymap_p_mms)
    dpf_eis_a = eis_skymap_combine_sc(dpf_skymap_a_mms)

    dpf_eis_p.phi.data += 180.
    dpf_eis_a.phi.data += 180.
    dpf_eis_p.theta.data += 90.
    dpf_eis_a.theta.data += 90.

    # %%
    dpf_eis_omni_p = vdf_omni(dpf_eis_p)
    dpf_eis_omni_a = vdf_omni(dpf_eis_a)
    energy_hydrogen = dpf_eis_omni_p.energy.data * 1e-3
    dpf_eis_omni_p = dpf_eis_omni_p.assign_coords(energy=energy_hydrogen)

    energy_helium = dpf_eis_omni_a.energy.data * 1e-3
    dpf_eis_omni_a = dpf_eis_omni_a.assign_coords(energy=energy_helium)

    # %% Load alpha particles omni-directional differential particle flux.
    # _, dpf_hpca_omni_a = load_hpca_flux_mmsx(tint, cfg)
    _, dpf_hpca_omni_a = load_hpca_flux(tint, 2, cfg)
    dpf_hpca_omni_a.data *= 1e3

    t_pre = dpf_eis_omni_p.time.data[0]
    t_post = dpf_eis_omni_p.time.data[np.argmax(dpf_eis_omni_p.data[:, 6])]
    dt_eis = np.timedelta64(int(calc_dt(dpf_eis_omni_p)), "s")
    tint_pre = np.array([t_pre, t_pre + 2 * dt_eis])
    tint_pre = list(datetime642iso8601(tint_pre))
    tint_post = np.array([t_post - dt_eis / 2, t_post + dt_eis / 2])
    tint_post = list(datetime642iso8601(tint_post))

    dpf_eis_peak_p = time_clip(dpf_eis_omni_p, tint_post)[0, :]
    cts_eis_peak_p = time_clip(cts_eis_omni_p, tint_post)[0, :]
    err_eis_peak_p = dpf_eis_peak_p / np.sqrt(cts_eis_peak_p.data)

    dpf_eis_peak_a = time_clip(dpf_eis_omni_a, tint_post)[0, :]
    cts_eis_peak_a = time_clip(cts_eis_omni_a, tint_post)[0, :]
    err_eis_peak_a = dpf_eis_peak_a / np.sqrt(cts_eis_peak_a.data)

    dpf_hpca_peak_a = time_clip(dpf_hpca_omni_a, tint_post)[0, :]
    dpf_hpca_peak_a.data[dpf_hpca_peak_a.data == 0.] = np.nan

    dpf_eis_sour_p = time_clip(dpf_eis_omni_p, tint_pre)
    dpf_eis_sour_p = dpf_eis_sour_p.mean(axis=0, skipna=True)
    cts_eis_sour_p = time_clip(cts_eis_omni_p, tint_pre)
    cts_eis_sour_p = cts_eis_sour_p.sum(axis=0, skipna=True)
    err_eis_sour_p = dpf_eis_sour_p / np.sqrt(cts_eis_sour_p.data)

    dpf_eis_sour_a = time_clip(dpf_eis_omni_a, tint_pre)
    dpf_eis_sour_a = dpf_eis_sour_a.mean(axis=0, skipna=True)
    cts_eis_sour_a = time_clip(cts_eis_omni_a, tint_pre)
    cts_eis_sour_a = cts_eis_sour_a.sum(axis=0, skipna=True)
    err_eis_sour_a = dpf_eis_sour_a / np.sqrt(cts_eis_sour_a.data)

    dpf_hpca_sour_a = time_clip(dpf_hpca_omni_a, tint_pre)
    dpf_hpca_sour_a = dpf_hpca_sour_a.mean(axis=0, skipna=True)
    dpf_hpca_sour_a.data[dpf_hpca_sour_a.data == 0.] = np.nan

    # Compute post/pre-enhancement flux ratio.
    dpf_ratio_p, err_ratio_p = calc_ratio(dpf_eis_sour_p, dpf_eis_peak_p,
                                          cts_eis_sour_p, cts_eis_peak_p)
    dpf_ratio_a, err_ratio_a = calc_ratio(dpf_eis_sour_a, dpf_eis_peak_a,
                                          cts_eis_sour_a, cts_eis_peak_a)

    # %%
    tint_pre_plot = iso86012datetime64(np.array(tint_pre))
    dt_eis_2 = np.timedelta64(int(1e9 * dt_eis.astype(int) / 2), "ns")
    tint_pre_plot += [-dt_eis_2, dt_eis_2]
    tint_pre_plot = datetime642iso8601(np.array(tint_pre_plot))

    idx_p = np.where(np.logical_and(dpf_eis_sour_p.energy.data > 50.,
                                    ~np.isnan(dpf_eis_sour_p.data)))[0]
    slope_sour_p = np.polyfit(np.log10(dpf_eis_sour_p.energy.data[idx_p]),
                              np.log10(dpf_eis_sour_p.data[idx_p]), 1)[0]
    idx_p = np.where(np.logical_and(dpf_eis_peak_p.energy.data > 50.,
                                    ~np.isnan(dpf_eis_peak_p.data)))[0]
    slope_peak_p = np.polyfit(np.log10(dpf_eis_peak_p.energy.data[idx_p]),
                              np.log10(dpf_eis_peak_p.data[idx_p]), 1)[0]

    idx_a = np.where(np.logical_and(dpf_eis_sour_a.energy.data > 2 * 50.,
                                    ~np.isnan(dpf_eis_sour_a.data)))[0]
    slope_sour_a = np.polyfit(np.log10(dpf_eis_sour_a.energy.data[idx_a]),
                              np.log10(dpf_eis_sour_a.data[idx_a]), 1)[0]
    idx_a = np.where(np.logical_and(dpf_eis_peak_a.energy.data > 2 * 50.,
                                    ~np.isnan(dpf_eis_peak_a.data)))[0]
    slope_peak_a = np.polyfit(np.log10(dpf_eis_peak_a.energy.data[idx_a]),
                              np.log10(dpf_eis_peak_a.data[idx_a]), 1)[0]
    gamma_p = f"$\gamma_{{s.}}$ = {slope_sour_p:2.1f}\n" \
              f"$\gamma_{{e.}}$ = {slope_peak_p:2.1f}"
    gamma_a = f"$\gamma_{{s.}}$ = {slope_sour_a:2.1f}\n" \
              f"$\gamma_{{e.}}$ = {slope_peak_a:2.1f}"


    # %%
    fig = plt.figure(figsize=(10.2, 13.4))
    gsp = fig.add_gridspec(15, 2, top=.95, bottom=.05, left=.12, right=.88,
                           hspace=0.1)

    gsp0 = gsp[:6, :].subgridspec(3, 1, hspace=0)
    gsp1 = gsp[7:, :].subgridspec(2, 2, hspace=0.3, wspace=.3)

    # Create axes in the grid spec
    axs00 = [fig.add_subplot(gsp0[i]) for i in range(3)]
    axs10 = [fig.add_subplot(gsp1[i, 0]) for i in range(2)]
    axs11 = [fig.add_subplot(gsp1[i, 1]) for i in range(2)]

    plot_line(axs00[0], b_gsm)
    plot_line(axs00[0], norm(b_gsm))
    axs00[0].legend(["$B_x$", "$B_y$", "$B_z$", "|B|"], frameon=True, ncol=4,
                    loc="upper right")
    axs00[0].set_ylabel("$B$" + "\n" + "[nT]")
    axs00[0].set_ylim([-8, 15])

    axs00[1], caxs001 = plot_spectr(axs00[1], dpf_eis_omni_p, yscale="log",
                                    cscale="log")
    axs00[1].set_ylim([15, 300])
    axs00[1].set_ylabel("$K_{H^{+}}$" + "\n" + "[keV]")
    caxs001.set_ylabel("Diff. Flux H$^{+}$" + "\n"
                       + "[(cm$^{2}$ s sr keV)$^{-1}$]")

    axs00[2], caxs002 = plot_spectr(axs00[2], dpf_eis_omni_a, yscale="log",
                                    cscale="log")
    axs00[2].set_ylim([60, 700])
    axs00[2].set_ylabel("$K_{He^{n+}}$" + "\n" + "[keV]")
    caxs002.set_ylabel("Diff. Flux He$^{n+}$" + "\n"
                       + "[(cm$^{2}$ s sr keV)$^{-1}$]")

    axs00[-1].get_shared_x_axes().join(*axs00)

    fig.align_ylabels(axs00)

    for ax in axs00[:-1]:
        ax.xaxis.set_ticklabels([])

    axs00[-1].set_xlim(mdates.datestr2num(tint))
    # Flux versus energy

    # Proton
    # Pre enhancement
    err_hp_s = axs10[0].errorbar(dpf_eis_sour_p.energy.data,
                                 dpf_eis_sour_p.data, err_eis_sour_p,
                                 color="tab:blue", linestyle="--", capsize=4)

    # Post enhancement
    err_hp_e = axs10[0].errorbar(dpf_eis_peak_p.energy.data,
                                 dpf_eis_peak_p.data, err_eis_peak_p,
                                 color="tab:blue", linestyle="-", capsize=4)

    # Helium
    # Pre enhancement
    err_he_s = axs10[0].errorbar(dpf_eis_sour_a.energy.data,
                                 dpf_eis_sour_a.data, err_eis_sour_a,
                                 capsize=4, color="tab:red", linestyle="--")

    # Post enhancement
    err_he_e = axs10[0].errorbar(dpf_eis_peak_a.energy.data,
                                dpf_eis_peak_a.data, err_eis_peak_a,
                                 color="tab:red", linestyle="-", capsize=4)

    axs10[0].plot(dpf_hpca_sour_a.energy.data / 1e3, dpf_hpca_sour_a.data,
                  linestyle="--", color="tab:red")
    axs10[0].plot(dpf_hpca_peak_a.energy.data / 1e3, dpf_hpca_peak_a.data,
                  linestyle="-", color="tab:red")

    axs10[0].set_xlim([1e1, 1e3])
    axs10[0].set_xlabel("$K$ [keV]")
    axs10[0].set_ylim([3e-3, 3e5])
    axs10[0].set_ylabel("Diff. Flux" + "\n" + "[(cm$^2$ s sr keV)$^{-1}$]")
    l00 = axs10[0].legend([(err_hp_s, err_he_s), (err_hp_e, err_he_e)],
                          ["Source (s.)", "Energized (e.)"], loc="lower left",
                          title="Population", frameon=True,
                          handler_map={
                              ErrorbarContainer: HandlerErrorbar(numpoints=1),
                              tuple: HandlerTuple(ndivide=None)})
    axs10[0].add_artist(l00)
    l01 = axs10[0].legend([(err_hp_s, err_hp_e), (err_he_s, err_he_e)],
                          ["H$^{+}$", "He$^{2+}$"], loc="upper right",
                          title="Species", frameon=True,
                          handler_map={
                              ErrorbarContainer: HandlerErrorbar(numpoints=1),
                              tuple: HandlerTuple(ndivide=None)})

    # Flux versus energy per charge
    # Proton
    # Pre enhancement
    err_hp_s = axs10[1].errorbar(dpf_eis_sour_p.energy.data / 1,
                                 dpf_eis_sour_p.data, err_eis_sour_p,
                                 color="tab:blue", linestyle="--",
                                 capsize=4)
    # Post enhancement
    err_hp_e = axs10[1].errorbar(dpf_eis_peak_p.energy.data / 1,
                                 dpf_eis_peak_p.data, err_eis_peak_p,
                                 color="tab:blue", linestyle="-", capsize=4)

    # Helium
    # Pre enhancement
    err_he_s = axs10[1].errorbar(dpf_eis_sour_a.energy.data / 2,
                                 dpf_eis_sour_a.data, err_eis_sour_a,
                                 color="tab:red", linestyle="--", capsize=4)
    # Post enhancement
    err_he_e = axs10[1].errorbar(dpf_eis_peak_a.energy.data / 2,
                                 dpf_eis_peak_a.data, err_eis_peak_a,
                                 color="tab:red", linestyle="-", capsize=4)

    axs10[1].plot(dpf_hpca_sour_a.energy.data / 2e3, dpf_hpca_sour_a.data,
                  linestyle="--", color="tab:red")
    axs10[1].plot(dpf_hpca_peak_a.energy.data / 2e3, dpf_hpca_peak_a.data,
                  linestyle="-", color="tab:red")

    axs10[1].set_xlim([1e1, 1e3])
    axs10[1].set_xlabel("$K / q$ [keV / q]")
    axs10[1].set_ylim([3e-3, 3e5])
    axs10[1].set_ylabel("Diff. Flux" + "\n" + "[(cm$^2$ s sr keV)$^{-1}$]")
    axs10[1].text(3.2e2, 1e1, gamma_p, color="tab:blue")
    axs10[1].text(3.2e2, 1e-1, gamma_a, color="tab:red")
    l10 = axs10[1].legend([(err_hp_s, err_he_s), (err_hp_e, err_he_e)],
                          ["Source (s.)", "Energized (e.)"], loc="lower left",
                          title="Population", frameon=True,
                          handler_map={
                              ErrorbarContainer: HandlerErrorbar(numpoints=1),
                              tuple: HandlerTuple(ndivide=None)})
    axs10[1].add_artist(l10)
    l11 = axs10[1].legend([(err_hp_s, err_hp_e), (err_he_s, err_he_e)],
                          ["H$^{+}$", "He$^{2+}$"], loc="upper right",
                          title="Species", frameon=True,
                          handler_map={
                              ErrorbarContainer: HandlerErrorbar(numpoints=1),
                              tuple: HandlerTuple(ndivide=None)})

    # Flux ratio versus energy
    # Proton
    axs11[0].errorbar(dpf_eis_peak_p.energy.data, dpf_ratio_p, err_ratio_p,
                      capsize=4, color="tab:blue", linestyle="-",
                      label="H$^{+}$")

    # Helium
    axs11[0].errorbar(dpf_eis_peak_a.energy.data, dpf_ratio_a, err_ratio_a,
                      capsize=4, color="tab:red", linestyle="-",
                      label="He$^{2+}$")

    axs11[0].set_xlim([0, 700])
    axs11[0].set_ylim([1e-1, 1e3])
    axs11[0].set_xlabel("$K$ [keV]")
    axs11[0].set_ylabel("Flux ratio (e./s.)")
    axs11[0].legend(title="Species", loc="upper right", frameon=True)

    # Flux ratio versus energy per charge
    # Proton
    axs11[1].errorbar(dpf_eis_peak_p.energy.data / 1, dpf_ratio_p, err_ratio_p,
                      capsize=4, color="tab:blue", linestyle="-",
                      label="H$^{+}$")

    # Helium
    axs11[1].errorbar(dpf_eis_peak_a.energy.data / 2, dpf_ratio_a, err_ratio_a,
                      capsize=4, color="tab:red", linestyle="-",
                      label="He$^{2+}$")

    axs11[1].set_xlim([0, 350])
    axs11[1].set_ylim([1e-1, 1e3])
    axs11[1].set_xlabel("$K/q$ [keV/q]")
    axs11[1].set_ylabel("Flux ratio (e./s.)")
    axs11[1].legend(title="Species", loc="upper right", frameon=True)

    for i in range(2):
        axs10[i].set_xscale("log")
        axs10[i].set_yscale("log")
        axs10[i].grid(which="major")
        axs11[i].set_yscale("log")
        axs11[i].grid(which="major")

    make_labels(axs00, [0.01, .87], pad=0)
    make_labels(axs10, [0.025, .92], pad=3)
    make_labels(axs11, [0.025, .92], pad=5)

    span_tint([axs00[0]], tint_pre_plot, facecolor="tab:orange",
              linestyle="none", alpha=.2)
    span_tint([axs00[0]], tint_post, facecolor="tab:olive", linestyle="none",
              alpha=.2)
    span_tint(axs00, tint_pre_plot, facecolor="none", linestyle="--",
              edgecolor="k")
    span_tint(axs00, tint_post, facecolor="none", linestyle="-", edgecolor="k")
    axs00[0].text((mdates.datestr2num(tint_pre_plot[1])
                   + mdates.datestr2num(tint_pre_plot[0])) / 2, 13, "s.",
                  ha="center", color="k")
    axs00[0].text((mdates.datestr2num(tint_post[1]) + mdates.datestr2num(
        tint_post[0])) / 2, 13, "e.", ha="center", color="k")

    fig.align_ylabels([*axs00, *axs10])

    fig.suptitle(f"Event {'I' * (args.t_id + 1)} ({tint[0][:-7]} - {tint[1][:-7]})")
    plt.savefig(f"./figures/figure_{args.t_id + 4}.pdf")
    plt.savefig(f"./figures/figure_{args.t_id + 4}.png", dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        help="Path to the configuration file (.yml)",
                        required=True, type=str)
    parser.add_argument("--t-id", help="Time interval index", required=True,
                        type=int)
    main(parser.parse_args())
