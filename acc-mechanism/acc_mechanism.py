#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import argparse
import warnings

# 3rd party imports
import yaml
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from uncertainties import ufloat
from scipy import interpolate, constants
from pyrfu import mms, pyrf
from pyrfu.plot import plot_line, span_tint, make_labels

from jfs.load import load_eb_mmsx, load_fpi_moments_mmsx, load_eis_allt_mmsx

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2021"
__license__ = "Apache 2.0"

tints_brst = [["2017-07-23T16:54:14.000000", "2017-07-23T16:59:25.745247"],
              ["2017-07-23T17:17:11.000000", "2017-07-23T17:22:36.000000"]]

plt.style.use("scientific")

m_p = constants.proton_mass
q_e = constants.elementary_charge
cel = constants.speed_of_light


def _rho_p(t_i, b_si):
    v_tp = cel * np.sqrt(1 - 1 / (t_i * q_e / (m_p * cel ** 2) + 1) ** 2)
    gamma_p = 1 / np.sqrt(1 - (v_tp / cel) ** 2)
    rho_p = m_p * cel / (q_e * b_si * 1e-9) * np.sqrt(gamma_p ** 2 - 1)
    return rho_p


def mod_resonnance(t_i, b_si, l_):
    return _rho_p(t_i, b_si) - l_


def mod_pulse(t_i, b_si, e_y):
    return 2 * e_y * 1e-3 * _rho_p(t_i, b_si)


def _delta_k(inp_sour, err_sour, inp_peak, err_peak):
    vdf_peak = inp_peak.data[inp_peak.energy.data < 250e3]
    err_peak = err_peak.data[inp_peak.energy.data < 250e3]
    ene_peak = inp_peak.energy.data[inp_peak.energy.data < 250e3] / 1e3

    vdf_peak_avg = vdf_peak.copy()
    vdf_peak_min = vdf_peak.copy() - err_peak.copy()
    vdf_peak_max = vdf_peak.copy() + err_peak.copy()

    vdf_sour_avg = inp_sour.data.copy()
    vdf_sour_min = inp_sour.data.copy() - err_sour.data.copy()
    vdf_sour_max = inp_sour.data.copy() + err_sour.data.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_avg = interpolate.interp1d(np.log10(vdf_peak_avg), ene_peak,
                                     fill_value="extrapolate")
        f_min = interpolate.interp1d(np.log10(vdf_peak_min), ene_peak,
                                     fill_value="extrapolate")
        f_max = interpolate.interp1d(np.log10(vdf_peak_max), ene_peak,
                                     fill_value="extrapolate")

        ene_peak_avg_ = f_avg(np.log10(vdf_sour_avg))
        ene_peak_min_ = f_min(np.log10(vdf_sour_max))
        ene_peak_max_ = f_max(np.log10(vdf_sour_min))

    delta_ene_avg = ene_peak_avg_ - inp_sour.energy.data / 1e3
    delta_ene_min = ene_peak_min_ - inp_sour.energy.data / 1e3
    delta_ene_max = ene_peak_max_ - inp_sour.energy.data / 1e3

    indices_energ = np.where(
        np.logical_and(~np.isnan(delta_ene_max), ~np.isnan(delta_ene_min)))[0]
    delta_ene_err = [
        np.abs(delta_ene_min[indices_energ] - delta_ene_avg[indices_energ]),
        np.abs(delta_ene_max[indices_energ] - delta_ene_avg[indices_energ])]
    delta_ene_avg = delta_ene_avg[indices_energ]
    delta_ene_ene = inp_sour.energy.data[indices_energ] / 1e3

    ene_peak_ = xr.DataArray(ene_peak_avg_, coords=[inp_sour.energy.data],
                             dims=["energy"])
    delta_ene = xr.DataArray(delta_ene_avg, coords=[delta_ene_ene],
                             dims=["energy"],
                             attrs={"err": delta_ene_err})

    return delta_ene, ene_peak_


def main(args):
    # Read time intervals
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    b_gse, e_gse = load_eb_mmsx(tints_brst[args.t_id], cfg)
    b_gsm, e_gsm = [pyrf.cotrans(field, "gse>gsm") for field in [b_gse, e_gse]]

    moms_i_mmsx, _ = load_fpi_moments_mmsx(tints_brst[args.t_id], cfg)
    _, v_gse_i, _, _ = moms_i_mmsx
    v_gsm_i = pyrf.cotrans(v_gse_i, "gse>gsm")

    dpf_allt_p, cts_allt_p = load_eis_allt_mmsx("proton",
                                                tints_brst[args.t_id], cfg)

    cts_omni_mms = [None for _ in range(3)]
    dpf_skymap_mms = [None for _ in range(3)]
    cts_skymap_mms = [None for _ in range(3)]

    for (i, dpf), cts in zip(enumerate(dpf_allt_p), cts_allt_p):
        cts_omni_mms[i] = mms.eis_omni(mms.eis_spin_avg(cts, method="sum"),method="sum")
        dpf_skymap_mms[i] = mms.eis_skymap(mms.eis_ang_ang(dpf))
        cts_skymap_mms[i] = mms.eis_skymap(mms.eis_ang_ang(cts), to_psd=False)

    cts_omni_p = mms.eis_spec_combine_sc(cts_omni_mms, method="sum")
    dpf_skymap = mms.eis_skymap_combine_sc(dpf_skymap_mms)
    cts_skymap = mms.eis_skymap_combine_sc(cts_skymap_mms, method="sum")

    vdf_eis_i = dpf_skymap.copy()
    vdf_eis_i.phi.data += 180.
    vdf_eis_i.theta.data += 90.

    vdf_eis_i_sfrm = vdf_eis_i
    vdf_eis_i_ifrm = mms.vdf_frame_transformation(vdf_eis_i_sfrm, 1e3 * v_gse_i)

    vdf_eis_omni_i_sfrm = mms.vdf_omni(vdf_eis_i_sfrm)
    vdf_eis_omni_i_ifrm = mms.vdf_omni(vdf_eis_i_ifrm)

    cts_eis_i = cts_skymap.copy()
    cts_eis_i.phi.data += 180.
    cts_eis_i.theta.data += 90.

    cts_eis_i_sfrm = cts_eis_i
    cts_eis_i_ifrm = mms.vdf_frame_transformation(cts_eis_i_sfrm,
                                                  1e3 * v_gse_i)

    cts_eis_omni_i_sfrm = mms.vdf_omni(cts_eis_i_sfrm, method="sum")
    cts_eis_omni_i_sfrm.data = np.floor(cts_eis_omni_i_sfrm.data)
    cts_eis_omni_i_ifrm = mms.vdf_omni(cts_eis_i_ifrm, method="sum")
    cts_eis_omni_i_ifrm.data = np.floor(cts_eis_omni_i_ifrm.data)

    dt_eis = np.timedelta64(int(pyrf.calc_dt(vdf_eis_omni_i_sfrm)), "s")

    # Source distribution at the beginning of the brst interval (furthest from
    # the front)
    t_pre = vdf_eis_omni_i_sfrm.time.data[0]
    tint_pre = np.array([t_pre, t_pre + dt_eis])
    tint_pre = list(pyrf.datetime642iso8601(tint_pre))

    # Post-enhancement distribution (peak or hard coded)
    idx_peak = np.argmax(vdf_eis_omni_i_sfrm.data[:, 6])
    t_post = vdf_eis_omni_i_sfrm.time.data[idx_peak]  # peak
    tint_post = np.array([t_post - dt_eis / 2, t_post + dt_eis / 2])
    tint_post = list(pyrf.datetime642iso8601(tint_post))

    tint_pre_plot = pyrf.iso86012datetime64(np.array(tint_pre))
    tint_pre_plot += [-np.timedelta64(int(1e9 * dt_eis.astype(int) / 2), "ns"),
                      np.timedelta64(int(1e9 * dt_eis.astype(int) / 2), "ns")]
    tint_pre_plot = pyrf.datetime642iso8601(np.array(tint_pre_plot))

    vdf_eis_omni_sour_sfrm = pyrf.time_clip(vdf_eis_omni_i_sfrm, tint_pre)
    vdf_eis_omni_sour_sfrm = vdf_eis_omni_sour_sfrm.mean(axis=0, skipna=True)
    vdf_eis_omni_peak_sfrm = vdf_eis_omni_i_sfrm.sel(time=t_post)

    vdf_eis_omni_sour_ifrm = pyrf.time_clip(vdf_eis_omni_i_ifrm, tint_pre)
    vdf_eis_omni_sour_ifrm = vdf_eis_omni_sour_ifrm.mean(axis=0, skipna=True)
    vdf_eis_omni_peak_ifrm = vdf_eis_omni_i_ifrm.sel(time=t_post)

    cts_eis_omni_sour_sfrm = pyrf.time_clip(cts_eis_omni_i_sfrm, tint_pre)
    cts_eis_omni_sour_sfrm = cts_eis_omni_sour_sfrm.sum(axis=0, skipna=True)
    cts_eis_omni_peak_sfrm = cts_eis_omni_i_sfrm.sel(time=t_post)

    cts_eis_omni_sour_ifrm = pyrf.time_clip(cts_eis_omni_i_ifrm, tint_pre)
    cts_eis_omni_sour_ifrm = cts_eis_omni_sour_ifrm.sum(axis=0, skipna=True)
    cts_eis_omni_peak_ifrm = cts_eis_omni_i_ifrm.sel(time=t_post)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        erd_eis_omni_sour_sfrm = vdf_eis_omni_sour_sfrm.copy()
        erd_eis_omni_peak_sfrm = vdf_eis_omni_peak_sfrm.copy()
        erd_eis_omni_sour_sfrm.data /= np.sqrt(cts_eis_omni_sour_sfrm.data)
        erd_eis_omni_peak_sfrm.data /= np.sqrt(cts_eis_omni_peak_sfrm.data)

        erd_eis_omni_sour_ifrm = vdf_eis_omni_sour_ifrm.copy()
        erd_eis_omni_peak_ifrm = vdf_eis_omni_peak_ifrm.copy()
        erd_eis_omni_sour_ifrm.data /= np.sqrt(cts_eis_omni_sour_ifrm.data)
        erd_eis_omni_peak_ifrm.data /= np.sqrt(cts_eis_omni_peak_ifrm.data)

    delta_en_sfrm, e_peak_sfrm = _delta_k(vdf_eis_omni_sour_sfrm,
                                          erd_eis_omni_sour_sfrm,
                                          vdf_eis_omni_peak_sfrm,
                                          erd_eis_omni_peak_sfrm)

    delta_en_ifrm, e_peak_ifrm = _delta_k(vdf_eis_omni_sour_ifrm,
                                          erd_eis_omni_sour_ifrm,
                                          vdf_eis_omni_peak_ifrm,
                                          erd_eis_omni_peak_ifrm)

    fig = plt.figure(figsize=(15, 10))
    gsp = fig.add_gridspec(20, 2, top=.95, bottom=.06, left=.08, right=.92,
                           hspace=0.)

    gsp0 = gsp[:9, :].subgridspec(3, 1, hspace=0)
    gsp1 = gsp[11:, :].subgridspec(1, 3, hspace=0.3, wspace=.3)

    # Create axes in the grid spec
    axs00 = [fig.add_subplot(gsp0[i]) for i in range(3)]
    axs10 = [fig.add_subplot(gsp1[i]) for i in range(3)]

    plot_line(axs00[0], b_gsm)
    plot_line(axs00[0], pyrf.norm(b_gsm))
    axs00[0].legend(["$B_x$", "$B_y$", "$B_z$", "|B|"], frameon=True, ncol=4,
                    loc="upper right")
    axs00[0].set_ylabel("$B$" + "\n" + "[nT]")
    axs00[0].set_ylim([-8, 15])

    plot_line(axs00[1], e_gsm[:, 1], label="$E_y$")
    plot_line(axs00[1], -1e-3 * pyrf.cross(v_gsm_i, b_gsm)[:, 1],
              label="$-(V_i\\times B)_y$")
    axs00[1].set_ylabel("$E_y$" + "\n" + "[mV m$^{-1}$]")
    axs00[1].legend(frameon=True, ncol=3, loc="upper right")
    axs00[1].set_ylim([-7, 8])

    plot_line(axs00[2], v_gsm_i)
    axs00[2].legend(["$V_{xi}$", "$V_{yi}$", "$V_{zi}$"], frameon=True,
                    ncol=3, loc="upper right")
    axs00[2].set_ylabel("$V_{p}$" + "\n" + "[km s$^{-1}$]")
    span_tint(axs00, tint_pre_plot, facecolor="tab:orange", linestyle="--",
              edgecolor="k", alpha=.2)
    span_tint(axs00, tint_post, facecolor="tab:olive", linestyle="-",
              edgecolor="k", alpha=.2)
    axs00[0].text((mdates.datestr2num(tint_pre_plot[1])
                   + mdates.datestr2num(tint_pre_plot[0])) / 2, 13, "s.",
                  ha="center", color="k")
    axs00[0].text((mdates.datestr2num(tint_post[1]) + mdates.datestr2num(
        tint_post[0])) / 2, 13, "e.", ha="center", color="k")

    axs00[-1].get_shared_x_axes().join(*axs00)

    for ax in axs00[:-1]:
        ax.xaxis.set_ticklabels([])
    axs00[-1].set_xlim(mdates.datestr2num(tints_brst[args.t_id]))

    fig.align_ylabels(axs00)

    axs10[0].errorbar(vdf_eis_omni_sour_sfrm.energy.data / 1e3,
                      vdf_eis_omni_sour_sfrm.data, erd_eis_omni_sour_sfrm,
                      color="tab:orange", linestyle="-", marker="s", zorder=2,
                      capsize=4, label="Source (s.)")

    axs10[0].errorbar(vdf_eis_omni_peak_sfrm.energy.data / 1e3,
                      vdf_eis_omni_peak_sfrm.data, erd_eis_omni_peak_sfrm,
                      color="tab:olive", linestyle="-", marker="s", zorder=1,
                      capsize=4, label="Energized (e.)")

    axs10[0].legend(loc="upper right", title="Spacecraft frame", frameon=True)

    axs10[0].fill_betweenx(vdf_eis_omni_sour_sfrm.data,
                           vdf_eis_omni_sour_sfrm.energy.data / 1e3,
                           e_peak_sfrm.data, color="tab:cyan", alpha=.3,
                           zorder=0)

    axs10[0].set_xlim([0, 250])
    axs10[0].set_ylim([5e-22, 5e-15])
    axs10[0].set_xlabel("$K$ [keV]")
    axs10[0].set_ylabel("$f$ [s$^3$ m$^{-6}$]")
    axs10[0].set_yscale("log")
    # axs10[0].set_title("Spacecraft frame")

    axs10[1].errorbar(vdf_eis_omni_sour_ifrm.energy.data / 1e3,
                      vdf_eis_omni_sour_ifrm.data, erd_eis_omni_sour_ifrm,
                      color="tab:orange", linestyle="-", marker="s", zorder=2,
                      capsize=4, label="Source (s.)")

    axs10[1].errorbar(vdf_eis_omni_peak_ifrm.energy.data / 1e3,
                      vdf_eis_omni_peak_ifrm.data, erd_eis_omni_peak_ifrm,
                      color="tab:olive", linestyle="-", marker="s",
                      zorder=1, capsize=4, label="Energized (e.)")

    axs10[1].legend(loc="upper right", title="Proton bulk frame", frameon=True)

    axs10[1].fill_betweenx(vdf_eis_omni_sour_ifrm.data,
                           vdf_eis_omni_sour_ifrm.energy.data / 1e3,
                           e_peak_ifrm, color="tab:purple", alpha=.3, zorder=0)
    # axs10[1].legend([l0, l1], ["s.", "e."], loc="upper right",
    #                 frameon=True)

    axs10[1].set_xlim([0, 250])
    axs10[1].set_ylim([5e-22, 5e-15])
    axs10[1].set_xlabel("$K$ [keV]")
    axs10[1].set_ylabel("$f$ [s$^3$ m$^{-6}$]")
    axs10[1].set_yscale("log")
    # axs10[1].set_title("Proton bulk frame")

    axs10[2].errorbar(delta_en_sfrm.energy.data, delta_en_sfrm.data,
                      delta_en_sfrm.attrs["err"], marker="s", capsize=4,
                      color="tab:cyan", label="Spacecraft frame")
    axs10[2].errorbar(delta_en_ifrm.energy.data, delta_en_ifrm.data,
                      delta_en_ifrm.attrs["err"], marker="s", capsize=4,
                      color="tab:purple", label="Proton bulk frame")
    axs10[2].set_xlim([0, 250])

    if args.t_id == 1:
        b_max = np.max(pyrf.norm(pyrf.time_clip(b_gsm, tint_post)).data)

        axs10[2].plot(np.linspace(0, 200),
                      mod_pulse(np.linspace(0, 200) * 1e3, 9.64, 7.18) / 1e3,
                      color="k", linestyle="-.",
                      label="$\\delta K = 2eE_y\\rho_p$")
        axs10[2].set_ylim([0, 250])

    elif args.t_id == 0:
        tint_front = ["2017-07-23T16:55:35", "2017-07-23T16:55:41"]
        v_avg = np.mean(pyrf.norm(pyrf.time_clip(v_gsm_i, tint_front)).data)
        v_std = np.std(pyrf.norm(pyrf.time_clip(v_gsm_i, tint_front)).data)
        v_bulk = ufloat(v_avg, v_std)
        b_max = np.max(pyrf.norm(pyrf.time_clip(b_gsm, tint_front)).data)

        e_std = np.std(pyrf.time_clip(e_gse, tint_pre)[:, 1]).data
        idx = np.where(np.abs(pyrf.time_clip(e_gse, tint_front)[:,
                              1]) > 3.16 * e_std)[0][[0, -1]]
        delta_t = np.diff(e_gse.time.data[idx])[0].astype(int) / 1e9

        e_ = (delta_t * v_bulk * b_max / 144) ** 2
        axs10[2].axvspan((e_.nominal_value - e_.std_dev) / 1e3,
                         (e_.nominal_value + e_.std_dev) / 1e3,
                         color="gold", alpha=.2)
        axs10[2].axvline(e_.nominal_value / 1e3, color="gold", linestyle="--")
        axs10[2].set_ylim([0, 50])

    else:
        raise ValueError("Invalid interval !!")

    axs10b = [None] * 3

    for ax, axb in zip(axs10, axs10b):
        axb = ax.twiny()
        axb.set_xticks(ax.get_xticks())
        rho_i = 144 * np.sqrt(ax.get_xticks() * 1e3) / b_max / 6371
        axb.set_xticklabels([f"{r:3.2f}" for r in rho_i])
        axb.set_xlabel("$\\rho_p$ [$R_E$]")

    axs10[2].set_xlabel("$K_0$ [keV]")
    axs10[2].set_ylabel("$\\delta K$ [keV]")
    axs10[2].legend(frameon=True, loc="upper right")
    axs10[2].set_xlim([0, 250])
    y_max = [50, 250]
    axs10[2].set_ylim([0, y_max[args.t_id]])
    make_labels(axs00, [.01, .85], 0)
    make_labels(axs10, [.025, .94], 3)

    fig.suptitle(f"Event {'I' * (args.t_id + 1)} ("
                 f"{tints_brst[args.t_id][0][:-7]} -"
                 f" {tints_brst[args.t_id][1][:-7]})")
    plt.savefig(f"./figures/figure_{args.t_id + 6}.pdf")
    plt.savefig(f"./figures/figure_{args.t_id + 6}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        help="Path to the configuration file (.yml)",
                        required=True, type=str)
    parser.add_argument("--t-id", help="Time interval index", required=True,
                        type=int)
    main(parser.parse_args())
