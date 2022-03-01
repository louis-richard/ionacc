#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Routines to load data from parameters stored in .yml files
"""

# 3rd party imports
import numpy as np

from scipy import constants

from pyrfu.pyrf import cotrans, avg_4sc
from pyrfu.mms import (get_data, remove_idist_background, get_feeps_alleyes,
                       feeps_correct_energies, feeps_omni, vdf_omni, psd2dpf,
                       feeps_flat_field_corrections, feeps_remove_bad_data,
                       feeps_split_integral_ch, feeps_remove_sun, eis_omni,
                       get_eis_allt, eis_spin_avg, eis_combine_proton_spec,
                       eis_spec_combine_sc, hpca_calc_anodes, db_init,
                       hpca_spin_sum)

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2021"
__license__ = "Apache 2.0"


def load_r_mmsx(tint: list, config: dict):
    r"""Load spacecraft locations, averages at the center of mass of the
    tetrahedron, and compute spacecraft separation

    Parameters
    ----------
    tint : list
        Time interval
    config : dict
        Loading configuration dictionary from .yml file.

    Returns
    -------
    r_gse_avg : numpy.ndarray
        Location of the tetrahedron's center of mass in GSE coordinates.
    r_gsm_avg : numpy.ndarray
        Location of the tetrahedron's center of mass in GSM coordinates.
    r_gsm_sep : numpy.ndarray
        Spacecraft separation in GSM coordinates.

    """

    assert config and isinstance(config, dict) and config.get("mec")

    # Create suffix with data rate and data level. If not filled default
    # values are SRVY mode and L2 data.
    data_rate = config["mec"].get("data_rate", "srvy")
    data_levl = config["mec"].get("level", "l2")
    suffx_mec = f"mec_{data_rate}_{data_levl}"

    r_gse_mms = [None for _ in range(4)]
    r_gsm_mms = [None for _ in range(4)]

    for i, ic in enumerate(range(1, 5)):
        r_gsm_mms[i] = get_data(f"r_gsm_{suffx_mec}", tint, ic,
                                data_path=config["data_path"])
        r_gse_mms[i] = cotrans(r_gsm_mms[i], "gsm>gse")

    r_gsm_mms_avg = [np.mean(r_xyz.data, axis=0) for r_xyz in r_gsm_mms]
    r_gsm_avg = np.mean(np.stack(r_gsm_mms_avg), axis=0)
    r_gsm_sep = [r_xyz - r_gsm_avg for r_xyz in r_gsm_mms_avg]

    r_gse_avg = np.mean(avg_4sc(r_gse_mms).data / 6371, axis=0)

    return r_gse_avg, r_gsm_avg, r_gsm_sep


def load_eb_mmsx(tint: list, config: dict):
    r"""Load magnetic and electric for all spacecraft and compute the
    average at the center of mass of the tetrahedron.

    Parameters
    ----------
    tint : list
        Time interval
    config : dict
        Loading configuration dictionary from .yml file.

    Returns
    -------
    b_gse_mmsx : xaray.DataArray
        Time series of the magnetic field in GSE coordinates at the center
        of mass of the tetrahedron.
    e_gse_mmsx : xarray.DataArray
        Timm series of the electric field in GSE coordinates at the center
        of mass of the tetrahedron.

    """

    assert config and isinstance(config, dict) and config.get("fgm")
    assert config and isinstance(config, dict) and config.get("edp")

    # Create suffix with data rate and data level. If not filled default
    # values are FAST mode and L2 data.
    data_rate_fgm = config["fgm"].get("data_rate", "srvy")
    data_rate_edp = config["edp"].get("data_rate", "fast")
    data_levl_fgm = config["fgm"].get("level", "l2")
    data_levl_edp = config["edp"].get("level", "l2")
    suffx_fgm = f"fgm_{data_rate_fgm}_{data_levl_fgm}"  # FGM
    suffx_edp = f"edp_{data_rate_edp}_{data_levl_edp}"  # EDP

    # Setup data path.
    db_init(config["data_path"])

    b_gse_mms = [None for _ in range(4)]
    e_gse_mms = [None for _ in range(4)]

    for i, ic in enumerate(range(1, 5)):
        b_gse_mms[i] = get_data(f"b_gse_{suffx_fgm}", tint, ic)
        e_gse_mms[i] = get_data(f"e_gse_{suffx_edp}", tint, ic)
        
    b_gse_mmsx = avg_4sc(b_gse_mms)
    e_gse_mmsx = avg_4sc(e_gse_mms)

    return b_gse_mmsx, e_gse_mmsx


def load_fpi_moments(tint, mms_id, config):
    r"""Load FPI-DIS and FPI-DES partial moments, split to remove
    photoelectron and remove penetrating radiations.

    Parameters
    ----------
    tint : list
        Time interval
    mms_id : int
        Spacecraft index
    config : dict
        Loading configuration dictionary from .yml file.

    Returns
    -------
    moments_i : list of xarray.DataArray
        FPI-DIS moments (number density, bulk velocity, temperature, pressure)
    moments_e : list of xarray.DataArray
        FPI-DES moments (number density, bulk velocity, temperature, pressure)

    Notes
    -----
    Temperature and pressure are tensors. Bulk velocities, temperature and
    pressure tensors are expressed in GSE coordinates.

    """

    assert config and isinstance(config, dict) and config.get("fpi")

    # Create suffix with data rate and data level. If not filled default
    # values are FAST mode and L2 data.
    data_rate = config["fpi"].get("data_rate", "fast")
    data_levl = config["fpi"].get("level", "l2")
    suffx_fpi = f"fpi_{data_rate}_{data_levl}"

    # Setup data path.
    db_init(config["data_path"])

    # Load partial moments
    partn_i = get_data(f"partni_{suffx_fpi}", tint, mms_id)
    partv_gse_i = get_data(f"partvi_gse_{suffx_fpi}", tint, mms_id)
    partt_gse_i = get_data(f"partti_gse_{suffx_fpi}", tint, mms_id)

    # Split partial momemnts
    n_i = partn_i[:, 13]
    v_gse_i = partv_gse_i[:, 13, ...]
    t_gse_i = partt_gse_i[:, 13, ...]
    p_gse_i = n_i.data[:, None, None] * t_gse_i
    p_gse_i.data *= 1e15 * constants.elementary_charge

    # Background radiation
    nbg_i = get_data(f"nbgi_{suffx_fpi}", tint, mms_id)
    pbg_i = get_data(f"pbgi_{suffx_fpi}", tint, mms_id)

    # Remove penetrating radiations
    moms_clean = remove_idist_background(n_i, v_gse_i, p_gse_i, nbg_i, pbg_i)
    n_i_clean, v_gse_i_clean, p_gse_i_clean = moms_clean

    # Compute scalar temperature from pressure tensor
    t_gse_i_clean = p_gse_i_clean / n_i_clean.data[:, None, None]
    t_gse_i_clean.data /= 1e15 * constants.elementary_charge

    # Remove extremely low density points
    v_gse_i = v_gse_i_clean[n_i_clean > .005, ...]
    t_gse_i = t_gse_i_clean[n_i_clean > .005, ...]
    n_i = n_i_clean[n_i_clean > .005]
    p_gse_i = n_i.data[:, None, None] * t_gse_i  # nPa
    p_gse_i.data *= 1e15 * constants.elementary_charge

    partn_e = get_data(f"partne_{suffx_fpi}", tint, mms_id)
    partv_gse_e = get_data(f"partve_gse_{suffx_fpi}", tint, mms_id)
    partt_gse_e = get_data(f"partte_gse_{suffx_fpi}", tint, mms_id)

    # Split partial moments
    n_e = partn_e[:, 9]
    v_gse_e = partv_gse_e[:, 9, ...]
    t_gse_e = partt_gse_e[:, 9, ...]

    # Compute scalar temperature and pressure
    p_gse_e = n_e.data[:, None, None] * t_gse_e  # nPa
    p_gse_e.data *= 1e15 * constants.elementary_charge

    moments_i = [n_i, v_gse_i, t_gse_i, p_gse_i]
    moments_e = [n_e, v_gse_e, t_gse_e, p_gse_e]

    return moments_i, moments_e


def load_fpi_moments_mmsx(tint, config):
    r"""Load FPI-DIS and FPI-DES partial moments for all spacecraft, split to
    remove photoelectron, remove penetrating radiations and average at the
    center of mass of the tetrahedron.

    Parameters
    ----------
    tint : list
        Time interval
    config : dict
        Loading configuration dictionary from .yml file.

    Returns
    -------
    moms_i_mmsx : list of xarray.DataArray
        FPI-DIS moments (number density, bulk velocity, temperature, pressure)
    moms_e_mmsx : list of xarray.DataArray
        FPI-DES moments (number density, bulk velocity, temperature, pressure)

    Notes
    -----
    Temperature and pressure are tensors. Bulk velocities, temperature and
    pressure tensors are expressed in GSE coordinates.

    """

    moms_i_mms = [[None for _ in range(4)] for _ in range(4)]
    moms_e_mms = [[None for _ in range(4)] for _ in range(4)]

    for i, ic in enumerate(range(1, 5)):
        moms_i_mms[i], moms_e_mms[i] = load_fpi_moments(tint, ic, config)

    # Number density
    n_i_mmsx = avg_4sc([probe[0] for probe in moms_i_mms])
    n_e_mmsx = avg_4sc([probe[0] for probe in moms_e_mms])

    # Bulk velocity
    v_gse_i_mmsx = avg_4sc([probe[1] for probe in moms_i_mms])
    v_gse_e_mmsx = avg_4sc([probe[1] for probe in moms_e_mms])

    # Temperature tensor
    t_gse_i_mmsx = avg_4sc([probe[2] for probe in moms_i_mms])
    t_gse_e_mmsx = avg_4sc([probe[2] for probe in moms_e_mms])

    # Pressure tensor
    p_gse_i_mmsx = avg_4sc([probe[3] for probe in moms_i_mms])
    p_gse_e_mmsx = avg_4sc([probe[3] for probe in moms_e_mms])

    moms_i_mmsx = [n_i_mmsx, v_gse_i_mmsx, t_gse_i_mmsx, p_gse_i_mmsx]
    moms_e_mmsx = [n_e_mmsx, v_gse_e_mmsx, t_gse_e_mmsx, p_gse_e_mmsx]

    return moms_i_mmsx, moms_e_mmsx


def load_hpca_moments(tint, mms_id, config):
    r""""Load HPCA proton (H+) and alpha particles (He2+) moments.

    Parameters
    ----------
    tint : list
        Time interval
    mms_id : int
        Spacecraft index
    config : dict
        Loading configuration dictionary from .yml file.

    Returns
    -------
    moments_p : list of xarray.DataArray
        H+ moments (number density, bulk velocity, temperature, pressure)
    moments_a : list of xarray.DataArray
        He2+ moments (number density, bulk velocity, temperature, pressure)

    Notes
    -----
    Temperature and pressure are scalars. Bulk velocities are expressed in GSE
    coordinates.

    """

    assert config and isinstance(config, dict) and config.get("hpca")

    # Create suffix with data rate and data level. If not filled default
    # values are SRVY mode and L2 data.
    data_rate_ = config["hpca"].get("data_rate", "srvy")
    data_level = config["hpca"].get("level", "l2")
    suffx_hpca = f"hpca_{data_rate_}_{data_level}"

    # Protons
    # Number density
    n_p = get_data(f"nhplus_{suffx_hpca}", tint, mms_id,
                   data_path=config["data_path"])

    # Bulk velocity in GSM coordinates
    v_gsm_p = get_data(f"vhplus_gsm_{suffx_hpca}", tint, mms_id,
                       data_path=config["data_path"])

    # Scalar temperature
    t_p = get_data(f"tshplus_{suffx_hpca}", tint, mms_id,
                   data_path=config["data_path"])

    # Scalar pressure
    p_p = 1e15 * constants.elementary_charge * n_p.data * t_p  # nPa

    # Alpha particles
    # Number density
    n_a = get_data(f"nheplusplus_{suffx_hpca}", tint, mms_id,
                   data_path=config["data_path"])

    # Bulk velocity in GSM coordinates
    v_gsm_a = get_data(f"vheplusplus_gsm_{suffx_hpca}", tint, mms_id,
                       data_path=config["data_path"])

    # Scalar temperature
    t_a = get_data(f"tsheplusplus_{suffx_hpca}", tint, mms_id,
                   data_path=config["data_path"])

    # Scalar pressure
    p_a = 1e15 * constants.elementary_charge * n_a.data * t_a  # nPa

    moments_p = [n_p, v_gsm_p, t_p, p_p]  # Proton
    moments_a = [n_a, v_gsm_a, t_a, p_a]  # Alpha

    return moments_p, moments_a


def load_hpca_moments_mmsx(tint, config):
    r""""Load HPCA proton (H+) and alpha particles (He2+) moments for all
    spacecraft and averages at the center of mass of the tetrahedron.

    Parameters
    ----------
    tint : list
        Time interval
    config : dict
        Loading configuration dictionary from .yml file.

    Returns
    -------
    moms_p_mmsx : list of xarray.DataArray
        H+ moments (number density, bulk velocity, temperature, pressure)
    moms_a_mmsx : list of xarray.DataArray
        He2+ moments (number density, bulk velocity, temperature, pressure)

    Notes
    -----
    Temperature and pressure are scalars. Bulk velocities are expressed in GSE
    coordinates.

    """

    moms_p_mms = [[None for _ in range(4)] for _ in range(4)]
    moms_a_mms = [[None for _ in range(4)] for _ in range(4)]

    for i, ic in enumerate(range(1, 5)):
        moms_p_mms[i], moms_a_mms[i] = load_hpca_moments(tint, ic, config)

    # Number density
    n_p_mmsx = avg_4sc([probe[0] for probe in moms_p_mms])
    n_a_mmsx = avg_4sc([probe[0] for probe in moms_a_mms])

    # Bulk velocity
    v_gsm_p_mmsx = avg_4sc([probe[1] for probe in moms_p_mms])
    v_gsm_a_mmsx = avg_4sc([probe[1] for probe in moms_a_mms])

    # Temperature tensor
    t_p_mmsx = avg_4sc([probe[2] for probe in moms_p_mms])
    t_a_mmsx = avg_4sc([probe[2] for probe in moms_a_mms])

    # Pressure tensor
    p_p_mmsx = avg_4sc([probe[3] for probe in moms_p_mms])
    p_a_mmsx = avg_4sc([probe[3] for probe in moms_a_mms])

    moms_p_mmsx = [n_p_mmsx, v_gsm_p_mmsx, t_p_mmsx, p_p_mmsx]
    moms_a_mmsx = [n_a_mmsx, v_gsm_a_mmsx, t_a_mmsx, p_a_mmsx]

    return moms_p_mmsx, moms_a_mmsx


def load_hpca_flux(tint, mms_id, config):
    r"""Load HPCA proton (H+) and alpha particles (He2+) omni-directional
    differential particle flux.

    Parameters
    ----------
    tint : list
        Time interval
    mms_id : int
        Spacecraft index
    config : dict
        Loading configuration dictionary from .yml file.

    Returns
    -------
    dpf_hpca_omni_p : list of xarray.DataArray
        Omni-directional H+ differential particle flux.
    dpf_hpca_omni_a : list of xarray.DataArray
        Omni-directional He2+ differential particle flux.

    """

    assert config and isinstance(config, dict) and config.get("hpca")

    # Create suffix with data rate and data level. If not filled default
    # values are SRVY mode and L2 data.
    data_rate_ = config["hpca"].get("data_rate", "srvy")
    data_level = config["hpca"].get("level", "l2")
    suffx_hpca = f"hpca_{data_rate_}_{data_level}"

    # Protons (H+)
    dpf_hpca_p = get_data(f"dpfhplus_{suffx_hpca}", tint, mms_id,
                          data_path=config["data_path"])

    # Alpha particles (He++)
    dpf_hpca_a = get_data(f"dpfheplusplus_{suffx_hpca}", tint, mms_id, data_path=config["data_path"])


    # Spin start
    s_az = get_data(f"saz_{suffx_hpca}", tint, mms_id, data_path=config["data_path"])

    dpf_hpca_p_despin = hpca_spin_sum(dpf_hpca_p, s_az)
    dpf_hpca_a_despin = hpca_spin_sum(dpf_hpca_a, s_az)

    dpf_hpca_omni_p = hpca_calc_anodes(dpf_hpca_p_despin)
    dpf_hpca_omni_a = hpca_calc_anodes(dpf_hpca_a_despin)

    return dpf_hpca_omni_p, dpf_hpca_omni_a


def load_hpca_flux_mmsx(tint, config):
    r"""Load HPCA proton (H+) and alpha particles (He2+) omni-directional
    differential particle flux for all spacecraft and averages at the center
    of mass of the tetrahedron.

    Parameters
    ----------
    tint : list
        Time interval
    config : dict
        Loading configuration dictionary from .yml file.

    Returns
    -------
    dpf_hpca_omni_p : list of xarray.DataArray
        Omni-directional H+ differential particle flux.
    dpf_hpca_omni_a : list of xarray.DataArray
        Omni-directional He2+ differential particle flux.

    """
    dpf_hpca_omni_p_mms = [None for _ in range(1, 5)]
    dpf_hpca_omni_a_mms = [None for _ in range(1, 5)]

    for i, ic in enumerate(range(1, 5)):
        spec_ = load_hpca_flux(tint, ic, config)
        dpf_hpca_omni_p_mms[i], dpf_hpca_omni_a_mms[i] = spec_

    # Proton flux
    dpf_hpca_p = avg_4sc(dpf_hpca_omni_p_mms)

    # Alpha particle flux
    dpf_hpca_a = avg_4sc(dpf_hpca_omni_a_mms)

    return dpf_hpca_p, dpf_hpca_a


def load_fpi_def_omni(tint, mms_id, config):
    r"""Load FPI-DIS and FPI-DES omni-directional differential energy flux.

    Parameters
    ----------
    tint : list
        Time interval
    mms_id : int
        Spacecraft index
    config : dict
        Loading configuration dictionary from .yml file.

    Returns
    -------
    def_fpi_omni_i : list of xarray.DataArray
        Omni-directional proton differential energy flux.
    def_fpi_omni_e : list of xarray.DataArray
        Omni-directional electron differential energy flux.

    """

    assert config and isinstance(config, dict) and config.get("fpi")

    # Create suffix with data rate and data level. If not filled default
    # values are FAST mode and L2 data.
    data_rate = config["fpi"].get("data_rate", "fast")
    data_levl = config["fpi"].get("level", "l2")
    suffx_fpi = f"fpi_{data_rate}_{data_levl}"

    def_fpi_omni_i = get_data(f"defi_{suffx_fpi}", tint, mms_id,
                              data_path=config["data_path"])

    def_fpi_omni_e = get_data(f"defe_{suffx_fpi}", tint, mms_id,
                              data_path=config["data_path"])

    return def_fpi_omni_i, def_fpi_omni_e


def load_fpi_def_omni_mmsx(tint, config):
    r"""Load FPI-DIS and FPI-DES omni-directional differential energy flux for
    all spacecraft and averages at the center of mass of the tetrahedron.

    Parameters
    ----------
    tint : list
        Time interval
    config : dict
        Loading configuration dictionary from .yml file.

    Returns
    -------
    def_fpi_omni_i : list of xarray.DataArray
        Omni-directional proton differential energy flux.
    def_fpi_omni_e : list of xarray.DataArray
        Omni-directional electron differential energy flux.

    """

    def_fpi_omni_i_mms = [None for _ in range(4)]
    def_fpi_omni_e_mms = [None for _ in range(4)]

    for i, ic in enumerate(range(1, 5)):
        spec_ = load_fpi_def_omni(tint, ic, config)
        def_fpi_omni_i_mms[i], def_fpi_omni_e_mms[i] = spec_

    def_fpi_omni_i_mmsx = avg_4sc(def_fpi_omni_i_mms)
    def_fpi_omni_e_mmsx = avg_4sc(def_fpi_omni_e_mms)

    return def_fpi_omni_i_mmsx, def_fpi_omni_e_mmsx


def load_fpi_dpf_omni(tint, mms_id, config):
    r"""Load FPI-DIS and FPI-DES omni-directional differential particle flux.

    Parameters
    ----------
    tint : list
        Time interval
    mms_id : int
        Spacecraft index
    config : dict
        Loading configuration dictionary from .yml file.

    Returns
    -------
    dpf_fpi_omni_i : list of xarray.DataArray
        Omni-directional proton differential particle flux.
    dpf_fpi_omni_e : list of xarray.DataArray
        Omni-directional electron differential particle flux.

    """
    assert config and isinstance(config, dict) and config.get("fpi")

    # Create suffix with data rate and data level. If not filled default
    # values are FAST mode and L2 data.
    data_rate = config["fpi"].get("data_rate", "fast")
    data_levl = config["fpi"].get("level", "l2")
    suffx_fpi = f"fpi_{data_rate}_{data_levl}"

    vdf_fpi_i = get_data(f"pdi_{suffx_fpi}", tint, mms_id,
                         data_path=config["data_path"])

    # Compute differential particle flux
    dpf_fpi_omni_i = vdf_omni(psd2dpf(vdf_fpi_i))

    vdf_fpi_e = get_data(f"pde_{suffx_fpi}", tint, mms_id,
                         data_path=config["data_path"])

    # Compute differential particle flux
    dpf_fpi_omni_e = vdf_omni(psd2dpf(vdf_fpi_e))

    return dpf_fpi_omni_i, dpf_fpi_omni_e


def load_fpi_dpf_omni_mmsx(tint, config):
    r"""Load FPI-DIS and FPI-DES omni-directional differential particle flux
    for all spacecraft and averages at the center of mass of the tetrahedron.

    Parameters
    ----------
    tint : list
        Time interval
    config : dict
        Loading configuration dictionary from .yml file.

    Returns
    -------
    dpf_fpi_omni_i : list of xarray.DataArray
        Omni-directional proton differential particle flux.
    dpf_fpi_omni_e : list of xarray.DataArray
        Omni-directional electron differential particle flux.

    """
    dpf_fpi_omni_i_mms = [None for _ in range(4)]
    dpf_fpi_omni_e_mms = [None for _ in range(4)]

    for i, ic in enumerate(range(1, 5)):
        spec_ = load_fpi_dpf_omni(tint, ic, config)
        dpf_fpi_omni_i_mms[i], dpf_fpi_omni_e_mms[i] = spec_

    dpf_fpi_omni_i_mmsx = avg_4sc(dpf_fpi_omni_i_mms)
    dpf_fpi_omni_e_mmsx = avg_4sc(dpf_fpi_omni_e_mms)

    return dpf_fpi_omni_i_mmsx, dpf_fpi_omni_e_mmsx


def load_feeps_dpf_omni(tint, mms_id, config):
    r"""Load FEEPS ion and electron omni-directional differential particle
    flux.

    Parameters
    ----------
    tint : list
        Time interval
    mms_id : int
        Spacecraft index
    config : dict
        Loading configuration dictionary from .yml file.

    Returns
    -------
    dpf_feeps_omni_i : list of xarray.DataArray
        Omni-directional proton differential particle flux.
    dpf_feeps_omni_e : list of xarray.DataArray
        Omni-directional electron differential particle flux.

    """
    assert config and isinstance(config, dict) and config.get("feeps")

    # Create suffix with data rate and data level. If not filled default
    # values are SRVY mode and L2 data.
    data_rate = config["feeps"].get("data_rate", "srvy")
    data_levl = config["feeps"].get("level", "l2")
    suffx_feeps = f"{data_rate}_{data_levl}"

    # Ions
    dpf_feeps_alle_i = get_feeps_alleyes(f"fluxi_{suffx_feeps}", tint, mms_id,
                                         data_path=config["data_path"])
    dpf_feeps_alle_i = feeps_correct_energies(dpf_feeps_alle_i)
    dpf_feeps_alle_i = feeps_flat_field_corrections(dpf_feeps_alle_i)
    dpf_feeps_alle_i = feeps_remove_bad_data(dpf_feeps_alle_i)
    dpf_feeps_alle_i_clean, _ = feeps_split_integral_ch(dpf_feeps_alle_i)
    dpf_feeps_omni_i = feeps_omni(feeps_remove_sun(dpf_feeps_alle_i_clean))

    # Electrons
    dpf_feeps_alle_e = get_feeps_alleyes(f"fluxe_{suffx_feeps}", tint, mms_id,
                                         data_path=config["data_path"])
    dpf_feeps_alle_e = feeps_correct_energies(dpf_feeps_alle_e)
    dpf_feeps_alle_e = feeps_flat_field_corrections(dpf_feeps_alle_e)
    dpf_feeps_alle_e = feeps_remove_bad_data(dpf_feeps_alle_e)
    dpf_feeps_alle_e_clean, _ = feeps_split_integral_ch(dpf_feeps_alle_e)
    dpf_feeps_omni_e = feeps_omni(feeps_remove_sun(dpf_feeps_alle_e_clean))

    return dpf_feeps_omni_i, dpf_feeps_omni_e


def load_feeps_dpf_omni_mmsx(tint, config):
    r"""Load FEEPS ion and electron omni-directional differential particle
    flux for all spacecraft and averages at the center of mass of the
    tetrahedron.

    Parameters
    ----------
    tint : list
        Time interval
    config : dict
        Loading configuration dictionary from .yml file.

    Returns
    -------
    dpf_feeps_omni_i_mmsx : list of xarray.DataArray
        Omni-directional proton differential particle flux.
    dpf_feeps_omni_e_mmsx : list of xarray.DataArray
        Omni-directional electron differential particle flux.

    """

    dpf_feeps_omni_i_mms = [None for _ in range(4)]
    dpf_feeps_omni_e_mms = [None for _ in range(4)]

    for i, ic in enumerate(range(1, 5)):
        spec_ = load_feeps_dpf_omni(tint, ic, config)
        dpf_feeps_omni_i_mms[i], dpf_feeps_omni_e_mms[i] = spec_

    dpf_feeps_omni_i_mmsx = avg_4sc(dpf_feeps_omni_i_mms)
    dpf_feeps_omni_e_mmsx = avg_4sc(dpf_feeps_omni_e_mms)

    return dpf_feeps_omni_i_mmsx, dpf_feeps_omni_e_mmsx


def load_eis_allt(specie, tint, mms_id, config):
    r"""Load EIS proton (H+) or alpha particle (He2+) or electron
    differential particle flux and counts for individual telescopes.


    Parameters
    ----------
    specie : str
        Particle specie.
    tint : list
        Time interval.
    mms_id : int
        Spacecraft index.
    config : dict
        Loading configuration dictionary from .yml file.

    Returns
    -------
    dpf_eis : xarray.Dataset
        Differential particle flux for all telescopes.
    cts_eis : xarray.Dataset
        Counts for all telescopes.

    Notes
    -----
    If specie is proton, combines PHxTOF and ExTOF.

    """

    assert config and isinstance(config, dict) and config.get("eis")

    # Create suffix with data rate and data level. If not filled default
    # values are SRVY mode and L2 data.
    data_rate = config["eis"].get("data_rate", "srvy")
    data_levl = config["eis"].get("level", "l2")
    suffx_eis = f"{specie}_{data_rate}_{data_levl}"

    if specie.lower() == "electron":
        dpf_eis = get_eis_allt(f"flux_electronenergy_{suffx_eis}", tint,
                               mms_id, data_path=config["data_path"])
        cts_eis = get_eis_allt(f"counts_electronenergy_{suffx_eis}", tint,
                               mms_id, data_path=config["data_path"])

    else:
        # Energy x Time-Of-Flight flux & counts
        dpf_extof_allt = get_eis_allt(f"flux_extof_{suffx_eis}", tint, mms_id,
                                      data_path=config["data_path"])
        cts_extof_allt = get_eis_allt(f"counts_extof_{suffx_eis}", tint,
                                      mms_id, data_path=config["data_path"])

        if specie.lower() == "proton":
            # Pulse-Height x Time-Of-Flight flux & counts
            dpf_phxtof_allt = get_eis_allt(f"flux_phxtof_{suffx_eis}", tint,
                                           mms_id,
                                           data_path=config["data_path"])
            cts_phxtof_allt = get_eis_allt(f"counts_phxtof_{suffx_eis}", tint,
                                           mms_id,
                                           data_path=config["data_path"])

            # Combine Pulse-Height x Time-Of-Flight and Energy x Time-Of-Flight
            dpf_eis = eis_combine_proton_spec(dpf_phxtof_allt, dpf_extof_allt)
            cts_eis = eis_combine_proton_spec(cts_phxtof_allt, cts_extof_allt)
        else:
            dpf_eis = dpf_extof_allt
            cts_eis = cts_extof_allt

    return dpf_eis, cts_eis


def load_eis_allt_mmsx(specie, tint, config):
    r"""Load EIS proton (H+) or alpha particle (He2+) or electron
    differential particle flux and counts for individual telescopes for all
    spacecraft and combine spacecraft using the nearest neighbour method.


    Parameters
    ----------
    specie : str
        Particle specie.
    tint : list
        Time interval.
    config : dict
        Loading configuration dictionary from .yml file.

    Returns
    -------
    dpf_eis : xarray.Dataset
        Differential particle flux for all telescopes.
    cts_eis : xarray.Dataset
        Counts for all telescopes.

    Notes
    -----
    If specie is proton, combines PHxTOF and ExTOF.

    """

    dpf_eis_allt_mms = [None for _ in range(3)]
    cts_eis_allt_mms = [None for _ in range(3)]

    if specie.lower() == "electron":
        ic = [1, 2, 4]
    else:
        ic = [2, 3, 4]

    for i, ic in enumerate(ic):
        spec_ = load_eis_allt(specie, tint, ic, config)
        dpf_eis_allt_mms[i], cts_eis_allt_mms[i] = spec_

    return dpf_eis_allt_mms, cts_eis_allt_mms


def load_eis_omni_mmsx(specie, tint, config):
    r"""Computes EIS proton (H+) or alpha particle (He2+) or electron
    omni-directional differential particle flux and counts for all spacecraft
    and combine spacecraft using the nearest neighbour method.


    Parameters
    ----------
    specie : str
        Particle specie.
    tint : list
        Time interval.
    config : dict
        Loading configuration dictionary from .yml file.

    Returns
    -------
    dpf_eis : xarray.Dataset
        Combined spacecraft omni-directional differential particle flux.
    cts_eis : xarray.Dataset
        Combined spacecraft omni-directional counts.

    Notes
    -----
    If specie is proton, combines PHxTOF and ExTOF.

    """

    eis_allt_mms = load_eis_allt_mmsx(specie, tint, config)
    dpf_eis_allt_mms, cts_eis_allt_mms = eis_allt_mms

    dpf_omni_mms = [None for _ in range(3)]
    cts_omni_mms = [None for _ in range(3)]

    for (i, dpf), cts in zip(enumerate(dpf_eis_allt_mms), cts_eis_allt_mms):
        dpf_omni_mms[i] = eis_omni(eis_spin_avg(dpf, method="mean"))
        cts_omni_mms[i] = eis_omni(eis_spin_avg(cts, method="sum"),
                                   method="sum")

    dpf_omni = eis_spec_combine_sc(dpf_omni_mms, method="mean")
    cts_omni = eis_spec_combine_sc(cts_omni_mms, method="sum")

    return dpf_omni, cts_omni
