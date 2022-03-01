#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generic functions"""

# 3rd party imports
import numpy as np
import xarray as xr

from scipy import special
from pyrfu.pyrf import resample, datetime642iso8601, iso86012datetime64


__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2020-2021"
__license__ = "MIT"
__version__ = "2.3.7"
__status__ = "Prototype"


def combine_flux_instruments(dpf_omni_inst0, dpf_omni_inst1):
    r"""Combine oni-directional differential particle flux spectra from two
    instruments.

    Parameters
    ----------
    dpf_omni_inst0 : xarray.DataArray
        Time series of the omni-directional differential particle flux for
        the 1st instrument.
    dpf_omni_inst1 : xarray.DataArray
        Time series of the omni-directional differential particle flux for
        the 2nd instrument.

    Returns
    -------
    dpf_omni : xarray.DataArray
        Combined instruments omni-directional differential particle flux

    """
    flux_inst0 = resample(dpf_omni_inst0, dpf_omni_inst1)
    flux_inst0 = flux_inst0.assign_coords(energy=flux_inst0.energy.data * 1e-3)

    flux_inst1 = dpf_omni_inst1

    energies = np.hstack([flux_inst0.energy.data, flux_inst1.energy.data])
    tmp_data = np.hstack([flux_inst0.data, flux_inst1.data])

    ord_energies = np.argsort(energies)

    dpf_omni = xr.DataArray(tmp_data[:, ord_energies],
                            coords=[flux_inst1.time.data,
                                    energies[ord_energies]],
                            dims=["time", "energy"])
    dpf_omni.data[dpf_omni.data == 0] = np.nan

    return dpf_omni


def find_feeps_clusters(inp):
    r"""Finds groups of timestamps corresponding to flux at 200 keV larger
    than 5 \sigma.

    Parameters
    ----------
    inp : xarray.DataArray
        Omni-directional differential particle flux.

    Returns
    -------
    time : numpy.ndarray
        Times where the condition is satisfied.
    tints : list of lists
        Time intervals of the clusters of points.

    """

    p = 1. * np.arange(len(inp.time)) / (len(inp.time) - 1)
    idx_95 = np.min(np.where(p > special.erf(2 / np.sqrt(2)))[0])  # 2 sigma

    thresh_flux = []
    for i in range(len(inp.energy)):
        thresh_flux.append(np.sort(inp.data[:, i])[idx_95])
    
    indices = np.where(inp.data[:, 7] > thresh_flux[7])[0]
    times = np.vstack([inp.time.data[indices[:-1]],
                       inp.time.data[indices[:-1] + 1]]).T
    times = [list(t_) for t_ in list(datetime642iso8601(times))]
    times_d64 = iso86012datetime64(np.array(times))

    idx = np.where(np.diff(times_d64[:, 0]).astype(int) > 140000000000.)[0] + 1

    times_clusters = [times_d64[:idx[0], :]]
    for i in range(len(idx) - 1):
        times_clusters.append(times_d64[idx[i]:idx[i + 1], :])

    times_clusters.append(iso86012datetime64(np.array(times))[idx[-1]:, :])
    times_clusters = [t_clust for t_clust in times_clusters if
                      len(t_clust) > 10]

    tints = np.array([[t_[0, 0], t_[-1, 1]] for t_ in times_clusters[:-1]])
    tints = tints + np.array([np.timedelta64(-2, "m"), np.timedelta64(0, "m")])

    # Check that intervals are within burst mode data intervals
    tint_brst = [["2017-07-23T16:54:14.000", "2017-07-23T17:13:10.000"],
                 ["2017-07-23T17:17:04.000", "2017-07-23T17:22:50.000"]]
    tint_brst = iso86012datetime64(np.array(tint_brst))

    brst_check = np.where(tints[:-1, 0] < tint_brst[:, 0])[0]
    tints[brst_check, 0] = tint_brst[brst_check, 0]
    brst_check = np.where(tints[:-1, 1] > tint_brst[:, 1])[0]
    tints[brst_check, 1] = tint_brst[brst_check, 1]
    tints = [list(t_) for t_ in list(datetime642iso8601(tints))]

    return times, tints
