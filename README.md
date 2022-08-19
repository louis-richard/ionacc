# Proton and Helium Ion Acceleration at Magnetotail Plasma Jets
[![GitHub license](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](./LICENSE.txt) [![LASP](https://img.shields.io/badge/datasets-MMS_SDC-orange.svg)](https://lasp.colorado.edu/mms/sdc/)

Code for the paper Proton and Helium Ion Acceleration at Magnetotail Plasma Jets

## Abstract

We investigate two flow bursts in a series of Earthward bursty bulk flows (BBFs) observed by the Magnetospheric Multiscale (MMS) spacecraft in Earth’s magnetotail at (-24, 7, 4)~$R_E$ in Geocentric Solar Magnetospheric (GSM) coordinates. At the leading edges of the BBFs, we observe complex magnetic field structures. In particular, we focus on one BBF which contains large-amplitude magnetic field fluctuations on the time scale of the proton gyroperiod, and another with a large scale dipolarization. For both events, the magnetic field structures are associated with flux increases of supra-thermal ions with energies $\gtrsim 100~\textrm{keV}$. We observe that helium ions dominate the ion flux at energies $\gtrsim 150$ keV. We investigate the ion acceleration mechanism and its dependence on the mass and charge state of H$^+$ and He$^{2+}$ ions. We show that for both events, the ions with gyroradii smaller than the dawn-dusk scale of the structure are accelerated by the ion bulk flow. For ions with larger gyroradii, the acceleration is likely due to a localized spatially limited electric field for the event with a large-scale dipolarization. For the event with fluctuating magnetic field, the acceleration of ions with gyroradii comparable with the scale of the magnetic fluctuations can be explained by resonance acceleration.

## Reproducing our results
- Instructions for reproduction are given within each section folder, in 
  the associated README.md file.

## Requirements
- A [`requirements.txt`](./requirements.txt) file is available at the root 
  of this repository, specifying the required packages for our analysis.

- Routines specific to this study [`JetFronts`](./JetFronts) is 
  pip-installable: from the [`JetFronts`](./JetFronts) folder run `pip 
  install .`


## Acknowledgement
We thank the entire MMS team and instrument PIs for data access and support.
All of the data used in this paper are publicly available from the MMS 
Science Data Center https://lasp.colorado.edu/mms/sdc/. Data analysis was 
performed using the pyrfu analysis package available at 
https://pypi.org/project/pyrfu/. This work is supported by the SNSA grant 
139/18.