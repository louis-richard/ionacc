# Turbulent Jet Fronts and Related Ion Acceleration
[![GitHub license](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE) [![LASP](https://img.shields.io/badge/datasets-MMS_SDC-orange.svg)](https://lasp.colorado.edu/mms/sdc/)

Code for the paper Ion Acceleration at Magnetotail Plasma Jets

## Abstract

We investigate a series of Earthward bursty bulk flows (BBFs) observed by the 
Magnetospheric Multiscale (MMS) spacecraft in Earth’s magnetotail at 
(-24, 7, 4)~$R_E$ in Geocentric Solar Magnetospheric (GSM) coordinates. At the 
leading edges of the BBFs, we observe complex magnetic field structures. In 
particular, we focus on one which presents a chain of small scale 
($\sim 0.5~R_E$) dipolarizations, and another with a large scale 
($\sim 3.5~R_E$) dipolarization. Although the two structures have different 
scales, both of these structures are associated with flux increases of 
supra-thermal ions with energies $\gtrsim 100~\textrm{keV}$. We investigate 
the ion acceleration mechanism and its dependence on the mass and charge state. 
We show that the ions with gyroradii smaller than the scale of the structure 
are accelerated by the ion bulk flow. We show that whereas in the small scale 
structure, ions with gyroradii comparable with the scale of the structure 
undergo resonance acceleration, and the acceleration in the larger scale 
structure is more likely due to a spatially limited electric field.

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