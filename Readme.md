# Investigating Red SN


For typical SN we expect a nonlinear changing beta as you transition from low dust to high dust. (assuming line-of-sight dust produces more scatter than intrinsic color variation)


## Installation

```shell
git clone git@github.com:benjaminrose/Investigating-Red-SN.git
cd Investigating-Red-SN
conda env create -f red_sn.yaml --yes
conda activate red_sn
```


A worked example is in `.github/workflows/install_instructions.yml` and `make setup`.


## Initial test

Look at red SN (c > 0.3). What are HR (from basic Tripp) vs color? - Is it linear (one type of dust) or not?


## Overview

Past papers have posited that the extinction law depends on the amount of reddening.  With the largest sample of red SNe, we confirm/refute this to be the case.

Here is what we see in the data (scatter as a function of c, alternative betas, ...). 
SALT2 is doing a fine job fitting the light curves. So would want to show some example fits (we have scripts for this) and say chi2 of fits are similar to blue SNe

Also here is what several SN reddening theories (SALT2, BS20, ...) predict red SN should behave?

### Plots

Fitprob for c < 0.3 vs c > 0.3: are red different than blue?

CLR for c < 0.3 vs c > 0.3

Scatter as a function of c (binned or rolling average?)


### Past Papers

Mandel paper