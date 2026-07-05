#!/usr/bin/env bash
set -e
echo Creating conda environment fanet...
conda env create -f environment.yml
echo Done. Activate with: conda activate fanet
