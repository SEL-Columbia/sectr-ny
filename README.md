# sectr-ny

Repository for the System Electrification and Capacity Transition -- New York State (SECTR-NY) model.
Corresponding author: Terry Conlon (terence.m.conlon@gmail.com). 

## Overview

This repository contains code, data, and results related to SECTR-NY, a model fully described in the following [paper](https://arxiv.org/abs/2203.11263) and summarized below. The SECTR-NY model is solved using Gurobi, a commercial optimization software package. Academics and students can sign up for a Gurobi license [here](https://www.gurobi.com/academia/academic-program-and-licenses/).

SECTR-NY is defined by individual nodes, i, representing geographical sub-areas within New York State (NYS). Along with existing electricity demand, each node contains electrified heating  and vehicle charging loads at each timestep, t, within the overall time period simulated, T. To determine the least-cost infrastructure mix in future model scenarios, decision variables are assigned node-specific costs. SECTR-NY uses a characterization of the state’s energy-related GHG emissions as both a reference quantity for GHG emissions reduction computations and to compute the emissions impact of reduced fossil fuel usage associated with heating and vehicle electrification; the model does not consider improved efficiency or growth of fossil fuel end uses.

SECTR-NY evaluates different low-carbon electricity supply and end use electrification scenarios by computing the total cost of new and existing infrastructure capacity and maintenance, fuels, and resource operation to estimate the total annual cost of electricity generation and transmission; these returned costs do not include delivery expenses (primarily distribution system costs). The modeling framework does not include the cost of replacing current fossil fuel-based building systems and vehicles or electricity distribution system costs; as such, SECTR cost computations can be considered those that typically constitute the “supply” portion of a utility customer’s bill. 

## Repository Structure

```
sectr-ny
├── __init__.py
├── README.md
├── setup.py
├── data_uploads/
├── model_results/
├── scripts/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   ├── params.yaml
│   ├── results_processing.py
│   ├── utils.py
```
## Root Level Repository Description

The `model_results/` folder is too large to include in this repository. It is instead made available [here](). This folder contains all results presented in the SECTR-NY [publication](https://arxiv.org/abs/2203.11263).

The `data_uploads/` folder holds the timeseries and assumptions spreadsheets required to run SECTR-NY. Each spreadhseet is named in a way that distinguishes its content. It is recommended that these files and filenames are left as is. 

The `scripts/` folder contains the Python files that instantiate and solve the SECTR-NY model. Descriptions of the files within the folder are included below. The code within each file is commented for interpretability. 


* `scripts/main.py`: The main script for SECTR-NY. This is the file to run to setup and solve a SECTR-NY simulation. In the script, a user specifies the model configuration and some combination of  1) emissions reductions target, 2) the low-carbon electricity percent, or 3) the heating and vehicle electrification percent for the model scenario. 

* `scripts/model.py`: This script defines the SECTR-NY model, a linear program representing the NYS electricity system. 

* `scripts/params.py`: This file defines parameters for the SECTR-NY including but not limited to: location-specific technology costs, existing infrastructure limits, and sectoral emissions quantities. 

* `scripts/results_processing.py`: This script turns SECTR-NY model outputs into `.xlsx` files in `model_results/` for interpretation and evaluation.

* `scripts/utils.py`: This script contains various helper functions for the SECTR-NY model. 

