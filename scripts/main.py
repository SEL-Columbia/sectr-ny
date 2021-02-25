from scripts.model import create_model
from scripts.utils import *
import numpy as np
import datetime
from scripts.results_processing import raw_results_retrieval, full_results_processing


if __name__ == '__main__':
    args = get_args()

    dir_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.__dict__['dir_time'] = dir_time
    # args.__dict__['dir_time'] = '20210217-110154'


    ### --- INPUT the model_config and the numbers in option 0, 1, 2, 3 --- ###
    # Define model config and set of heating, EV loads, and/or GHG reduction target appropriately
    model_config = 1

    # 0: LCT + Elec. specified, GHG returned
    if model_config == 0:
        elec_ratios  = [0.5, 0.8]
        lowc_targets = [0.5, 0.8]
        dghg_targets = [np.nan]*len(elec_ratios) # indeterminate

    # 1: LCT + GHG specified, Elec. returned
    elif model_config == 1:
        lowc_targets = [0.7, 0.7, 1]
        dghg_targets = [0.4,0.7, 0.7]
        elec_ratios = [np.nan]*len(lowc_targets) # indeterminate

    # 2: Elec. + GHG specified, LCT returned.
    elif model_config  == 2:
        dghg_targets    = [0.4, 0.6]
        elec_ratios = [0.6, 0.8]
        lowc_targets   = [np.nan]*len(dghg_targets) # indeterminate

    # 3: Minimize LCOE for GHG specified, LCT/RG and Elec. returned
    elif model_config == 3:
        dghg_targets = [0.3]
        lowc_targets = [np.nan]*len(dghg_targets) # indeterminate
        elec_ratios = [np.nan]*len(dghg_targets) # indeterminate

    else:
        raise ValueError(f'model_config {model_config} must be one of 0,1,2,3.')


    for scen_ix in range(len(lowc_targets)):
        # Initialize scenario parameters
        lct         = lowc_targets[scen_ix]
        ghgt        = dghg_targets[scen_ix]
        elec_ratio  = elec_ratios[scen_ix]

        # Create the model
        m = create_model(args, model_config, lct, ghgt, elec_ratio)

        # Set model solver parameters
        m = set_gurobi_model_params(args, m)

        # # Solve the model
        m.optimize()

        # Retrieve the model solution
        allvars = m.getVars()

        # Process the model solution
        raw_results_retrieval(args, m, model_config, scen_ix)

    full_results_processing(args)