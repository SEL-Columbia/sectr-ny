from scripts.model import create_model
from scripts.utils import *
import numpy as np
import datetime
from scripts.results_processing import raw_results_retrieval, full_results_processing
import gurobipy as gp
import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

if __name__ == '__main__':
    args = get_args()

    dir_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.__dict__['dir_time'] = dir_time


    ### --- INPUT the model_config and the numbers in option 0, 1, 2, 3 --- ###
    # Define model config and set of heating, EV loads, and/or GHG reduction target appropriately
    model_config = 4

    # 0: LCT + Elec. specified, GHG returned
    if model_config == 0:
        lowc_targets = [0.381646687]
        elec_ratios = [0]
        proj_years = [2019]
        dghg_targets = [np.nan]*len(elec_ratios) # indeterminate

    # 1: LCT + GHG specified, Elec. returned
    elif model_config == 1:
        lowc_targets = [0.95] 
        dghg_targets = [0.55]
        proj_years = [2035]
        elec_ratios = [np.nan]*len(lowc_targets) # indeterminate
        min_offshore_capacity = [9000]
        min_solar_capacity = [0]
        min_battery_capacity_mwh = [0]

    # 2: Elec. + GHG specified, LCT returned.
    elif model_config  == 2:
        dghg_targets = [0.1]
        elec_ratios = [0]
        proj_years = [2050]
        lowc_targets   = [np.nan]*len(dghg_targets) # indeterminate
        min_capacity = {2050: {'min_offshore_capacity': 0, 'min_battery_capacity_mwh': 0}}

    # 3: Minimize LCOE for GHG specified, LCT/RG and Elec. returned
    elif model_config == 3:
        dghg_targets = [0.1]
        proj_years = [2050]
        lowc_targets = [np.nan]*len(dghg_targets) # indeterminate
        elec_ratios = [np.nan]*len(dghg_targets) # indeterminate
        min_capacity = {2050: {'min_offshore_capacity': 0, 'min_battery_capacity_mwh': 0}}

    elif model_config == 4:
        dghg_targets = [0.85]
        lowc_targets = [1]
        proj_years = [2050]
        elec_ratios = [np.nan]*len(lowc_targets) # indeterminate
        min_capacity = {2050: {'min_offshore_capacity': 0, 'min_battery_capacity_mwh': 0}}

    else:
        raise ValueError(f'model_config {model_config} must be one of 0,1,2,3,4.')

    for scen_ix in range(len(lowc_targets)):
        # Initialize scenario parameters
        lct                     = lowc_targets[scen_ix]
        ghgt                    = dghg_targets[scen_ix]
        elec_ratio              = elec_ratios[scen_ix]
        proj_year               = proj_years[scen_ix]

        # Create the model
        m = create_model(args, model_config, lct, ghgt, elec_ratio, proj_year, min_capacity)

        # Set model solver parameters
        m = set_gurobi_model_params(args, m)

        # # Solve the model
        try:
            m.optimize()
        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))

        if m.status == gp.GRB.INFEASIBLE:
            print('Model is infeasible')
            m.computeIIS()
            m.write("model.ilp")
        else:
            # Continue with the rest of your code
            allvars = m.getVars()
            # Process the model solution
            raw_results_retrieval(args, m, model_config, scen_ix, proj_year)

    full_results_processing(args)
