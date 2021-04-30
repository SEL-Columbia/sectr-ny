import os
import math
import argparse
import yaml
import numpy as np
import pandas as pd
import glob

def get_args():
    '''
    Returns a namespace dictionary `args` with all the parameters specified in params.yaml
    :return: args
    '''
    # Store all parameters for easy retrieval
    parser = argparse.ArgumentParser(
        description = 'nys-cem')
    parser.add_argument('--params_filename',
                        type=str,
                        default='params.yaml',
                        help = 'Loads model parameters')
    args = parser.parse_args()
    config = yaml.load(open(args.params_filename), Loader=yaml.FullLoader)
    for k,v in config.items():
        args.__dict__[k] = v

    return args

def annualization_rate(i, years):
    '''
    Return the annualization rate for capacity costs
    :param i: interest rate
    :param years: annualization period
    :return:
    '''
    return (i*(1+i)**years)/((1+i)**years-1)

def btmpv_capacity_projection(year):
    '''
    Function to determine the amount of BTM solar present in a scenario year
    :param year: year
    :return: BTM capacity
    '''
    K = 10982.023
    Q = 0.0001680925
    B = 0.1202713
    M = 1995.067
    v = 0.000004955324
    tt_btmpv_cap = K / (1 + Q * math.exp(-B * (year - M))) ** (1 / v)

    return tt_btmpv_cap

def set_gurobi_model_params(args, model):
    '''
    Set gurobi solve parameters
    :param args: args dictionary
    :param model: gurobi model
    :return: gurobi model with solve parameters applied
    '''

    model.setParam("FeasibilityTol", args.feasibility_tol)
    model.setParam("OptimalityTol", args.optimality_tol)
    model.setParam("Method", args.method)
    model.setParam("BarConvTol", args.bar_conv_tol)
    model.setParam("BarOrder", args.bar_order)
    model.setParam("Crossover", args.crossover)
    model.setParam("NonConvex", args.nonconvex)

    return model


def load_timeseries(args):
    '''
    Function for loading the input timeseries required by the model.
    :param args: args dictionary
    :return: Demand and potential renewable generation timeseries for the model
    '''
    T = args.num_hours

    # Load all potential generation and actual hydro generation time-series
    onshore_pot_hourly    = np.array(pd.read_csv(f'{args.data_dir}/onshore_power_hourly_norm.csv', index_col=0))[0:T]
    offshore_pot_hourly   = np.array(pd.read_csv(f'{args.data_dir}/offshore_power_hourly_norm.csv', index_col=0))[0:T]
    solar_pot_hourly      = np.array(pd.read_csv(f'{args.data_dir}/gridpv_power_hourly_norm.csv', index_col=0))[0:T]
    btmpv_pot_hourly      = np.array(pd.read_csv(f'{args.data_dir}/btmpv_power_hourly_norm.csv', index_col=0))[0:T]
    flex_hydro_daily_mwh  = np.array(pd.read_csv(f'{args.data_dir}/flex_hydro_daily_mwh.csv', index_col=0))[0:int(T/24)]
    fixed_hydro_hourly_mw = np.array(pd.read_csv(f'{args.data_dir}/fixed_hydro_hourly_mw.csv', index_col=0))[0:T]


    # Load baseline and full heating electric and thermal demand timeseries
    baseline_demand_hourly_mw = np.array(pd.read_csv(f'{args.data_dir}/baseline_demand_hourly_mw.csv',
                                                      index_col=0))[0:T]
    if args.dss_synthetic_ts:
        full_elec_heating_load_hourly_mw = np.array(pd.read_csv(f'{args.data_dir}/elec_heating_dss50_hourly_mw.csv',
                                                           index_col=0))[0:T]
    else:
        full_elec_heating_load_hourly_mw = np.array(pd.read_csv(f'{args.data_dir}/elec_heating_hourly_mw.csv',
                                                           index_col=0))[0:T]

    full_ff_heating_load_hourly_mw = np.array(pd.read_csv(f'{args.data_dir}/ff_heating_hourly_mw.csv',
                                                       index_col=0))[0:T]
    full_ff_dss50_hourly_mw = np.array(pd.read_csv(f'{args.data_dir}/ff_heating_dss50_hourly_mw.csv',
                                                   index_col=0))[0:T]

    ## Set average hydropower generation
    hydro_avg_gen_mw = np.mean(fixed_hydro_hourly_mw, axis=0) + np.mean(flex_hydro_daily_mwh, axis=0)/24
    args.__dict__['hydro_avg_gen_mw'] = hydro_avg_gen_mw

    # Load full EV demand timeseries from the corresponding .csv, and calculate the region-wide average EV load.
    # If not ev_set_profile, take the distributed nodal average load repeated at every timestep in T
    full_ev_load_hourly_mw =  np.array(pd.read_csv(f'{args.data_dir}/elec_vehicle_hourly_mw.csv',
                                                       index_col=0))[0:args.num_hours,:]
    full_ev_avg_load_hourly_mw      = np.mean(full_ev_load_hourly_mw, axis=0)

    if not args.ev_set_profile_boolean:
        full_ev_load_hourly_mw  = np.tile(full_ev_avg_load_hourly_mw,(args.num_hours,1))

    # Clipping option to remove renewable generation potential values less than `min_val` MW/MW
    clip_ts_for_numerical_issues = True
    if clip_ts_for_numerical_issues:
        min_val = 1e-3
        onshore_pot_hourly  = np.where(onshore_pot_hourly < min_val, 0, onshore_pot_hourly)
        offshore_pot_hourly = np.where(offshore_pot_hourly < min_val, 0, offshore_pot_hourly)
        solar_pot_hourly    = np.where(solar_pot_hourly < min_val, 0, solar_pot_hourly)
        btmpv_pot_hourly    = np.where(btmpv_pot_hourly < min_val, 0, btmpv_pot_hourly)

    return baseline_demand_hourly_mw, full_elec_heating_load_hourly_mw, full_ff_heating_load_hourly_mw, \
           full_ff_dss50_hourly_mw, full_ev_load_hourly_mw, full_ev_avg_load_hourly_mw, onshore_pot_hourly, \
           offshore_pot_hourly, solar_pot_hourly, btmpv_pot_hourly, fixed_hydro_hourly_mw, \
           flex_hydro_daily_mwh

def return_costs_for_model(args):
    '''
    Function for returning costs used in the model. The returned cost dictionary is used in both the model
    paramterization (i.e. to apply costs to decision variables) and in the results_processing.py script in order to
    determine the total cost quantities.

    :param args: args dictionary
    :return: cost_dict, a dictionary with per-unit costs for new capacity and generation, which includes both
    annualized capex and opex where appropriate
    '''

    # Here, collect two separate annualization rates -- one for 20 years and one for 10 years
    # Currently, we apply the 20 year annualization rate to new generating capacity (wind, solar, gt) and new
    # transmission. We apply the 10 year annualization rate to new storage capacity (battery + hydrogen).
    # The interest rate is set in params.yaml
    ann_rate_20years = annualization_rate(args.i_rate, 20)
    ann_rate_10years = annualization_rate(args.i_rate, 10)

    # Determine whether we are using the low or medium cost assumptions
    if args.re_cost_scenario == 'low':
        onshore_capex_mw     = args.onshore_capex_low_mw
        solar_capex_mw       = args.solar_capex_low_mw
        offshore_capex_mw    = args.offshore_capex_low_mw
        battery_capex_mw     = args.battery_capex_low_mw
        battery_capex_mwh    = args.battery_capex_low_mwh
    elif args.re_cost_scenario == 'medium':
        onshore_capex_mw     = args.onshore_capex_medium_mw
        solar_capex_mw       = args.solar_capex_medium_mw
        offshore_capex_mw    = args.offshore_capex_medium_mw
        battery_capex_mw     = args.battery_capex_medium_mw
        battery_capex_mwh    = args.battery_capex_medium_mwh

    else:
        raise ValueError(f'args.cost_scenario {args.cost_scenario} must be either low or medium')

    # Set up cost dict to hold costs for model
    cost_dict = {}

    # Determining the annualized capex costs
    ann_onshore_capex = args.num_years * ann_rate_20years * float(onshore_capex_mw)
    ann_offshore_capex = args.num_years * ann_rate_20years * float(offshore_capex_mw)
    ann_solar_capex = np.array([args.num_years * ann_rate_20years * float(x) for x in solar_capex_mw])
    ann_battery_capex_mwh = args.num_years * ann_rate_10years * float(battery_capex_mwh)
    ann_battery_capex_mw = args.num_years * ann_rate_10years * float(battery_capex_mw)
    ann_h2_capex_mwh = np.array([args.num_years * ann_rate_10years * float(x) for x in args.h2_capex_mwh])
    ann_h2_capex_mw = np.array([args.num_years * ann_rate_10years * float(x) for x in args.h2_capex_mw])
    ann_gt_capex_mw = np.array([args.num_years * ann_rate_20years * args.reserve_req * float(x)
                               for x in args.gt_capex_mw])

    # Determining the FOM costs
    onshore_fom_cost = args.num_years * float(args.onshore_om_cost_mw_yr)
    offshore_fom_cost = args.num_years * float(args.offshore_om_cost_mw_yr)
    solar_fom_cost = args.num_years * float(args.solar_om_cost_mw_yr)
    gt_fom_cost = args.num_years * float(args.new_gt_om_cost_mw_yr) * args.reserve_req

    # Determining the VOM costs
    gt_vom_cost = float(args.new_gt_om_cost_mwh)
    gt_fuel_cost = np.array(args.gt_fuel_cost_mwh)

    # Per-MW costs associated with new capacity are the combination of annualized capex costs and fixed O&M costs
    # (where applicable).
    cost_dict['onshore_cost_per_mw']  = ann_onshore_capex + onshore_fom_cost
    cost_dict['offshore_cost_per_mw'] = ann_offshore_capex + offshore_fom_cost
    cost_dict['solar_cost_per_mw'] = ann_solar_capex + solar_fom_cost # varies by node
    cost_dict['gt_cost_per_mw'] = ann_gt_capex_mw + gt_fom_cost # varies by node
    cost_dict['battery_cost_per_mw'] = ann_battery_capex_mw
    cost_dict['battery_cost_per_mwh'] = ann_battery_capex_mwh
    cost_dict['h2_cost_per_mw'] = ann_h2_capex_mw # varies by node
    cost_dict['h2_cost_per_mwh'] = ann_h2_capex_mwh # varies by node

    # Per-MWh costs associated with generation are a combination of fuel costs and variable O&M costs (where applicable)
    cost_dict['new_gt_cost_mwh'] = gt_vom_cost + gt_fuel_cost / args.new_gt_efficiency # varies by node
    cost_dict['existing_gt_cost_mwh'] = gt_fuel_cost / args.existing_gt_efficiency # varies by node

    return cost_dict

def calculate_constant_costs(args):
    '''
    This function calculates the constant costs that are added as-is to every model run. They include annual costs for
    existing transmission and generation capacity multiplied by the number of model-simulated years, args.num_years;
    and costs for hydropower and nuclear, and biofuel generation over the number of model timesteps, T.

    :param args: args dictionary
    :return: const_costs_total, a number representing the total constant costs added to the model
    '''
    # Load timeseries to populate avg hydropower generation in case it hasn't been done yet
    _ = load_timeseries(args)

    # Extract # of hours
    T = args.num_hours

    # Find the cost of maintaining existing transmission over the # years in the study period
    existing_tx_costs = np.sum([args.existing_trans_cost_mwh[i] * float(args.existing_trans_load_mwh_yr[i]) for i in
                                range(len(args.existing_trans_load_mwh_yr))]) * args.num_years

    # Find the total amount of generation capacity eligible for capacity maintenance payments
    existing_cap_for_payments_mw = (int(args.nuclear_boolean) * np.array(args.nuc_cap_mw) + np.array(args.hydro_cap_mw)
                                    + np.array(args.biofuel_cap_mw) + np.array(args.existing_gt_cap_mw))

    # Find the cost of maintaining existing capacity over the # years in the study period by multiplying the amount
    # of eligible capacity by the nodal cost of that capacity
    existing_cap_cost = np.sum(existing_cap_for_payments_mw * np.array(args.cap_market_cost_mw_yr)) * args.num_years

    # Find the total costs of hydropower and nuclear generation by multiplying the average
    # generation quantities by their per-MWh costs and the number of time periods simulated (T hours for hydro and
    # nuclear)

    total_hydro_cost = np.sum([args.hydro_avg_gen_mw[k] * args.hydro_cost_mwh[k] for k in range(args.num_nodes)]) * T
    total_nuclear_cost = args.nuclear_boolean * np.sum([args.nuc_avg_gen_mw[k] * args.nuc_cost_mwh[k] for
                                                        k in range(args.num_nodes)]) * T

    # Find the total amount of constant costs by adding the individual cost terms determined above
    const_costs_total = (existing_tx_costs + existing_cap_cost + total_hydro_cost + total_nuclear_cost)

    return const_costs_total


def return_tx_dict(args):
    '''
    Function for returning a dictionary of transmission interfaces and associated existing limits and costs
    Each key in the dictionary corresponds to an tx interface plus a directionality. Each key's value is a tuple
    containing 1) the existing tx limit, and 2) the per-MW cost associated with new transmission capacity -- a
    combination of capex + o&m costs)

    :param args: args dictionary
    :return: tx_dict, dictionary containing relevant tx limit + cost information
    '''
    # Load transmission cost and current capacity parameters
    tx_matrix_limits = pd.read_excel(f'{args.data_dir}/transmission_matrix_limits.xlsx', header=0, index_col=0)
    tx_matrix_install_costs = pd.read_excel(f'{args.data_dir}/transmission_matrix_install_costs.xlsx', header=0,
                                            index_col=0)
    tx_matrix_om_costs = pd.read_excel(f'{args.data_dir}/transmission_matrix_om_costs.xlsx', header=0,
                                       index_col=0)
    tx_matrix_distances = pd.read_excel(f'{args.data_dir}/transmission_matrix_distances_mi.xlsx', header=0,
                                       index_col=0)

    # Define annualization rate -- here, we assume a 20 year annualization period for transmission
    annualization_cap = annualization_rate(args.i_rate, 20)

    ## Define and populate transmission dictionaries
    # Dictionary for transmission parameters
    tx_dict = {}


    # Create a transmission cost containing existing limits and costs information
    for i in range(len(tx_matrix_limits)):
        for j in range(len(tx_matrix_limits.columns)):
            if tx_matrix_limits.iloc[i, j] > 0:
                tx_dict[f'existing_tx_limit_{i+1}_{j+1}'] = (tx_matrix_limits.iloc[i, j],
                                                             args.num_years *
                                                                 (annualization_cap *
                                                                  tx_matrix_install_costs.iloc[i, j] +
                                                                  tx_matrix_om_costs.iloc[i, j]),
                                                             tx_matrix_distances.iloc[i,j])

    return tx_dict

if __name__ == '__main__':
    args = get_args()
    tx_dict = return_tx_dict(args)

    print(tx_dict)
