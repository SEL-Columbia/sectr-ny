import os
import math
import argparse
import yaml
import numpy as np
import pandas as pd
from glob import glob
from utils import get_args, load_timeseries, return_tx_dict, btmpv_capacity_projection, return_costs_for_model


def load_ts_based_results(args, processed_df):
    '''
    Function for producing results based on the timeseries outputs of the model

    :param args: args dictionary
    :param processed_df: Dataframe for storing the processed results of the model solution
    :return: processed_df, with results from the timeseries data added
    '''
    T = args.num_hours

    # Collect transmission array data
    tx_dict = return_tx_dict(args)

    # Load all the timeseries files present in the corresponding folder
    ts_results_dir = f'{args.results_dir}/{args.dir_time}/ts_results'
    ts_results_files = sorted(glob(f'{ts_results_dir}/*.csv'))

    ## Create arrays to store the timeseries results before assigning to the processed Dataframe
    total_gt_new_util = np.zeros((len(ts_results_files), args.num_nodes))
    total_gt_existing_util = np.zeros((len(ts_results_files), args.num_nodes))
    total_gt_new_ramping = np.zeros(len(ts_results_files))
    total_gt_existing_ramping = np.zeros(len(ts_results_files))

    gt_new_max_util = np.zeros(len(ts_results_files))
    gt_existing_max_util = np.zeros(len(ts_results_files))
    gt_total_max_util = np.zeros(len(ts_results_files))

    curtailment = np.zeros((len(ts_results_files), args.num_nodes))
    biofuel_util = np.zeros((len(ts_results_files), args.num_nodes))
    battery_discharge = np.zeros((len(ts_results_files), args.num_nodes))
    h2_discharge = np.zeros((len(ts_results_files), args.num_nodes))
    elec_import = np.zeros((len(ts_results_files), args.num_nodes))
    tx_util = np.zeros((len(ts_results_files), len(tx_dict)))

    # Iterate through the ts results files, each corresponding to a single scenario
    for ix, file in enumerate(ts_results_files):
        # Load the csv
        ts_csv = pd.read_csv(file)

        # Collect GT util data
        gt_new_util = np.array([ts_csv[f'gt_new_util_node_{i+1}'] for i in range(args.num_nodes)]).T
        gt_existing_util = np.array([ts_csv[f'gt_existing_util_node_{i+1}'] for i in range(args.num_nodes)]).T

        # Find GT utilization maximums
        gt_new_max_util[ix] = np.max(np.sum(gt_new_util, axis=1))
        gt_existing_max_util[ix] = np.max(np.sum(gt_existing_util, axis=1))
        gt_total_max_util[ix] = np.max(np.sum(gt_new_util + gt_existing_util, axis=1))

        # Find total GT utilization and ramping
        total_gt_new_util[ix] = np.sum(gt_new_util, axis=0)
        total_gt_existing_util[ix] = np.sum(gt_existing_util, axis=0)
        total_gt_new_ramping[ix] = np.sum(np.array([ts_csv[f'gt_new_abs_node_{i+1}']
                                                    for i in range(args.num_nodes)]).T)
        total_gt_existing_ramping[ix] = np.sum(np.array([ts_csv[f'gt_existing_abs_node_{i+1}']
                                                    for i in range(args.num_nodes)]).T)

        # Find curtailment, battery_discharge, elec_import
        curtailment[ix] = np.sum(np.array([ts_csv[f'energy_balance_slack_node_{i+1}']
                                           for i in range(args.num_nodes)]).T, axis=0)
        battery_discharge[ix] = np.sum(np.array([ts_csv[f'batt_discharge_node_{i+1}'] for i in
                                                 range(args.num_nodes)]).T, axis=0)
        h2_discharge[ix] = np.sum(np.array([ts_csv[f'h2_discharge_node_{i+1}'] for i in range(args.num_nodes)]).T,
                                  axis=0)
        elec_import[ix] = np.sum(np.array([ts_csv[f'elec_import_node_{i+1}'] for i in range(args.num_nodes)]).T,
                                 axis=0)
        biofuel_util[ix] = np.sum(np.array([ts_csv[f'biofuel_util_node_{i+1}'] for i in range(args.num_nodes)]).T,
                                 axis=0)

        # Find tx flow
        tx_output_cols = []
        for jx, tx_key in enumerate(tx_dict.keys()):
            node_out = int(tx_key.split('_')[-2])
            node_in = int(tx_key.split('_')[-1])

            tx_ts_string = f'tx_ts_{node_out}_to_{node_in}'
            tx_output_cols.append(f'tx_avg_util_{node_out}_to_{node_in}')

            tx_util[ix, jx] = np.mean(ts_csv[tx_ts_string])

    ## Begin populating the processed dataframe with stats from the timeseries
    # Add curtailment calculations, regional and by node
    processed_df['curtailment_regional_avg_mw'] = np.sum(curtailment, axis=1)/T
    for ix in range(args.num_nodes):
        processed_df[f'curtailment_node_{ix + 1}_avg_mw'] = curtailment[:, ix] / T

    # Add average new GT utilization results, regional and by node
    processed_df[f'gt_new_util_regional_avg_mw'] = np.sum(total_gt_new_util, axis=1) / T
    for ix in range(args.num_nodes):
        processed_df[f'gt_new_util_node_{ix+1}_avg_mw'] = total_gt_new_util[:, ix]/T

    # Add average existing GT utilization results, regional and by node
    processed_df[f'gt_existing_util_regional_avg_mw'] = np.sum(total_gt_existing_util, axis=1) / T
    for ix in range(args.num_nodes):
        processed_df[f'gt_existing_util_node_{ix+1}_avg_mw'] = total_gt_existing_util[:, ix]/T

    # Add averaging ramping results for the full region
    processed_df['new_gt_ramp_avg_mw'] = total_gt_new_ramping/T
    processed_df['existing_gt_ramp_avg_mw'] = total_gt_existing_ramping/T

    # Add biofuel utilization, regional by node
    processed_df[f'biofuel_util_regional_avg_mw'] = np.sum(biofuel_util, axis=1) / T
    for ix in range(args.num_nodes):
        processed_df[f'biofuel_util_node_{ix+1}_avg_mw'] = biofuel_util[:, ix]/T

    # Add battery discharge, regional and by node
    processed_df['battery_discharge_regional_avg_mw'] = np.sum(battery_discharge, axis=1)/T
    for ix in range(args.num_nodes):
        processed_df[f'battery_discharge_node_{ix+1}_avg_mw'] = battery_discharge[:, ix]/T

    # Add average H2 discharge, regional and by node
    processed_df['h2_discharge_regional_avg_mw'] = np.sum(h2_discharge, axis=1) / T
    for ix in range(args.num_nodes):
        processed_df[f'h2_discharge_node_{ix + 1}_avg_mw'] = h2_discharge[:, ix] / T

    # Add average electricity import, region and by node
    processed_df[f'elec_import_regional_avg_mw'] = np.sum(elec_import, axis=1) / T
    for ix in range(args.num_nodes):
        processed_df[f'elec_import_node_{ix+1}_avg_mw'] = elec_import[:, ix]/T

    # Add average utilization of transmission lines
    for jx, tx_key in enumerate(tx_dict.keys()):
        tx_output_col = f'tx_avg_util_{tx_key.split("_")[-2]}_to_{tx_key.split("_")[-1]}_mw'
        processed_df[tx_output_col] = tx_util[:, jx]

    return processed_df

def cost_calculations(args, cap_results_df, processed_df):
    '''
    Function for determining the total costs and LCOEs associated with model solutions


    :param args: args dictionary
    :param cap_results_df: Raw capacity results from the model solution. These results contain the nodal capacities
    of certain technologies with nodal-specific pricing in order to determine total cost
    :param processed_df: Dataframe with processed results, both capacity and timeseries-based

    :return: processed_df, full parameterized with cost calculations
    '''
    T = args.num_hours

    # Collect cost and transmission dictionaries
    cost_dict = return_costs_for_model(args)
    tx_dict = return_tx_dict(args)

    ## Find the total new capacity costs. Costs are determined for each new energy technology. Here, the costs are
    # combinations of capex and  o&m, where applicable.
    new_onshore_costs = np.array(processed_df['onshore_cap_mw']) * cost_dict['onshore_cost_per_mw']
    new_offshore_cost = np.array(processed_df['offshore_cap_mw']) * cost_dict['offshore_cost_per_mw']
    new_solar_cost = np.sum(np.array([cap_results_df[f'solar_cap_node_{ix+1}'] for ix in range(args.num_nodes)]).T
                            * np.array(cost_dict['solar_cost_per_mw']), axis=1)
    new_gt_cost = np.sum(np.array([cap_results_df[f'gt_new_cap_node_{ix+1}'] for ix in range(args.num_nodes)]).T *
                   np.array(cost_dict['gt_cost_per_mw']), axis=1)
    new_batt_cost = (np.array(processed_df['battery_power_cap_mw']) * cost_dict['battery_cost_per_mw'] +
                     np.array(processed_df['battery_energy_cap_mwh']) * cost_dict['battery_cost_per_mwh'])
    new_h2_cost = np.sum((np.array([cap_results_df[f'h2_power_cap_node_{ix+1}'] for ix in range(args.num_nodes)]).T
                   * np.array(cost_dict['h2_cost_per_mw']) +
                   np.array([cap_results_df[f'h2_energy_cap_node_{ix+1}'] for ix in range(args.num_nodes)]).T
                   * np.array(cost_dict['h2_cost_per_mwh'])), axis=1)

    # Determine total cost of new transmission by adding the cost of new transmission capacity at each interface +
    # directionality
    new_tx_cost = 0
    for jx, tx_key in enumerate(tx_dict.keys()):
        node_out = int(tx_key.split('_')[-2])
        node_in = int(tx_key.split('_')[-1])
        tx_cap_string = f'new_tx_limit_{node_out}_{node_in}_mw'

        new_tx_cost += (np.array(processed_df[tx_cap_string]) - tx_dict[tx_key][0]) * tx_dict[tx_key][1]


    ## Find the total generation cost. Costs are determine for each generation technology
    total_hydro_cost = np.sum([args.hydro_avg_gen_mw[k] * args.hydro_cost_mwh[k] for k in range(args.num_nodes)]) * T
    total_nuclear_cost = args.nuclear_boolean * np.sum([args.nuc_avg_gen_mw[k] * args.nuc_cost_mwh[k] for
                                                        k in range(args.num_nodes)]) * T

    total_biofuel_cost = np.sum([args.biofuel_daily_gen_mwh[k] * args.biofuel_cost_mwh[k]
                                 for k in range(args.num_nodes)]) * T / 24

    total_new_gt_fuel_cost = np.sum(np.array([processed_df[f'gt_new_util_node_{ix+1}_avg_mw'] for ix in range(
                                    args.num_nodes)]).T * np.array(args.gt_fuel_cost_mwh) * T, axis=1)
    total_existing_gt_fuel_cost = np.sum(np.array([processed_df[f'gt_existing_util_node_{ix+1}_avg_mw'] for ix in
                                    range(args.num_nodes)]).T * np.array(args.gt_fuel_cost_mwh) * T, axis=1)

    total_new_gt_ramp_cost = np.array(processed_df['new_gt_ramp_avg_mw']) * args.new_gt_startup_cost_mw/2 * T
    total_existing_gt_ramp_cost = np.array(processed_df['existing_gt_ramp_avg_mw']) * \
                                  args.existing_gt_startup_cost_mw/2 * T

    total_imports_cost = np.sum(np.array([processed_df[f'elec_import_node_{ix+1}_avg_mw'] for ix in
                                 range(args.num_nodes)]).T * np.array(args.import_cost_mwh), axis=1)


    ## Find the total supplmentary costs. Here total costs are the combinations of costs associated with maintaining
    # existing generation capacity and existing transmission
    existing_tx_costs = np.sum([args.existing_trans_cost_mwh[i]*float(args.existing_trans_load_mwh_yr[i]) for i in
                                range(len(args.existing_trans_load_mwh_yr))]) * T/8760
    existing_cap_for_payments = (int(args.nuclear_boolean)*np.array(args.nuc_cap_mw) + np.array(args.hydro_cap_mw) +
                                  np.array(args.biofuel_cap_mw) + np.array(args.existing_gt_cap_mw))
    existing_cap_cost = np.sum(existing_cap_for_payments * np.array(args.cap_market_cost_mw_yr))

    supp_cost = existing_tx_costs + existing_cap_cost

    ## Put together total new capacity + generation costs and return
    new_cap_cost = (new_onshore_costs + new_offshore_cost + new_solar_cost + new_gt_cost + new_batt_cost +
                    new_h2_cost + new_tx_cost)

    generation_cost = (total_hydro_cost + total_nuclear_cost + total_biofuel_cost + total_new_gt_fuel_cost +
                       total_existing_gt_fuel_cost + total_new_gt_ramp_cost + total_existing_gt_ramp_cost +
                       total_imports_cost)

    return new_cap_cost, generation_cost, supp_cost



def raw_results_retrieval(args, m, model_config, scen_ix):
    T = args.num_hours

    baseline_demand_hourly_mw, full_heating_load_hourly_mw, full_heating_load_hourly_mmbtu, \
    full_ev_load_hourly_mw, full_ev_avg_load_hourly_mw, onshore_pot_hourly, offshore_pot_hourly, \
    solar_pot_hourly, btmpv_pot_hourly, fixed_hydro_hourly_mw, flex_hydro_daily_mwh, = load_timeseries(args)

    tx_dict = return_tx_dict(args)


    # If minimizing LCOE, need to transform results, otherwise taken directly from model output by making cf_mult=1
    if model_config == 3:
        cf_mult = m.getVarByName('cc_transform').X
    else:
        cf_mult = 1

    print(cf_mult)

    # BTMPV Capacity
    if args.proj_year == 2019:
        btmpv_cap = args.btmpv_existing_mw
    else:
        btmpv_cap = [btmpv_capacity_projection(args.proj_year) * k for k in args.btmpv_dist]


    cap_columns = ['eheating_rate_node_', 'ev_rate_node_', 'onshore_cap_node_', 'offshore_cap_node_',
                   'solar_cap_node_', 'gt_new_cap_node_', 'batt_energy_cap_node_', 'batt_power_cap_node_',
                   'h2_energy_cap_node_', 'h2_power_cap_node_']

    # Populate the capacity results
    cap_results_df = pd.DataFrame()
    for ix, col in enumerate(cap_columns):
        for jx in range(args.num_nodes):
            column_string = f'{col}{jx+1}'
            cap_results_df[column_string] = [m.getVarByName(column_string).X/cf_mult]

    # Add BTM capacity results
    for ix in range(args.num_nodes):
        cap_results_df[f'btm_cap_node_{ix+1}'] = [btmpv_cap[ix]/cf_mult]


    ts_columns = ['energy_balance_slack_node_', 'flex_hydro_node_', 'batt_charge_node_', 'batt_discharge_node_',
                  'batt_level_node_', 'h2_charge_node_', 'h2_discharge_node_', 'h2_level_node_',
                  'gt_new_util_node_', 'gt_new_diff_node_', 'gt_new_abs_node_', 'gt_existing_util_node_',
                  'gt_existing_diff_node_', 'gt_existing_abs_node_', 'biofuel_util_node_', 'elec_import_node_',
                  'ev_charging_node_']

    ## Populate timeseries Dataframe
    ts_results_df = pd.DataFrame()
    for ix, col in enumerate(ts_columns):
        ts_results_array = np.zeros((T, args.num_nodes))
        if ix == 0: # Collect the energy balance slack ts
            for jx in range(args.num_nodes):
                column_string = f'{col}{jx+1}'
                for kx in range(T):
                    ts_results_array[kx, jx] = m.getConstrByName(f'{column_string}[{kx}]').Slack/cf_mult
                ## Assign curtailment to the results df, changing the sign to +
                ts_results_df[column_string] = -ts_results_array[:, jx]
        else: # Collect the variable values
            for jx in range(args.num_nodes):
                column_string = f'{col}{jx+1}'
                for kx in range(T):
                    ts_results_array[kx, jx] = m.getVarByName(f'{column_string}[{kx}]').X/cf_mult
                ts_results_df[column_string] = ts_results_array[:, jx]


    # Transmission result processing
    tx_new_cap_results   = np.zeros(len(tx_dict))
    tx_ts_results        = np.zeros((T, len(tx_dict)))

    for ix, tx_key in enumerate(tx_dict.keys()):
        node_out = int(tx_key.split('_')[-2])
        node_in = int(tx_key.split('_')[-1])

        new_tx_cap_string = f'new_tx_limit_{node_out}_{node_in}'
        tx_ts_string = f'tx_ts_{node_out}_to_{node_in}'

        tx_new_cap_results[ix] = m.getVarByName(new_tx_cap_string).X/cf_mult
        cap_results_df[new_tx_cap_string] = tx_new_cap_results[ix] + tx_dict[tx_key][0]

        for j in range(T):
            tx_ts_results[j, ix] = m.getVarByName(f'{tx_ts_string}[{j}]').X/cf_mult
        ts_results_df[f'{tx_ts_string}'] = tx_ts_results[:, ix]


    # Find the LCT
    if model_config == 0 or model_config == 1:
        lct = m.getVarByName('lowc_target').X/cf_mult

    else: # model_config == 2 or model_config == 3:
        gas_gen = np.sum([ts_results_df[f'gt_new_util_node_{i+1}'] + ts_results_df[f'gt_existing_util_node_{i+1}']
                          for i in range(args.num_nodes)])/cf_mult

        biofuel_gen = np.sum([ts_results_df[f'biofuel_util_node_{i+1}'] for i in range(args.num_nodes)])/cf_mult

        total_heat_demand = np.sum([cap_results_df[f'eheating_rate_node_{i+1}'] for i in range((args.num_nodes))] *
                                    np.sum(full_heating_load_hourly_mw, axis=0))/cf_mult
        total_ev_demand = np.sum([cap_results_df[f'ev_rate_node_{i+1}'] for i in range((args.num_nodes))] *
                                  np.sum(full_ev_load_hourly_mw, axis=0))/cf_mult
        total_imports = np.sum([ts_results_df[f'elec_import_node_{i+1}'] for i in range(args.num_nodes)])/cf_mult

        demand_for_lct = np.sum(baseline_demand_hourly_mw) + total_heat_demand + total_ev_demand - total_imports

        lct = 1 - (gas_gen + biofuel_gen) / demand_for_lct

    ## Find the electrification ratio
    # Heating electrification rate
    full_therm_heating_load_nodal_avg = np.mean(full_heating_load_hourly_mmbtu, axis=0)
    therm_heating_load_nodal_avg = np.array([full_therm_heating_load_nodal_avg[i] *
                                        (1 - cap_results_df[f'eheating_rate_node_{i+1}']) for i in range(
                                        args.num_nodes)])

    heating_elecfx = 1 - np.sum(therm_heating_load_nodal_avg)/np.sum(full_therm_heating_load_nodal_avg)

    # Find the vehicle electrification rate
    ev_elecfx_nodal_ratios = np.array([args.ev_load_dist[i] * cap_results_df[f'ev_rate_node_{i+1}'] for i in
                                                range(args.num_nodes)])

    veh_elecfx = np.sum(ev_elecfx_nodal_ratios)


    # LCT GHGT elec results
    dghg_target = m.getVarByName('ghg_target').X / cf_mult

    ## Add additional results to the dataframe
    cap_results_df['model_config'] = model_config
    cap_results_df['lct'] = lct
    cap_results_df['ghg_reduction'] = dghg_target
    cap_results_df['heating_elecfx_rate'] = heating_elecfx
    cap_results_df['veh_elecfx_rate'] = veh_elecfx


    results_dir = f'{args.results_dir}/{args.dir_time}'
    cap_dir = f'{results_dir}/cap_results'
    ts_dir = f'{results_dir}/ts_results'

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    if not os.path.exists(cap_dir):
        os.mkdir(cap_dir)
    if not os.path.exists(ts_dir):
        os.mkdir(ts_dir)

    cap_results_save_str = f'{cap_dir}/cap_results_scenix_{scen_ix}.csv'
    ts_results_save_str = f'{ts_dir}/ts_results_scenix_{scen_ix}.csv'

    cap_results_df.round(decimals=3).to_csv(cap_results_save_str)
    ts_results_df.round(decimals=3).to_csv(ts_results_save_str)

    return cap_results_df, ts_results_df


def full_results_processing(args):

    T = args.num_hours

    # Retrieve necessary model timeseries and dictionary of existing transmission parameters
    baseline_demand_hourly_mw, full_heating_load_hourly_mw, full_heating_load_hourly_mmbtu, \
    full_ev_load_hourly_mw, full_ev_avg_load_hourly_mw, onshore_pot_hourly, offshore_pot_hourly, \
    solar_pot_hourly, btmpv_pot_hourly, fixed_hydro_hourly_mw, flex_hydro_daily_mwh = load_timeseries(args)

    tx_dict = return_tx_dict(args)

    # Collect all the raw capacity results, which are saved by scenario
    cap_results_dir = f'{args.results_dir}/{args.dir_time}/cap_results'
    cap_results_csvs = sorted(glob(f'{cap_results_dir}/*scenix*.csv'))

    # Add all the raw capacity results by scenario to a single dataframe. This dataframe contains all the capacity
    # results present in the model run folder
    cap_results_df = pd.DataFrame()
    for file in cap_results_csvs:
        cap_results_df = cap_results_df.append(pd.read_csv(file))

    # Collect results from the raw capacity dataframe
    heating_rate = np.array([cap_results_df[f'eheating_rate_node_{i+1}'] for i in range(args.num_nodes)]).T
    ev_rate  = np.array([cap_results_df[f'ev_rate_node_{i+1}'] for i in range(args.num_nodes)]).T
    onshore_cap = np.array([cap_results_df[f'onshore_cap_node_{i+1}'] for i in range(args.num_nodes)]).T
    offshore_cap = np.array([cap_results_df[f'offshore_cap_node_{i+1}'] for i in range(args.num_nodes)]).T
    solar_cap = np.array([cap_results_df[f'solar_cap_node_{i+1}'] for i in range(args.num_nodes)]).T
    gt_new_cap = np.array([cap_results_df[f'gt_new_cap_node_{i+1}'] for i in range(args.num_nodes)]).T
    battery_energy_cap = np.array([cap_results_df[f'batt_energy_cap_node_{i+1}'] for i in range(args.num_nodes)]).T
    battery_power_cap = np.array([cap_results_df[f'batt_power_cap_node_{i+1}'] for i in range(args.num_nodes)]).T
    h2_energy_cap = np.array([cap_results_df[f'h2_energy_cap_node_{i+1}'] for i in range(args.num_nodes)]).T
    h2_power_cap = np.array([cap_results_df[f'h2_power_cap_node_{i+1}'] for i in range(args.num_nodes)]).T
    btm_cap = np.array([cap_results_df[f'btm_cap_node_{i+1}'] for i in range(args.num_nodes)]).T

    # Determine average electric heating and vehicle demand
    avg_heating_demand = np.sum(heating_rate * np.sum(full_heating_load_hourly_mw, axis=0)/T, axis=1)
    avg_ev_demand = np.sum(ev_rate * np.sum(full_ev_load_hourly_mw, axis=0)/T, axis=1)


    ## Populate processed dataframe
    # Add model run specifications
    processed_df = pd.DataFrame()
    processed_df['model_config'] = cap_results_df['model_config']
    processed_df['rgt/lct'] = cap_results_df['lct']
    processed_df['ghg_reduction'] = cap_results_df['ghg_reduction']
    # Add heating and vehicle electrification rates for the entire region. These are based on the thermal loads of each
    processed_df['heating_elecfx_rate'] = cap_results_df['heating_elecfx_rate']
    processed_df['veh_elecfx_rate'] = cap_results_df['veh_elecfx_rate']
    # Add additional electic heating and vehicle loads
    processed_df['addl_heating_load_mw'] = avg_heating_demand
    processed_df['addl_ev_load_mw'] = avg_ev_demand
    # Continue parameterizing the processed dataframe with other model configuration parameters
    processed_df['rgt_boolean'] = [int(args.rgt_boolean)] * len(cap_results_df)
    processed_df['nuc_boolean'] = [int(args.nuclear_boolean)] * len(cap_results_df)
    processed_df['h2_boolean'] = [int(args.h2_boolean)] * len(cap_results_df)
    processed_df['same_eheating_ev_rate_boolean'] = [int(args.same_eheating_ev_rate_boolean)] * len(cap_results_df)
    processed_df['btmpv_count_re'] = [int(args.btmpv_count_re)] * len(cap_results_df)
    processed_df['elec_constraint_ge'] = [int(args.elecfx_constraint_ge)] * len(cap_results_df)
    processed_df['gt_based_on_current'] = [int(args.gt_based_on_current)] * len(cap_results_df)
    processed_df['ev_set_profile_boolean'] = [int(args.ev_set_profile_boolean)] * len(cap_results_df)
    # Add the electrification rates by node and by EV/heating
    for ix in range(args.num_nodes):
        processed_df[f'heating_elecfx_rate_node_{ix+1}'] = cap_results_df[f'eheating_rate_node_{ix+1}']
    for ix in range(args.num_nodes):
        processed_df[f'ev_elecfx_rate_node_{ix+1}'] = cap_results_df[f'ev_rate_node_{ix+1}']
    # Add returned capacities of new infrastructure
    processed_df['onshore_cap_mw'] = np.sum(onshore_cap, axis=1)
    processed_df['offshore_cap_mw'] = np.sum(offshore_cap, axis=1)
    processed_df['solar_cap_mw'] = np.sum(solar_cap, axis=1)
    processed_df['new_gt_cap_mw'] = np.sum(gt_new_cap, axis=1)
    processed_df['battery_energy_cap_mwh'] = np.sum(battery_energy_cap, axis=1)
    processed_df['battery_power_cap_mw'] = np.sum(battery_power_cap, axis=1)
    processed_df['h2_energy_cap_mwh'] = np.sum(h2_energy_cap, axis=1)
    processed_df['h2_power_cap_mw'] = np.sum(h2_power_cap, axis=1)
    processed_df['btm_power_cap_mw'] = np.sum(btm_cap, axis=1)
    # Add new transmission limits
    for ix, tx_key in enumerate(tx_dict.keys()):
        new_tx_cap_string = f'new_tx_limit_{int(tx_key.split("_")[-2])}_{int(tx_key.split("_")[-1])}'
        processed_df[new_tx_cap_string+'_mw'] = cap_results_df[new_tx_cap_string]

    # Load results that depend on timeseries
    processed_df = load_ts_based_results(args, processed_df)

    # Load total costs
    new_cap_cost, generation_cost, supp_cost = cost_calculations(args, cap_results_df, processed_df)

    # Find total MWh for LCOE calculations
    btm_avg_mwh = np.sum(btm_cap * np.mean(btmpv_pot_hourly, axis=0), axis=1)
    total_mwh_for_lcoe = np.sum(baseline_demand_hourly_mw) + (avg_heating_demand + avg_ev_demand - btm_avg_mwh) * T

    ## Add LCOEs to dataframe
    processed_df['new_cap_lcoe']    = new_cap_cost/total_mwh_for_lcoe
    processed_df['generation_lcoe'] = generation_cost/total_mwh_for_lcoe
    processed_df['supp_cost_lcoe']  = supp_cost/total_mwh_for_lcoe
    processed_df['total_lcoe'] = (new_cap_cost + generation_cost + supp_cost)/total_mwh_for_lcoe

    # Write out the processed dataframe!
    processed_df_filename = f'{args.results_dir}/{args.dir_time}/processed_results_{args.dir_time}.csv'
    processed_df.to_csv(processed_df_filename)


if __name__ == '__main__':
    args = get_args()
    args.__dict__['dir_time'] = '20210220-105808'


    full_results_processing(args)