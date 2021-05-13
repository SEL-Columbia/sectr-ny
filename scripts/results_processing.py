import os
import math
import argparse
import yaml
import numpy as np
import pandas as pd
from datetime import datetime as dt
import datetime
from glob import glob
from scripts.utils import (get_args, load_timeseries, return_tx_dict,
                   btmpv_capacity_projection, return_costs_for_model)


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

    total_biofuel_cost = np.sum(np.array([processed_df[f'biofuel_util_node_{ix+1}_avg_mw'] for ix in range(
                                    args.num_nodes)]).T * np.array(args.biofuel_cost_mwh) * T, axis=1)

    total_new_gt_fuel_cost = np.sum(np.array([processed_df[f'gt_new_util_node_{ix+1}_avg_mw'] for ix in range(
                                    args.num_nodes)]).T * cost_dict['new_gt_cost_mwh'] * T, axis=1)

    total_existing_gt_fuel_cost = np.sum(np.array([processed_df[f'gt_existing_util_node_{ix+1}_avg_mw'] for ix in
                                    range(args.num_nodes)]).T * cost_dict['existing_gt_cost_mwh'] * T, axis=1)

    total_new_gt_ramp_cost = np.array(processed_df['new_gt_ramp_avg_mw']) * args.new_gt_startup_cost_mw/2 * T
    total_existing_gt_ramp_cost = np.array(processed_df['existing_gt_ramp_avg_mw']) * \
                                  args.existing_gt_startup_cost_mw/2 * T

    total_imports_cost = np.sum(np.array([processed_df[f'elec_import_node_{ix+1}_avg_mw'] for ix in
                                 range(args.num_nodes)]).T * np.array(args.import_cost_mwh) * T, axis=1)



    ## Find the total supplmentary costs. Here total costs are the combinations of costs associated with maintaining
    # existing generation capacity and existing transmission
    existing_tx_costs = np.sum([args.existing_trans_cost_mwh[i]*float(args.existing_trans_load_mwh_yr[i]) for i in
                                range(len(args.existing_trans_load_mwh_yr))]) * args.num_years
    existing_cap_for_payments = (int(args.nuclear_boolean)*np.array(args.nuc_cap_mw) + np.array(args.hydro_cap_mw) +
                                  np.array(args.biofuel_cap_mw) + np.array(args.existing_gt_cap_mw))
    existing_cap_cost = np.sum(existing_cap_for_payments * np.array(args.cap_market_cost_mw_yr)) * args.num_years

    supp_cost = existing_tx_costs + existing_cap_cost


    ## Put together total new capacity + generation costs and return
    new_cap_cost = (new_onshore_costs + new_offshore_cost + new_solar_cost + new_gt_cost + new_batt_cost +
                    new_h2_cost + new_tx_cost)


    generation_cost = (total_hydro_cost + total_nuclear_cost + total_biofuel_cost + total_new_gt_fuel_cost +
                       total_existing_gt_fuel_cost + total_new_gt_ramp_cost + total_existing_gt_ramp_cost +
                       total_imports_cost)

    ## Find wind, solar LCOEs based on generation and curtailment
    # Calculate the LCOES of wind and solar


    processed_df['onshore_gen_lcoe'] =  np.maximum(new_onshore_costs/ \
        np.array((processed_df['onshore_uc_gen_mw'] - processed_df['onshore_curtailed_gen_mw']) * T), 0)

    processed_df['offshore_gen_lcoe'] = np.maximum(new_offshore_cost /\
        np.array((processed_df['offshore_uc_gen_mw'] - processed_df['offshore_curtailed_gen_mw']) * T), 0)

    processed_df['solar_gen_lcoe'] = np.maximum(new_solar_cost /\
        np.array((processed_df['solar_uc_gen_mw'] - processed_df['solar_curtailed_gen_mw']) * T), 0)

    processed_df['wind_solar_gen_lcoe'] = (new_onshore_costs + new_offshore_cost + new_solar_cost) / \
        np.array((processed_df[f'wind_solar_uc_gen_regional_avg_mw'] - processed_df['curtailment_regional_avg_mw']) * T)


    # Find gas LCOE
    existing_gas_gen_cost = np.sum((np.array(args.existing_gt_cap_mw) * np.array(args.cap_market_cost_mw_yr)) *
                                   args.num_years)
    processed_df['gas_gen_lcoe'] = (new_gt_cost + total_new_gt_fuel_cost + total_existing_gt_fuel_cost +
                                    total_new_gt_ramp_cost + total_existing_gt_ramp_cost + existing_gas_gen_cost) / \
                                   ((processed_df['gt_new_util_regional_avg_mw'] +
                                     processed_df['gt_existing_util_regional_avg_mw']) * T)

    ## Find battery cost per MWh throughput

    processed_df['battery_cost_per_mwh_discharge'] = new_batt_cost/(processed_df['battery_discharge_regional_avg_mw'] * T)


    return new_cap_cost, generation_cost, supp_cost

def allocate_curtailment(args, ts_results_df):

    onshore_df = pd.DataFrame()
    offshore_df = pd.DataFrame()
    solar_df = pd.DataFrame()

    for ix in range(args.num_nodes):
        frac_curtail = ts_results_df[f'energy_balance_slack_node_{ix+1}'] / \
                       (ts_results_df[f'onshore_uc_gen_node_{ix+1}'] + ts_results_df[f'offshore_uc_gen_node_{ix+1}'] +
                        ts_results_df[f'solar_uc_gen_node_{ix+1}'])


        onshore_df[f'onshore_curtailed_gen_node_{ix+1}'] = frac_curtail * ts_results_df[f'onshore_uc_gen_node_{ix+1}']
        offshore_df[f'offshore_curtailed_gen_node_{ix+1}'] = frac_curtail * ts_results_df[f'offshore_uc_gen_node_{ix+1}']
        solar_df[f'solar_curtailed_gen_node_{ix+1}'] = frac_curtail * ts_results_df[f'solar_uc_gen_node_{ix+1}']



    ts_results_df = ts_results_df.join(onshore_df).join(offshore_df).join(solar_df)
    ts_results_df = ts_results_df.fillna(0)


    return ts_results_df

def load_ts_based_results(args, processed_df):
    '''
    Function for producing results based on the timeseries outputs of the model

    :param args: args dictionary
    :param processed_df: Dataframe for storing the processed results of the model solution
    :return: processed_df, with results from the timeseries data added
    '''
    T = args.num_hours
    dt_start = datetime.datetime(year=2007, month=1, day=1, hour=0)
    dt_delta = datetime.timedelta(hours=1)

    # Collect transmission array data
    tx_dict = return_tx_dict(args)

    # Get cost dict
    cost_dict = return_costs_for_model(args)

    # Load potential timeseries for curtailment calcs
    baseline_demand_hourly_mw, full_elec_heating_load_hourly_mw, full_ff_heating_load_hourly_mw, \
    full_ff_dss50_hourly_mw, full_ev_load_hourly_mw, full_ev_avg_load_hourly_mw, onshore_pot_hourly, \
    offshore_pot_hourly, solar_pot_hourly, btmpv_pot_hourly, fixed_hydro_hourly_mw, \
    flex_hydro_daily_mwh = load_timeseries(args)

    # Load all the timeseries files present in the corresponding folder
    ts_results_dir = f'{args.results_dir}/{args.dir_time}/ts_results'
    ts_results_files = sorted(glob(f'{ts_results_dir}/*.csv'))

    ## Create arrays to store the timeseries results before assigning to the processed Dataframe
    total_gt_new_util = np.zeros((len(ts_results_files), args.num_nodes))
    total_gt_existing_util = np.zeros((len(ts_results_files), args.num_nodes))
    total_gt_new_ramping = np.zeros(len(ts_results_files))
    total_gt_existing_ramping = np.zeros(len(ts_results_files))

    gt_new_max_regional_util = np.zeros(len(ts_results_files))
    gt_existing_max_regional_util = np.zeros(len(ts_results_files))
    gt_total_max_regional_util = np.zeros(len(ts_results_files))

    gt_new_peak_load_by_node = np.zeros((len(ts_results_files), args.num_nodes))
    gt_existing_peak_load_by_node = np.zeros((len(ts_results_files), args.num_nodes))
    biofuel_peak_load_by_node     = np.zeros((len(ts_results_files), args.num_nodes))

    peak_regional_demand = np.zeros(len(ts_results_files))
    peak_nodal_demand = np.zeros((len(ts_results_files), args.num_nodes))
    solar_wind_gen_regional_avg_mw = np.zeros(len(ts_results_files))
    solar_wind_gen_nodal_avg_mw = np.zeros((len(ts_results_files), args.num_nodes))
    solar_wind_gen_min_mw = np.zeros(len(ts_results_files))
    min_solar_wind_gen_datetime = []

    curtailment = np.zeros((len(ts_results_files), args.num_nodes))
    biofuel_util = np.zeros((len(ts_results_files), args.num_nodes))
    battery_charge = np.zeros((len(ts_results_files), args.num_nodes))
    battery_discharge = np.zeros((len(ts_results_files), args.num_nodes))
    h2_charge = np.zeros((len(ts_results_files), args.num_nodes))
    h2_discharge = np.zeros((len(ts_results_files), args.num_nodes))
    elec_import = np.zeros((len(ts_results_files), args.num_nodes))
    hydro_gen = np.zeros((len(ts_results_files), args.num_nodes))
    tx_util = np.zeros((len(ts_results_files), len(tx_dict)))
    tx_nodal_avg_import = np.zeros((len(ts_results_files), args.num_nodes)) # transmission import sum for each node

    battery_losses = np.zeros((len(ts_results_files), args.num_nodes))
    h2_losses      = np.zeros((len(ts_results_files), args.num_nodes))
    tx_losses      = np.zeros((len(ts_results_files), args.num_nodes))

    onshore_curtailed_mw = np.zeros((len(ts_results_files), args.num_nodes))
    offshore_curtailed_mw = np.zeros((len(ts_results_files), args.num_nodes))
    solar_curtailed_mw   = np.zeros((len(ts_results_files), args.num_nodes))

    onshore_uc_gen_mw = np.zeros((len(ts_results_files), args.num_nodes))
    offshore_uc_gen_mw = np.zeros((len(ts_results_files), args.num_nodes))
    solar_uc_gen_mw = np.zeros((len(ts_results_files), args.num_nodes))

    excess_lowc_gen = np.zeros(len(ts_results_files))

    # Iterate through the ts results files, each corresponding to a single scenario
    for ix, file in enumerate(ts_results_files):
        # Load the csv
        ts_csv = pd.read_csv(file)

        ## Find demand timeseries
        baseline_demand = np.array([ts_csv[f'baseline_demand_node_{i+1}'] for i in range(args.num_nodes)]).T
        heating_demand = np.array([ts_csv[f'heating_demand_node_{i+1}'] for i in range(args.num_nodes)]).T
        ev_demand = np.array([ts_csv[f'ev_charging_node_{i+1}'] for i in range(args.num_nodes)]).T

        # Find demand peaks
        peak_regional_demand[ix] = np.max(np.sum((baseline_demand + heating_demand + ev_demand), axis=1))
        peak_nodal_demand[ix] = np.max((baseline_demand + heating_demand + ev_demand), axis=0)

        # Find solar and wind based results
        onshore_uc_gen = np.array([ts_csv[f'onshore_uc_gen_node_{i+1}'] for i in range(args.num_nodes)]).T
        offshore_uc_gen = np.array([ts_csv[f'offshore_uc_gen_node_{i+1}'] for i in range(args.num_nodes)]).T
        solar_uc_gen = np.array([ts_csv[f'solar_uc_gen_node_{i+1}'] for i in range(args.num_nodes)]).T

        # Find average stats
        total_gen = np.sum((onshore_uc_gen + offshore_uc_gen + solar_uc_gen), axis=1)
        solar_wind_gen_regional_avg_mw[ix] = np.mean(total_gen)
        solar_wind_gen_nodal_avg_mw[ix] = np.mean((onshore_uc_gen + offshore_uc_gen + solar_uc_gen), axis=0)

        # Extract minimum and find the datetime that it occurs
        solar_wind_gen_min_mw[ix] = np.min(total_gen)
        min_ix = np.argwhere(total_gen == np.min(total_gen))[0][0]
        min_dt = datetime.datetime.strftime(dt_start + min_ix * dt_delta, '%m/%d/%Y %H:%M')
        min_solar_wind_gen_datetime.append(min_dt)

        # Collect GT util data
        gt_new_util = np.array([ts_csv[f'gt_new_util_node_{i+1}'] for i in range(args.num_nodes)]).T
        gt_existing_util = np.array([ts_csv[f'gt_existing_util_node_{i+1}'] for i in range(args.num_nodes)]).T

        # Find regional GT utilization maximums
        gt_new_max_regional_util[ix] = np.max(np.sum(gt_new_util, axis=1))
        gt_existing_max_regional_util[ix] = np.max(np.sum(gt_existing_util, axis=1))
        gt_total_max_regional_util[ix] = np.max(np.sum(gt_new_util + gt_existing_util, axis=1))

        # Find total GT utilization and ramping
        total_gt_new_util[ix] = np.sum(gt_new_util, axis=0)
        total_gt_existing_util[ix] = np.sum(gt_existing_util, axis=0)
        total_gt_new_ramping[ix] = np.sum(np.array([ts_csv[f'gt_new_abs_node_{i+1}']
                                                    for i in range(args.num_nodes)]).T)
        total_gt_existing_ramping[ix] = np.sum(np.array([ts_csv[f'gt_existing_abs_node_{i+1}']
                                                    for i in range(args.num_nodes)]).T)

        # Find the existing GT, biofuel peak load by node 
        gt_new_peak_load_by_node[ix] = np.max(gt_new_util, axis=0)
        gt_existing_peak_load_by_node[ix] = np.max(gt_existing_util, axis=0)
        biofuel_peak_load_by_node[ix] = np.max(np.array([ts_csv[f'biofuel_util_node_{i+1}'] 
                                            for i in range(args.num_nodes)]).T, axis=0)

        # Find curtailment, battery_discharge, elec_import
        curtailment[ix] = np.sum(np.array([ts_csv[f'energy_balance_slack_node_{i+1}']
                                           for i in range(args.num_nodes)]).T, axis=0)
        battery_charge[ix] = np.sum(np.array([ts_csv[f'batt_charge_node_{i + 1}'] for i in
                                                 range(args.num_nodes)]).T, axis=0)
        battery_discharge[ix] = np.sum(np.array([ts_csv[f'batt_discharge_node_{i+1}'] for i in
                                                 range(args.num_nodes)]).T, axis=0)
        h2_charge[ix] = np.sum(np.array([ts_csv[f'h2_charge_node_{i+1}'] for i in range(args.num_nodes)]).T,
                                  axis=0)
        h2_discharge[ix] = np.sum(np.array([ts_csv[f'h2_discharge_node_{i+1}'] for i in range(args.num_nodes)]).T,
                                  axis=0)
        elec_import[ix] = np.sum(np.array([ts_csv[f'elec_import_node_{i+1}'] for i in range(args.num_nodes)]).T,
                                 axis=0)
        biofuel_util[ix] = np.sum(np.array([ts_csv[f'biofuel_util_node_{i+1}'] for i in range(args.num_nodes)]).T,
                                 axis=0)

        # Battery / h2 losses
        battery_losses[ix] = np.sum(np.array([ts_csv[f'batt_discharge_node_{i+1}'] for i in range(args.num_nodes)]).T,
                                    axis=0) * (1-args.battery_eff) + \
                             np.sum(np.array([ts_csv[f'batt_charge_node_{i+1}'] for i in range(args.num_nodes)]).T,
                                    axis=0) * (1/args.battery_eff-1) + \
                             np.sum(np.array([ts_csv[f'batt_level_node_{i+1}'] for i in range(args.num_nodes)]).T,
                                    axis=0) * args.battery_self_discharge

        h2_losses[ix]      = np.sum(np.array([ts_csv[f'h2_discharge_node_{i+1}'] for i in range(args.num_nodes)]).T,
                                    axis=0) * (1-args.h2_eff) + \
                             np.sum(np.array([ts_csv[f'h2_charge_node_{i+1}'] for i in range(args.num_nodes)]).T,
                                    axis=0) * (1/args.h2_eff-1) + \
                             np.sum(np.array([ts_csv[f'h2_level_node_{i+1}'] for i in range(args.num_nodes)]).T,
                                    axis=0) * args.h2_self_discharge

        # Hydropower generation
        hydro_gen[ix] = (np.sum(np.array([ts_csv[f'fixed_hydro_gen_node_{i+1}'] for i in range(args.num_nodes)]).T,
                                  axis=0) +
                         np.sum(np.array([ts_csv[f'flex_hydro_node_{i+1}'] for i in range(args.num_nodes)]).T,
                                  axis=0))

        # Determine generation and curtailment sums

        onshore_curtailed_mw[ix] = np.sum(np.array([ts_csv[f'onshore_curtailed_gen_node_{i+1}'] for i
                                                             in range(args.num_nodes)]).T, axis=0)
        offshore_curtailed_mw[ix] = np.sum(np.array([ts_csv[f'offshore_curtailed_gen_node_{i+1}'] for i
                                                    in range(args.num_nodes)]).T, axis=0)
        solar_curtailed_mw[ix] = np.sum(np.array([ts_csv[f'solar_curtailed_gen_node_{i+1}'] for i
                                                    in range(args.num_nodes)]).T, axis=0)

        onshore_uc_gen_mw[ix] = np.sum(np.array([ts_csv[f'onshore_uc_gen_node_{i+1}'] for i
                                                             in range(args.num_nodes)]).T, axis=0)
        offshore_uc_gen_mw[ix] = np.sum(np.array([ts_csv[f'offshore_uc_gen_node_{i+1}'] for i
                                                 in range(args.num_nodes)]).T, axis=0)
        solar_uc_gen_mw[ix] = np.sum(np.array([ts_csv[f'solar_uc_gen_node_{i + 1}'] for i
                                                 in range(args.num_nodes)]).T, axis=0)

        ## Find excess generation (equal to curtailment plus battery charging)
        demand = (np.sum(ts_csv.loc[:, 'baseline_demand_node_1':'baseline_demand_node_4'], axis=1) +
                  np.sum(ts_csv.loc[:, 'ev_charging_node_1':'ev_charging_node_4'], axis=1) +
                  np.sum(ts_csv.loc[:, 'heating_demand_node_1':'heating_demand_node_4'], axis=1))

        imports_and_btm = (np.sum(ts_csv.loc[:, 'elec_import_node_1':'elec_import_node_4'], axis=1) +
                           np.sum(ts_csv.loc[:, 'btmpv_uc_gen_node_1':'btmpv_uc_gen_node_4'], axis=1))

        wind_and_solar = (np.sum(ts_csv.loc[:, 'onshore_uc_gen_node_1':'onshore_uc_gen_node_4'], axis=1) +
                          np.sum(ts_csv.loc[:, 'offshore_uc_gen_node_1':'offshore_uc_gen_node_4'], axis=1) +
                          np.sum(ts_csv.loc[:, 'solar_uc_gen_node_1':'solar_uc_gen_node_4'], axis=1))

        hydro = (np.sum(ts_csv.loc[:, 'fixed_hydro_gen_node_1':'fixed_hydro_gen_node_4'], axis=1) +
                 np.sum(ts_csv.loc[:, 'flex_hydro_node_1':'flex_hydro_node_2'], axis=1))

        gas = np.round((np.sum(ts_csv.loc[:, 'gt_new_util_node_1':'gt_new_util_node_4'], axis=1) +
                        np.sum(ts_csv.loc[:, 'gt_existing_util_node_1':'gt_existing_util_node_4'], axis=1)))

        diff = demand - imports_and_btm - wind_and_solar - hydro

        pos_days = np.argwhere(diff > 0)[:, 0]
        neg_days = np.argwhere(diff < 0)[:, 0]
        gas_days = np.argwhere(gas == 0)[:, 0]

        excess_lowc_gen[ix] = -np.sum(diff[gas_days]) / T

        # curtail = np.sum(ts_csv.loc[:, 'energy_balance_slack_node_1':  'energy_balance_slack_node_4'], axis=1)
        # batt_charging =  np.sum(ts_csv.loc[:, 'batt_charge_node_1':'batt_charge_node_4'], axis=1)
        # excess_lowc_gen[ix] = np.mean(curtail + batt_charging)

        # Find tx flow
        tx_output_cols = []
        for jx, tx_key in enumerate(tx_dict.keys()):
            node_out = int(tx_key.split('_')[-2])
            node_in = int(tx_key.split('_')[-1])

            tx_ts_string = f'tx_ts_{node_out}_to_{node_in}'
            tx_output_cols.append(f'tx_avg_util_{node_out}_to_{node_in}')

            tx_util[ix, jx] = np.mean(ts_csv[tx_ts_string])

            # sum up the transmission import for each ndoe
            tx_nodal_avg_import[ix, node_in-1] += tx_util[ix, jx]

        # find the transmission losses
        tx_losses[ix, :] = tx_nodal_avg_import[ix, :] * args.trans_loss

    ## Begin populating the processed dataframe with stats from the timeseries

    # Add peak load
    processed_df['peak_demand_regional_mw'] = peak_regional_demand
    for ix in range(args.num_nodes):
        processed_df[f'peak_demand_node_{ix+1}_mw'] = peak_nodal_demand[:, ix]

    # Add hydro generation
    processed_df['hydro_gen_regional_mw'] = np.sum(hydro_gen, axis=1) /T
    for ix in range(args.num_nodes):
        processed_df[f'hydro_gen_node_{ix+1}_mw'] = hydro_gen[:, ix] / T

    ## Add uncurtailed generation from wind + solar
    processed_df[f'wind_solar_uc_gen_regional_avg_mw'] = solar_wind_gen_regional_avg_mw
    for ix in range(args.num_nodes):
        processed_df[f'wind_solar_uc_gen_node_{ix+1}_mw'] = solar_wind_gen_nodal_avg_mw[:, ix]

    # Find minimum generation and when it occurs
    processed_df[f'wind_solar_uc_gen_regional_min_mw'] = solar_wind_gen_min_mw
    processed_df[f'wind_solar_uc_gen_regional_min_datetime'] = min_solar_wind_gen_datetime

    # Add curtailment calculations, regional and by node
    processed_df['curtailment_regional_avg_mw'] = np.sum(curtailment, axis=1)/T
    for ix in range(args.num_nodes):
        processed_df[f'curtailment_node_{ix + 1}_avg_mw'] = curtailment[:, ix] / T

    # Add curtailment %
    processed_df['curtailment_regional_avg_%'] = processed_df['curtailment_regional_avg_mw'] / \
                                                 processed_df[f'wind_solar_uc_gen_regional_avg_mw'] * 100

    # Add resource specific curtailment calculations
    processed_df['onshore_curtailed_gen_mw'] = np.sum(onshore_curtailed_mw, axis=1)/T
    processed_df['offshore_curtailed_gen_mw'] = np.sum(offshore_curtailed_mw, axis=1)/T
    processed_df['solar_curtailed_gen_mw'] = np.sum(solar_curtailed_mw, axis=1)/T

    # Add uncurtailed gen
    processed_df['onshore_uc_gen_mw'] = np.sum(onshore_uc_gen_mw, axis=1) / T
    processed_df['offshore_uc_gen_mw'] = np.sum(offshore_uc_gen_mw, axis=1) / T
    processed_df['solar_uc_gen_mw'] = np.sum(solar_uc_gen_mw, axis=1) / T

    # Add curtailment percents
    processed_df['onshore_curtailment_%'] = (100 * processed_df['onshore_curtailed_gen_mw'].round(decimals=0) /
                                             processed_df['onshore_uc_gen_mw'])
    processed_df['offshore_curtailment_%'] = (100 * processed_df['offshore_curtailed_gen_mw'].round(decimals=0) /
                                             processed_df['offshore_uc_gen_mw'])
    processed_df['solar_curtailment_%'] = (100 * processed_df['solar_curtailed_gen_mw'].round(decimals=0) /
                                             processed_df['solar_uc_gen_mw'])

    ## New GT
    # Add average new GT utilization results, regional and by node
    processed_df['gt_new_util_regional_avg_mw'] = np.sum(total_gt_new_util, axis=1) / T
    for ix in range(args.num_nodes):
        processed_df[f'gt_new_util_node_{ix+1}_avg_mw'] = total_gt_new_util[:, ix]/T

    # Add max new GT utilization results, regional and by node
    processed_df['gt_new_util_regional_max_mw'] = gt_new_max_regional_util
    for ix in range(args.num_nodes):
        processed_df[f'gt_new_util_node_{ix+1}_max_mw'] = gt_new_peak_load_by_node[:, ix]

    ## Existing GT
    # Add average existing GT utilization results, regional and by node
    processed_df['gt_existing_util_regional_avg_mw'] = np.sum(total_gt_existing_util, axis=1) / T
    for ix in range(args.num_nodes):
        processed_df[f'gt_existing_util_node_{ix+1}_avg_mw'] = total_gt_existing_util[:, ix]/T

    # Add maximum existing GT utilization results, by node
    processed_df['gt_existing_util_regional_max_mw'] = gt_existing_max_regional_util
    for ix in range(args.num_nodes):
        processed_df[f'gt_existing_util_node_{ix+1}_max_mw'] = gt_existing_peak_load_by_node[:, ix]

    ## Add GT CF results
    # New GT
    processed_df['gt_new_util_regional_cf_%'] = (processed_df['gt_new_util_regional_avg_mw'] /
                                               processed_df['new_gt_cap_mw']).fillna(0) * 100
    # Existing GT
    processed_df['gt_existing_util_regional_cf_%'] = (processed_df['gt_existing_util_regional_avg_mw'] /
                                                    processed_df['existing_gt_cap_mw']).fillna(0) * 100

    ## Add ramping
    # Add averaging ramping results for the full region
    processed_df['new_gt_ramp_avg_mw'] = total_gt_new_ramping/T
    processed_df['existing_gt_ramp_avg_mw'] = total_gt_existing_ramping/T

    ## Add upstate and downstate GT utilization
    processed_df['new_gt_util_upstate_cf_%'] = (np.sum(processed_df.loc[:, 'gt_new_util_node_1_avg_mw':
                                                                       'gt_new_util_node_2_avg_mw'], axis=1) /
                                            processed_df['new_gt_cap_upstate_mw']) * 100
    processed_df['new_gt_util_downstate_cf_%'] = (np.sum(processed_df.loc[:, 'gt_new_util_node_3_avg_mw':
                                                                        'gt_new_util_node_4_avg_mw'], axis=1) /
                                             processed_df['new_gt_cap_downstate_mw']) * 100
    processed_df['existing_gt_util_upstate_cf_%'] = (np.sum(processed_df.loc[:, 'gt_existing_util_node_1_avg_mw':
                                                                        'gt_existing_util_node_2_avg_mw'], axis=1) /
                                             processed_df['existing_gt_cap_upstate_mw']) * 100
    processed_df['existing_gt_util_downstate_cf_%'] = (np.sum(processed_df.loc[:, 'gt_existing_util_node_3_avg_mw':
                                                                          'gt_existing_util_node_4_avg_mw'], axis=1) /
                                               processed_df['existing_gt_cap_downstate_mw']) * 100

    # Add biofuel utilization, regional by node
    processed_df['biofuel_util_regional_avg_mw'] = np.sum(biofuel_util, axis=1) / T
    for ix in range(args.num_nodes):
        processed_df[f'biofuel_util_node_{ix+1}_avg_mw'] = biofuel_util[:, ix]/T
    # # Add maximum biofuel load results, by node
    # for ix in range(args.num_nodes):
    #     processed_df[f'biofuel_peak_load_node_{ix+1}_mw'] = biofuel_peak_load_by_node[:, ix]

    # Add battery charge, regional and by node
    processed_df['battery_charge_regional_avg_mw'] = np.sum(battery_charge, axis=1) / T
    for ix in range(args.num_nodes):
        processed_df[f'battery_charge_node_{ix+1}_avg_mw'] = battery_charge[:, ix] / T

    # Add battery discharge, regional and by node
    processed_df['battery_discharge_regional_avg_mw'] = np.sum(battery_discharge, axis=1)/T
    for ix in range(args.num_nodes):
        processed_df[f'battery_discharge_node_{ix+1}_avg_mw'] = battery_discharge[:, ix]/T

    # Add battery cycling, regional and by node
    battery_throughput_regional = (processed_df['battery_charge_regional_avg_mw'] +
                                   processed_df['battery_discharge_regional_avg_mw'])/2
    battery_throughput_nodal = (battery_charge + battery_discharge)/(2*T)

    processed_df['battery_regional_full_cycles_yr'] = (battery_throughput_regional * 8760 /
                                                  (2*processed_df['battery_energy_cap_mwh']))
    for ix in range(args.num_nodes):
        processed_df[f'battery_node_{ix+1}_full_cycles_yr'] = (battery_throughput_nodal[:, ix] * 8760 /
                                                          (2*processed_df[f'batt_energy_cap_node_{ix+1}_mwh']))

    # Add average H2 charge, regional and by node
    processed_df['h2_charge_regional_avg_mw'] = np.sum(h2_charge, axis=1) / T
    for ix in range(args.num_nodes):
        processed_df[f'h2_charge_node_{ix+1}_avg_mw'] = h2_charge[:, ix] / T

    # Add average H2 discharge, regional and by node
    processed_df['h2_discharge_regional_avg_mw'] = np.sum(h2_discharge, axis=1) / T
    for ix in range(args.num_nodes):
        processed_df[f'h2_discharge_node_{ix + 1}_avg_mw'] = h2_discharge[:, ix] / T

    # Add H2 cycling, regional and by node
    h2_throughput_regional = (processed_df['h2_charge_regional_avg_mw'] +
                              processed_df['h2_discharge_regional_avg_mw']) / 2
    h2_throughput_nodal = (h2_charge + h2_discharge) / (2 * T)

    processed_df['h2_regional_full_cycles_yr'] = (h2_throughput_regional * 8760 /
                                             (2*processed_df['h2_energy_cap_mwh']))
    for ix in range(args.num_nodes):
        processed_df[f'h2_node_{ix+1}_full_cycles_yr'] = (h2_throughput_nodal[:, ix] * 8760 /
                                                    (2*processed_df[f'h2_energy_cap_node_{ix+1}_mwh']))

    # Add average electricity import, region and by node
    processed_df['elec_import_regional_avg_mw'] = np.sum(elec_import, axis=1) / T
    for ix in range(args.num_nodes):
        processed_df[f'elec_import_node_{ix+1}_avg_mw'] = elec_import[:, ix]/T

    # Add average utilization of transmission lines
    for jx, tx_key in enumerate(tx_dict.keys()):
        tx_output_col = f'tx_avg_util_{tx_key.split("_")[-2]}_to_{tx_key.split("_")[-1]}_mw'
        processed_df[tx_output_col] = tx_util[:, jx]

    # Find average utilization of positive and negative transmission
    tx_pos_util = 0
    tx_neg_util = 0
    tx_pos_cap = 0
    tx_neg_cap = 0

    for jx, tx_key in enumerate(tx_dict.keys()):
        out_node = int(tx_key.split("_")[-2])
        in_node = int(tx_key.split("_")[-1])

        if out_node < in_node:
            tx_pos_util += processed_df[f'tx_avg_util_{out_node}_to_{in_node}_mw']
            tx_pos_cap += processed_df[f'new_tx_limit_{out_node}_{in_node}_mw']
        else:
            tx_neg_util += processed_df[f'tx_avg_util_{out_node}_to_{in_node}_mw']
            tx_neg_cap += processed_df[f'new_tx_limit_{out_node}_{in_node}_mw']

    processed_df['tx_pos_avg_util_%'] = tx_pos_util/tx_pos_cap * 100
    processed_df['tx_neg_avg_util_%'] = tx_neg_util/tx_neg_cap * 100


    # Add average losses from battery and h2, regional and by node
    processed_df['battery_losses_regional_avg_mw'] = np.sum(battery_losses, axis=1)/T
    for ix in range(args.num_nodes):
        processed_df[f'battery_losses_node_{ix+1}_avg_mw'] = battery_losses[:, ix]/T
    processed_df['h2_losses_regional_avg_mw'] = np.sum(h2_losses, axis=1)/T
    for ix in range(args.num_nodes):
        processed_df[f'h2_losses_node_{ix+1}_avg_mw'] = h2_losses[:, ix]/T

    # Add average losses from transmission, regional and by node
    processed_df[f'tx_losses_regional_avg_mw'] = np.sum(tx_losses, axis=1)
    for ix in range(args.num_nodes):
        processed_df[f'tx_losses_node_{ix+1}_avg_mw'] = tx_losses[:, ix]

    ## Add excess low-carbon generation
    processed_df['excess_lowc_gen_regional_avg_mw'] = excess_lowc_gen
    # Fill NA values
    processed_df = processed_df.fillna('NaN')

    return processed_df


def raw_results_retrieval(args, m, model_config, scen_ix):
    T = args.num_hours

    baseline_demand_hourly_mw, full_heating_load_hourly_mw, full_ff_heating_load_hourly_mw, \
    full_ff_dss50_hourly_mw, full_ev_load_hourly_mw, full_ev_avg_load_hourly_mw, onshore_pot_hourly, \
    offshore_pot_hourly, solar_pot_hourly, btmpv_pot_hourly, fixed_hydro_hourly_mw, \
    flex_hydro_daily_mwh = load_timeseries(args)

    tx_dict = return_tx_dict(args)


    # If minimizing LCOE, need to transform results, otherwise taken directly from model output by making cf_mult=1
    if model_config == 3:
        cf_mult = m.getVarByName('cc_transform').X
    else:
        cf_mult = 1

    # BTMPV Capacity
    if args.proj_year == 2019:
        btmpv_cap = args.btmpv_cap_existing_mw
    else:
        btmpv_cap = [btmpv_capacity_projection(args.proj_year) * k for k in args.btmpv_dist]


    cap_columns = ['eheating_rate_node_', 'ev_rate_node_', 'onshore_cap_node_', 'offshore_cap_node_',
                   'solar_cap_node_', 'gt_new_cap_node_', 'batt_energy_cap_node_', 'batt_power_cap_node_',
                   'h2_energy_cap_node_', 'h2_power_cap_node_']

    # Populate the capacity results
    cap_results_df = pd.DataFrame()

    cap_results_df['obj_value'] = [m.objVal]

    for ix, col in enumerate(cap_columns):
        if col == 'gt_new_cap_node_':
            for jx in range(args.num_nodes):
                column_string = f'{col}{jx+1}'
                cap_results_df[column_string] = [m.getVarByName(column_string).X * args.reserve_req/cf_mult]
        else:
            for jx in range(args.num_nodes):
                column_string = f'{col}{jx+1}'
                cap_results_df[column_string] = [m.getVarByName(column_string).X/cf_mult]

    # Add BTM capacity results
    for ix in range(args.num_nodes):
        cap_results_df[f'btm_cap_node_{ix+1}'] = btmpv_cap[ix]

    ts_columns = ['ev_charging_node_', 'energy_balance_slack_node_', 'flex_hydro_node_', 'batt_charge_node_',
                  'batt_discharge_node_',
                  'batt_level_node_', 'h2_charge_node_', 'h2_discharge_node_', 'h2_level_node_',
                  'gt_new_util_node_', 'gt_new_diff_node_', 'gt_new_abs_node_', 'gt_existing_util_node_',
                  'gt_existing_diff_node_', 'gt_existing_abs_node_', 'biofuel_util_node_', 'elec_import_node_',
                  ]

    ## Populate timeseries Dataframe
    ts_results_df = pd.DataFrame()

    ## First put in datetime array:
    dt_parse = '%m/%d/%Y %H:%M'
    base_date = dt.strptime(args.base_datetime, dt_parse)
    dt_list =  [dt.strftime(base_date + datetime.timedelta(hours=x), dt_parse) for x in range(args.num_hours)]
    ts_results_df['date.time'] = dt_list


    # First put in demand timeseries
    for ix in range(args.num_nodes):
        ts_results_df[f'baseline_demand_node_{ix+1}'] = baseline_demand_hourly_mw[:, ix]

    # Add heating timeseries
    for ix in range(args.num_nodes):
        ts_results_df[f'heating_demand_node_{ix+1}'] = np.array(cap_results_df[f'eheating_rate_node_{ix+1}']) * \
                                                             full_heating_load_hourly_mw[:, ix]

    # Add onshore wind uncurtailed generation timeseries
    for ix in range(args.num_nodes):
        ts_results_df[f'onshore_uc_gen_node_{ix+1}'] = np.array(cap_results_df[f'onshore_cap_node_{ix+1}']) * \
                                                             onshore_pot_hourly[:, ix]

    # Add offshore wind uncurtailed generation timeseries
    for ix in range(args.num_nodes):
        ts_results_df[f'offshore_uc_gen_node_{ix+1}'] = np.array(cap_results_df[f'offshore_cap_node_{ix+1}']) * \
                                                      offshore_pot_hourly[:, ix]

    # Add solar uncurtailed generation timeseries
    for ix in range(args.num_nodes):
        ts_results_df[f'solar_uc_gen_node_{ix+1}'] = np.array(cap_results_df[f'solar_cap_node_{ix+1}']) * \
                                                       solar_pot_hourly[:, ix]

    # Add BTM uncurtailed generation timeseries
    for ix in range(args.num_nodes):
        ts_results_df[f'btmpv_uc_gen_node_{ix+1}'] = btmpv_cap[ix] * btmpv_pot_hourly[:, ix]

    # Add fixed hydropower generation timeseries
    for ix in range(args.num_nodes):
        ts_results_df[f'fixed_hydro_gen_node_{ix+1}'] = fixed_hydro_hourly_mw[:, ix]

    # Add nuclear generation timeseries
    for ix in range(args.num_nodes):
        ts_results_df[f'nuclear_gen_node_{ix+1}'] = int(args.nuclear_boolean) * args.nuc_avg_gen_mw[ix]

    # Add timeseries from ts_columns
    for ix, col in enumerate(ts_columns):
        ts_results_array = np.zeros((T, args.num_nodes))
        if ix == 1: # Collect the energy balance slack ts
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

    # Allocate the curtailment
    ts_results_df = allocate_curtailment(args, ts_results_df)

    # Find the LCT
    if model_config == 0 or model_config == 1:
        lct = m.getVarByName('lowc_target').X/cf_mult

    else: # model_config == 2 or model_config == 3:
        gas_gen = np.sum([ts_results_df[f'gt_new_util_node_{i+1}'] + ts_results_df[f'gt_existing_util_node_{i+1}']
                          for i in range(args.num_nodes)])/cf_mult
        biofuel_gen = np.sum([ts_results_df[f'biofuel_util_node_{i+1}'] for i in range(args.num_nodes)])/cf_mult
        total_heat_demand = np.sum([ts_results_df[f'heating_demand_node_{i+1}'] for i in range((args.num_nodes))]) \
                                   / cf_mult
        total_ev_demand = np.sum([ts_results_df[f'ev_charging_node_{i+1}'] for i in range((args.num_nodes))]) \
                                   / cf_mult
        total_imports = np.sum([ts_results_df[f'elec_import_node_{i+1}'] for i in range(args.num_nodes)])/cf_mult
        total_btm_gen = np.sum([ts_results_df[f'btmpv_uc_gen_node_{i+1}'] for i in range(args.num_nodes)])/cf_mult
        total_nuclear_gen = np.sum(int(args.nuclear_boolean) * int(args.rgt_boolean) * np.array(args.nuc_avg_gen_mw)
                                   * T)

        demand_for_lct = (np.sum(baseline_demand_hourly_mw) + total_heat_demand + total_ev_demand - total_imports -
                          total_btm_gen)
        lct = 1 - (gas_gen + biofuel_gen + total_nuclear_gen) / demand_for_lct

    ## Find the electrification ratio
    # Heating electrification rate
    full_therm_heating_load_nodal_avg = np.mean(full_ff_heating_load_hourly_mw, axis=0)
    therm_heating_load_nodal_avg = np.array([full_therm_heating_load_nodal_avg[i] *
                                             (1 - cap_results_df[f'eheating_rate_node_{i+1}'])
                                             for i in range(args.num_nodes)])

    heating_elecfx = 1 - np.sum(therm_heating_load_nodal_avg)/np.sum(full_therm_heating_load_nodal_avg)

    # Find the vehicle electrification rate
    ev_elecfx_nodal_ratios = np.array([args.icv_load_dist[i] * cap_results_df[f'ev_rate_node_{i+1}'] for i in
                                                range(args.num_nodes)])

    veh_elecfx = np.sum(ev_elecfx_nodal_ratios)


    # LCT GHGT elec results
    dghg_target = m.getVarByName('ghg_target').X / cf_mult

    # Round

    ## Add additional results to the dataframe
    cap_results_df['model_config'] = model_config
    cap_results_df['re_cost_scenario'] = args.re_cost_scenario
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

    cap_results_df.round(decimals=5).to_csv(cap_results_save_str)
    ts_results_df.round(decimals=5).to_csv(ts_results_save_str)

    return cap_results_df, ts_results_df


def full_results_processing(args):

    T = args.num_hours

    # Retrieve necessary model timeseries and dictionary of existing transmission parameters
    baseline_demand_hourly_mw, full_heating_load_hourly_mw, full_ff_heating_load_hourly_mw, \
    full_ff_dss50_hourly_mw, full_ev_load_hourly_mw, full_ev_avg_load_hourly_mw, onshore_pot_hourly, \
    offshore_pot_hourly, solar_pot_hourly, btmpv_pot_hourly, fixed_hydro_hourly_mw, \
    flex_hydro_daily_mwh = load_timeseries(args)

    tx_dict = return_tx_dict(args)

    # Collect all the raw capacity results, which are saved by scenario
    cap_results_dir = f'{args.results_dir}/{args.dir_time}/cap_results'
    cap_results_csvs = sorted(glob(f'{cap_results_dir}/*scenix*.csv'))
    scen_ixs = [int(i.split('_')[-1].replace('.csv', '')) for i in cap_results_csvs]


    # Add all the raw capacity results by scenario to a single dataframe. This dataframe contains all the capacity
    # results present in the model run folder
    cap_results_df = pd.DataFrame()
    for file in cap_results_csvs:
        cap_results_df = cap_results_df.append(pd.read_csv(file))
    cap_results_df = cap_results_df.reset_index()

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
    processed_df['scen_ix'] = scen_ixs
    processed_df['model_config'] = cap_results_df['model_config']
    processed_df['rgt/lct'] = cap_results_df['lct'].round(decimals=3)
    processed_df['ghg_reduction'] = cap_results_df['ghg_reduction']
    # Add heating and vehicle electrification rates for the entire region. These are based on the thermal loads of each
    processed_df['heating_elecfx_rate'] = cap_results_df['heating_elecfx_rate']
    processed_df['veh_elecfx_rate'] = cap_results_df['veh_elecfx_rate']
    # Add additional electic heating and vehicle loads
    processed_df['addl_heating_load_mw'] = avg_heating_demand
    processed_df['addl_ev_load_mw'] = avg_ev_demand
    processed_df['total_demand_mw'] = (np.sum(np.mean(baseline_demand_hourly_mw, axis=0)) +
                                       processed_df['addl_heating_load_mw'] + processed_df['addl_ev_load_mw'])
    # Continue parameterizing the processed dataframe with other model configuration parameters
    # processed_df['re_cost_scenario'] = cap_results_df['re_cost_scenario']
    processed_df['re_cost_scenario'] = cap_results_df['re_cost_scenario']
    processed_df['rgt_boolean'] = [int(args.rgt_boolean)] * len(cap_results_df)
    processed_df['nuc_boolean'] = [int(args.nuclear_boolean)] * len(cap_results_df)
    processed_df['h2_boolean'] = [int(args.h2_boolean)] * len(cap_results_df)
    processed_df['same_eheating_ev_rate_boolean'] = [int(args.same_eheating_ev_rate_boolean)] * len(cap_results_df)
    processed_df['btmpv_count_re'] = [int(args.btmpv_count_re)] * len(cap_results_df)
    processed_df['elec_constraint_ge'] = [int(args.elecfx_constraint_ge)] * len(cap_results_df)
    processed_df['gt_based_on_current'] = [int(args.gt_based_on_current)] * len(cap_results_df)
    processed_df['ev_set_profile_boolean'] = [int(args.ev_set_profile_boolean)] * len(cap_results_df)
    processed_df['ev_set_profile_boolean'] = [int(args.ev_set_profile_boolean)] * len(cap_results_df)
    processed_df['greenfield_boolean'] = [int(args.greenfield_boolean)] * len(cap_results_df)
    processed_df['copper_plate_boolean'] = [int(args.copper_plate_boolean)] * len(cap_results_df)
    processed_df['same_nodal_costs'] = [int(args.same_nodal_costs)] * len(cap_results_df)
    processed_df['no_hydro_gen'] = [int(args.no_hydro_gen)] * len(cap_results_df)

    # Add the electrification rates by node and by EV/heating
    for ix in range(args.num_nodes):
        processed_df[f'heating_elecfx_rate_node_{ix+1}'] = cap_results_df[f'eheating_rate_node_{ix+1}']
    for ix in range(args.num_nodes):
        processed_df[f'ev_elecfx_rate_node_{ix+1}'] = cap_results_df[f'ev_rate_node_{ix+1}']
    # Add returned capacities of new infrastructure
    processed_df['onshore_cap_mw'] = np.sum(onshore_cap, axis=1)
    processed_df['offshore_cap_mw'] = np.sum(offshore_cap, axis=1)
    processed_df['total_wind_gw'] = (processed_df['onshore_cap_mw'] + processed_df['offshore_cap_mw']) / 1000
    processed_df['solar_cap_mw'] = np.sum(solar_cap, axis=1)
    processed_df['solar_cap_gw'] = processed_df['solar_cap_mw']/1000
    processed_df['total_windsolar_cap_gw'] = processed_df['total_wind_gw'] + processed_df['solar_cap_gw']
    processed_df['new_gt_cap_mw'] = np.sum(gt_new_cap, axis=1)
    processed_df['existing_gt_cap_mw'] = np.sum(args.existing_gt_cap_mw)
    processed_df['total_gas_cap_gw'] = (processed_df['new_gt_cap_mw'] + processed_df['existing_gt_cap_mw'])/1000
    processed_df['battery_energy_cap_mwh'] = np.sum(battery_energy_cap, axis=1)
    processed_df['battery_power_cap_mw'] = np.sum(battery_power_cap, axis=1)
    processed_df['battery_energy_cap_gwh'] = processed_df['battery_energy_cap_mwh']/1000

    processed_df['h2_energy_cap_mwh'] = np.sum(h2_energy_cap, axis=1)
    processed_df['h2_power_cap_mw'] = np.sum(h2_power_cap, axis=1)
    processed_df['btm_power_cap_mw'] = np.sum(btm_cap, axis=1)
    # Add load met by btm solar
    processed_df['btm_gen_avg_mw'] = np.sum(btm_cap * np.mean(btmpv_pot_hourly, axis=0), axis=1)
    processed_df['nuc_gen_avg_mw'] = int(args.nuclear_boolean) * np.sum(args.nuc_avg_gen_mw)

    # Add new transmission limits
    # total_pos_cap_gw_mi = pd.Series()
    # total_neg_cap_gw_mi = pd.Series()

    processed_df['tx_pos_gw-mi'] = 0
    processed_df['tx_neg_gw-mi'] = 0

    for ix, tx_key in enumerate(tx_dict.keys()):
        out_node = int(tx_key.split("_")[-2])
        in_node = int(tx_key.split("_")[-1])
        new_tx_cap_string = f'new_tx_limit_{out_node}_{in_node}'
        tx_dict_string = f'existing_tx_limit_{out_node}_{in_node}'
        if out_node < in_node:
            processed_df['tx_pos_gw-mi'] += (cap_results_df[new_tx_cap_string] * tx_dict[tx_dict_string][2]) / 1000
        else:
            processed_df['tx_neg_gw-mi'] += (cap_results_df[new_tx_cap_string] * tx_dict[tx_dict_string][2]) / 1000

        processed_df[new_tx_cap_string+'_mw'] = cap_results_df[new_tx_cap_string]


    nodal_capacity_strings = ['onshore_cap_node_', 'offshore_cap_node_', 'solar_cap_node_', 'gt_new_cap_node_',
                              'batt_energy_cap_node_', 'h2_energy_cap_node_']
    # Add new wind, solar, and gt_capacity by node
    for ix, cap_str in enumerate(nodal_capacity_strings):
        for jx in range(args.num_nodes):
            if ix < 4:
                processed_df[f'{cap_str}{jx+1}_mw'] = cap_results_df[f'{cap_str}{jx+1}']
            else:
                processed_df[f'{cap_str}{jx+1}_mwh'] = cap_results_df[f'{cap_str}{jx + 1}']

    # Add upstate and downstate quantities
    processed_df['low-c_cap_upstate_mw'] = \
        np.sum(processed_df.loc[:, 'onshore_cap_node_1_mw':'onshore_cap_node_2_mw'], axis=1) + \
        np.sum(processed_df.loc[:, 'solar_cap_node_1_mw':'solar_cap_node_2_mw'], axis=1) + \
        (1-processed_df['no_hydro_gen']) * np.sum(args.hydro_cap_mw)
    processed_df['low-c_cap_downstate_mw'] = \
        np.sum(processed_df.loc[:, 'offshore_cap_node_3_mw':'offshore_cap_node_4_mw'], axis=1) + \
        np.sum(processed_df.loc[:, 'solar_cap_node_3_mw':'solar_cap_node_4_mw'], axis=1)

    processed_df['batt_cap_upstate_mwh'] = np.sum(processed_df.loc[:, 'batt_energy_cap_node_1_mwh':
                                                                  'batt_energy_cap_node_2_mwh'], axis=1)
    processed_df['batt_cap_downstate_mwh'] = np.sum(processed_df.loc[:, 'batt_energy_cap_node_3_mwh':
                                                                    'batt_energy_cap_node_4_mwh'], axis=1)
    processed_df['new_gt_cap_upstate_mw'] = np.sum(processed_df.loc[:, 'gt_new_cap_node_1_mw':'gt_new_cap_node_2_mw'],
                                                axis=1)
    processed_df['new_gt_cap_downstate_mw'] = np.sum(processed_df.loc[:, 'gt_new_cap_node_3_mw':'gt_new_cap_node_4_mw'],
                                                  axis=1)
    processed_df['existing_gt_cap_upstate_mw'] = args.existing_gt_cap_mw[0] + args.existing_gt_cap_mw[1]
    processed_df['existing_gt_cap_downstate_mw'] = args.existing_gt_cap_mw[2] + args.existing_gt_cap_mw[3]

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
    processed_df['total_annualized_cost'] = (new_cap_cost + generation_cost + supp_cost)/args.num_years


    ## calculate the renewable electricity ratio / low-carbon electricity ratio
    # average demand for renewable depend on
    avg_mwh_for_rg = (total_mwh_for_lcoe/T + int(args.btmpv_count_re) * btm_avg_mwh -
                      processed_df['elec_import_regional_avg_mw'])

    processed_df['model_rgt'] = (1 - (processed_df['gt_existing_util_regional_avg_mw'] +
                                    processed_df['gt_new_util_regional_avg_mw'] +
                                    processed_df['biofuel_util_regional_avg_mw'] +
                                    args.nuclear_boolean * np.sum(args.nuc_avg_gen_mw)) /
                                 avg_mwh_for_rg).round(decimals=3)
    processed_df['model_lct'] = (1 - (processed_df['gt_existing_util_regional_avg_mw'] +
                                    processed_df['gt_new_util_regional_avg_mw'] +
                                    processed_df['biofuel_util_regional_avg_mw']) / avg_mwh_for_rg).round(decimals=3)

    processed_df['rgt/lct'] = (int(args.rgt_boolean) * processed_df['model_rgt'] +
                               (1 - int(args.rgt_boolean)) * processed_df['model_lct'])

    # Write out the processed dataframe!
    processed_df = processed_df.round(decimals=3)
    processed_df = processed_df.set_index('scen_ix')
    processed_df = processed_df.sort_index()
    processed_df_filename = f'{args.results_dir}/{args.dir_time}/processed_results_{args.dir_time}.xlsx'
    processed_df.to_excel(processed_df_filename)


if __name__ == '__main__':
    args = get_args()
    full_results_processing(args)
