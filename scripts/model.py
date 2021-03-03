import numpy as np
from gurobipy import *
from scripts.utils import (load_timeseries, btmpv_capacity_projection, return_tx_dict,
                    return_costs_for_model, calculate_constant_costs)



def create_model(args, model_config, lct, ghgt, elec_ratio):
    '''
    Function that create the Gurobi model that will be optimized.

    :param args: args dictionary
    :param model_config: A 0, 1, 2, or 3 that corresponds to which constraints to apply to the model.
        0: Minimization of total costs. LCT + total electrification percent of heating + vehicle are
        specified.
        1: Minimization of total costs. LCT + GHG reduction percent are specified
        2: Minimization of total costs. Total electrification percent of heating + vehicle and GHG
        reduction percent are specified
        3: Minimization of LCOE. GHG reduction target is specified.
    :param lct: Low carbon target to be met (reneable generation target if args.rgt_boolean=True)
    :param ghgt: Greenhouse gas reduction target to be met.
    :param elec_ratio: Total percent electrification of heating and vehicle demand to be simulated.

    :return: m, the fully-parameterized Gurobi model.
    '''

    # Set up model parameters
    m = Model("capacity_optimization")
    T = args.num_hours
    trange = range(T)

    # Load in time-series data
    baseline_demand_hourly_mw, full_heating_load_hourly_mw, full_ff_heating_load_hourly_mw, \
    full_ff_dss50_hourly_mw, full_ev_load_hourly_mw, full_ev_avg_load_hourly_mw, onshore_pot_hourly, \
    offshore_pot_hourly, solar_pot_hourly, btmpv_pot_hourly, fixed_hydro_hourly_mw, \
    flex_hydro_daily_mwh = load_timeseries(args)

    # Load in formatted costs for variable assignment
    cost_dict = return_costs_for_model(args)

    # Set up LCT variable
    lowc_target = m.addVar(name = 'lowc_target')
    if model_config == 0 or model_config == 1:
        m.addConstr(lowc_target - lct >= 0)

    # Set up GHG variable
    ghg_target = m.addVar(name = 'ghg_target')
    if model_config == 1 or model_config == 2 or model_config == 3:
        m.addConstr(ghg_target - ghgt == 0)

    # Set up electrification % variable


    # Load dictionary with tx costs and current capacities. Create dictionary to store the transmission time series
    tx_dict = return_tx_dict(args)
    tx_ts_dict = {}

    # Initialize nuclear generation constraint base on nuclear boolean
    nuc_gen_mw = [int(args.nuclear_boolean) * float(args.nuc_avg_gen_mw[i]) for i in range(args.num_nodes)]

    # Determine BTM PV and waste emissions based on whether the projection year is 2019
    if args.proj_year == 2019:
        btmpv_cap_mw = args.btmpv_cap_existing_mw
        waste_emissions_mmt = args.waste_emissions_mmt
    else:
        btmpv_state_cap_mw = btmpv_capacity_projection(args.proj_year)
        btmpv_cap_mw = [btmpv_state_cap_mw * k for k in args.btmpv_dist]
        waste_emissions_mmt = 0

    # Define existing usable cap based on reserve requirement
    gt_existing_cap = [x / args.reserve_req for x in args.existing_gt_cap_mw]

    #####----------------------------------------------------------------------------------------------------------#####
    #####----------------------------------------------------------------------------------------------------------#####
    ##### !!!!!                     Initialize constraints for single node variables                       !!!!! #####
    #####----------------------------------------------------------------------------------------------------------#####

    ### 4 in-state node set up ###
    for i in range(0, args.num_nodes):

        # set total, regional electrification rates
        eheating_rate = m.addVar(name=f'eheating_rate_node_{i+1}')
        m.addConstr(eheating_rate <= 1)

        ev_rate = m.addVar(name=f'ev_rate_node_{i+1}')
        m.addConstr(ev_rate <= 1)

        m.update()
        if args.same_eheating_ev_rate_boolean:
            # Set the eheating and ev_rate equal
            m.addConstr(eheating_rate == ev_rate)
        
        # Find all interconnections for the current node
        tx_lines_set = sorted([j for j in tx_dict.keys() if str(i + 1) in j])
        # Find the complementary indices for the interconnections
        tx_export_nodes_set = np.unique([j.split('_')[-1] for j in tx_lines_set])
        # Only take the indices larger than i
        tx_export_nodes_set = [j for j in tx_export_nodes_set if int(j) > i + 1]

        # Create new tx capacity variables, two for each interface, an 'export' and an 'import'
        for export_node in tx_export_nodes_set:
            new_export_cap = m.addVar(obj=tx_dict[f'existing_tx_limit_{i+1}_{export_node}'][1],
                                      name=f'new_tx_limit_{i+1}_{export_node}')
            new_import_cap = m.addVar(obj=tx_dict[f'existing_tx_limit_{export_node}_{i+1}'][1],
                                      name=f'new_tx_limit_{export_node}_{i+1}')

        m.update()

        ## Initialize capacity variables
        onshore_cap     = m.addVar(obj=cost_dict['onshore_cost_per_mw'], name=f'onshore_cap_node_{i+1}')
        offshore_cap    = m.addVar(obj=cost_dict['offshore_cost_per_mw'], name=f'offshore_cap_node_{i+1}')
        solar_cap       = m.addVar(obj=cost_dict['solar_cost_per_mw'][i], name = f'solar_cap_node_{i+1}')
        gt_new_cap      = m.addVar(obj=cost_dict['gt_cost_per_mw'][i], name = f'gt_new_cap_node_{i+1}')
        battery_cap_mwh = m.addVar(obj=cost_dict['battery_cost_per_mwh'], name = f'batt_energy_cap_node_{i+1}')
        battery_cap_mw  = m.addVar(obj=cost_dict['battery_cost_per_mw'], name=f'batt_power_cap_node_{i+1}')

        h2_cap_mwh      = m.addVar(obj=cost_dict['h2_cost_per_mwh'][i], name = f'h2_energy_cap_node_{i+1}')
        h2_cap_mw       = m.addVar(obj=cost_dict['h2_cost_per_mw'][i], name = f'h2_power_cap_node_{i+1}')

        # Add capacity constraints
        m.addConstr(onshore_cap <= args.onshore_cap_limit_mw[i])
        m.addConstr(onshore_cap >= args.onshore_cap_existing_mw[i])

        m.addConstr(offshore_cap>=args.offshore_cap_existing_mw[i])
        m.addConstr(solar_cap <= args.solar_cap_limit_mw[i])
        m.addConstr(solar_cap >= args.solar_cap_existing_mw[i])

        m.addConstr(gt_new_cap >= int(args.gt_based_on_current) * args.current_scenario_addl_gt_cap[i])
        m.addConstr(battery_cap_mwh >= args.existing_battery_cap_mwh[i])
        m.addConstr(battery_cap_mw >= args.existing_battery_cap_mw[i])


        # Set the amount of new GT cap if no new capacity is allowed
        if not args.new_gt_boolean:
            m.addConstr(gt_new_cap == int(args.gt_based_on_current) * args.current_scenario_addl_gt_cap[i])

        # Fix renewable (wind, solar, and battery) capacities if required
        if args.fix_re_cap_boolean:
            m.addConstr(onshore_cap     == args.onshore_cap_existing_mw[i])
            m.addConstr(offshore_cap    == args.offshore_cap_existing_mw[i])
            m.addConstr(solar_cap       == args.solar_cap_existing_mw[i])
            m.addConstr(battery_cap_mwh == args.existing_battery_cap_mwh[i])
            m.addConstr(battery_cap_mw  == args.existing_battery_cap_mw[i])
        else:
            # Constrain battery power and energy to ratio limits in args
            m.addConstr(battery_cap_mw >= args.battery_p2e_ratio_range[0] * battery_cap_mwh)
            m.addConstr(battery_cap_mw <= args.battery_p2e_ratio_range[1] * battery_cap_mwh)


        ## Initialize time-series variables
        flex_hydro_mw   = m.addVars(trange, name = f'flex_hydro_node_{i+1}')
        biofuel_gen_mw  = m.addVars(trange, obj=args.biofuel_cost_mwh[i], name=f'biofuel_util_node_{i+1}')
        batt_charge     = m.addVars(trange, obj=args.nominal_storage_cost, name=f'batt_charge_node_{i+1}')
        batt_discharge  = m.addVars(trange, obj=args.nominal_storage_cost, name=f'batt_discharge_node_{i+1}')
        h2_charge       = m.addVars(trange, obj=args.nominal_storage_cost, name=f'h2_charge_node_{i+1}')
        h2_discharge    = m.addVars(trange, obj=args.nominal_storage_cost, name=f'h2_discharge_node_{i+1}')
        elec_import     = m.addVars(trange, obj=args.import_cost_mwh[i], name=f'elec_import_node_{i+1}')

        m.update()


        # Create transmission time series and total export/import capacity variables
        for export_node in tx_export_nodes_set:
            # Time series variable, both-ways
            tx_export_vars = m.addVars(trange, obj = args.nominal_trans_cost_mwh,
                                       name=f'tx_ts_{i+1}_to_{export_node}')
            tx_import_vars = m.addVars(trange, obj = args.nominal_trans_cost_mwh,
                                       name=f'tx_ts_{export_node}_to_{i+1}')

            # Export cap is = new cap + existing cap (from dictionary)
            tx_export_cap = m.getVarByName(f'new_tx_limit_{i+1}_{export_node}') + \
                             tx_dict[f'existing_tx_limit_{i+1}_{export_node}'][0]

            # Import cap is = new cap + existing cap (from dictionary)
            tx_import_cap = m.getVarByName(f'new_tx_limit_{export_node}_{i+1}') + \
                            tx_dict[f'existing_tx_limit_{export_node}_{i+1}'][0]

            m.update()

            # Constrain individual Tx flow variables to the export import capacity
            for j in trange:
                m.addConstr(tx_export_vars[j] - tx_export_cap <= 0)
                m.addConstr(tx_import_vars[j] - tx_import_cap <= 0)

            # Store these tx flow variables in the time series dictionary for energy balance equation
            tx_ts_dict[f'tx_ts_{i+1}_to_{export_node}'] = tx_export_vars
            # both-ways
            tx_ts_dict[f'tx_ts_{export_node}_to_{i+1}'] = tx_import_vars

        m.update()

        # Initialize battery level and EV charging variables
        batt_level   = m.addVars(trange, name = f'batt_level_node_{i+1}')
        h2_level     = m.addVars(trange, name = f'h2_level_node_{i+1}')
        ev_charging  = m.addVars(trange, name = f'ev_charging_node_{i+1}')

        # Initialize gt variables
        gt_new_util = m.addVars(trange, obj=cost_dict['new_gt_cost_mwh'][i], name=f'gt_new_util_node_{i+1}')
        gt_new_diff = m.addVars(trange, lb=-GRB.INFINITY, name = f'gt_new_diff_node_{i+1}')
        gt_new_abs  = m.addVars(trange, obj=args.new_gt_startup_cost_mw/2, name = f'gt_new_abs_node_{i+1}')

        gt_existing_util = m.addVars(trange, obj=cost_dict['existing_gt_cost_mwh'][i],
                                     name=f'gt_existing_util_node_{i+1}')
        gt_existing_diff = m.addVars(trange, lb=-GRB.INFINITY, name=f'gt_existing_diff_node_{i+1}')
        gt_existing_abs  = m.addVars(trange, obj=args.existing_gt_startup_cost_mw/2,
                                     name=f'gt_existing_abs_node_{i+1}')


        # Initialize H2 constraints based on model run specifics
        if not args.h2_boolean:
            m.addConstr(h2_cap_mwh == 0)
            m.addConstr(h2_cap_mw == 0)
        else:
            m.addConstr(h2_cap_mw >= args.h2_p2e_ratio_range[0] * h2_cap_mwh)
            m.addConstr(h2_cap_mw <= args.h2_p2e_ratio_range[1] * h2_cap_mwh)


        # Find all export/import time series for energy balance -- these variables will find the same time series
        # but in different nodes
        tx_export_keys     = [k for k in tx_ts_dict.keys() if 'ts_{}'.format(i + 1) in k]
        tx_import_keys     = [k for k in tx_ts_dict.keys() if 'to_{}'.format(i + 1) in k]

        m.update()


        # Add time-series Constraints
        for j in trange:

            # Maximum of various existing electricity sources
            m.addConstr(flex_hydro_mw[j] <= args.flex_hydro_cap_mw[i])
            m.addConstr(biofuel_gen_mw[j] <= args.biofuel_cap_mw[i])
            m.addConstr(elec_import[j] <= args.import_limit_mw[i])
            m.addConstr(gt_existing_util[j] <= gt_existing_cap[i])

            # Sum all the transmission export time series for node i at time step j
            if len(tx_export_keys) > 0:
                total_exports = quicksum(tx_ts_dict[tx_export_keys[k]][j] for k in range(len(tx_export_keys)))
            else:
                total_exports = 0

            # Sum all the transmission import time series for node i at time step j
            if len(tx_import_keys) > 0:
                total_imports = quicksum(tx_ts_dict[tx_import_keys[k]][j] for k in range(len(tx_import_keys)))
            else:
                total_imports = 0


            # Contrain Gas turbine capacity
            m.addConstr(gt_new_util[j] <= gt_new_cap)

            # Load constraint: No battery/H2 operation in time t=0
            # First transmission loss constraint
            m.addConstr((offshore_cap * offshore_pot_hourly[j, i]) + (onshore_cap * onshore_pot_hourly[j, i]) +
                        (solar_cap * solar_pot_hourly[j, i]) + flex_hydro_mw[j] + biofuel_gen_mw[j] -
                        batt_charge[j] + batt_discharge[j] - h2_charge[j] + h2_discharge[j] -
                        ev_charging[j] - total_exports + (1 - args.trans_loss) * total_imports +
                        elec_import[j] + gt_new_util[j] + gt_existing_util[j]  >=
                        baseline_demand_hourly_mw[j, i] + full_heating_load_hourly_mw[j, i] * eheating_rate -
                        fixed_hydro_hourly_mw[j, i] - nuc_gen_mw[i] - btmpv_cap_mw[i] * btmpv_pot_hourly[j, i],
                        name= f'energy_balance_slack_node_{i+1}[{j}]')

            # Battery operation constraints
            m.addConstr(batt_charge[j] - battery_cap_mw <= 0)
            m.addConstr(batt_discharge[j] - battery_cap_mw <= 0)
            m.addConstr(batt_level[j] - battery_cap_mwh <= 0)

            # H2 operation constraints
            m.addConstr(h2_charge[j] - h2_cap_mw <= 0)
            m.addConstr(h2_discharge[j] - h2_cap_mw <= 0)
            m.addConstr(h2_level[j] - h2_cap_mwh <= 0)

            if j == 0:
                # # Battery/H2 energy conservation constraints for first time step (based on ending battery/H2 level)
                m.addConstr(batt_discharge[j] / args.battery_eff - args.battery_eff * batt_charge[j] ==
                            ((1 - args.battery_self_discharge) * batt_level[args.num_hours-1] - batt_level[j]))
                m.addConstr(h2_discharge[j] / args.h2_eff - args.h2_eff * h2_charge[j] ==
                            ((1 - args.h2_self_discharge) * h2_level[args.num_hours-1] - h2_level[j]))
                # print('')

            else:
                # Battery/H2 energy conservation constraints
                m.addConstr(batt_discharge[j] / args.battery_eff - args.battery_eff * batt_charge[j] ==
                            ((1 - args.battery_self_discharge) * batt_level[j - 1] - batt_level[j]))
                m.addConstr(h2_discharge[j] / args.h2_eff - args.h2_eff * h2_charge[j] ==
                            ((1 - args.h2_self_discharge) * h2_level[j - 1] - h2_level[j]))

                # Net load ramping constraints
                m.addConstr(gt_new_diff[j] - (gt_new_util[j] - gt_new_util[j - 1]) == 0)
                m.addConstr(gt_new_abs[j] >= gt_new_diff[j])
                m.addConstr(gt_new_abs[j] >= -gt_new_diff[j])

                m.addConstr(gt_existing_diff[j] - (gt_existing_util[j] - gt_existing_util[j - 1]) == 0)
                m.addConstr(gt_existing_abs[j] >= gt_existing_diff[j])
                m.addConstr(gt_existing_abs[j] >= -gt_existing_diff[j])

            # Add constraints for new HQ imports into NYC -- This is to ensure constant flow of power
            if i == 2:
                m.addConstr(elec_import[j] - args.hqch_capacity_factor * args.import_limit_mw[i] == 0)

        m.update()

        # Initialize flexible hydro and biofuel dispatch
        for j in range(0, int(T / 24)):
            jrange_daily = range(j * 24, (j + 1) * 24)
            m.addConstr(quicksum(flex_hydro_mw[k] for k in jrange_daily) == flex_hydro_daily_mwh[j, i])
            # biofuel total generation would be equal or smaller than the current generation
            m.addConstr(quicksum(biofuel_gen_mw[k] for k in jrange_daily) <= args.biofuel_daily_gen_mwh[i])

        m.update()



        # Initialize EV charging constraints
        if args.ev_set_profile_boolean:
            for j in trange:
                m.addConstr(ev_charging[j] == ev_rate * full_ev_load_hourly_mw[j,i])
        
        else:
            # Constrain EV charging rate
            m.addConstr(ev_charging[j] - ev_rate * full_ev_avg_load_hourly_mw[i] / float(args.ev_charging_p2e_ratio) <= 0)

            for j in range(0, int(T / 24)):
                jrange_ev = range(j * 24, j * 24 + args.ev_charging_hours)
    
                if args.ev_charging_method == 'flexible':
                    m.addConstr(quicksum(ev_charging[args.ev_hours_start + k] for k in jrange_ev) ==
                                ev_rate * full_ev_avg_load_hourly_mw[i] * 24)
                elif args.ev_charging_method == 'fixed':
                    for k in jrange_ev:
                        m.addConstr(ev_charging[args.ev_hours_start + k]  ==
                                    ev_rate * full_ev_avg_load_hourly_mw[i] * 24 / args.ev_charging_hours)
                else:
                    raise ValueError('Invalid EV charging method')

        m.update()
        


    #####----------------------------------------------------------------------------------------------------------#####
    ##### !!!!!                     Initialize constraints for multi-node variables                        !!!!! #####
    #####----------------------------------------------------------------------------------------------------------#####

    # Dicts for setting gas utilization equal to a percent of the load
    model_data_gt_new_util = {}
    model_data_gt_existing_util = {}
    # Dicts for setting offshore wind capacity equal to NREL limit
    model_data_offshore_cap = {}
    ## Dict for biofuel results
    model_data_biofuel = {}
    ## Dicts for retrieving Hydro Quebec import
    model_data_imports = {}
    ## electrification rate constraints
    model_data_eheating_ratio = {}
    model_data_ev_ratio = {}

    # Collect new and existing GT, biofuel, and import util variables, all time dependent
    # Collect offshore capacity, and eheating + ev ratio, one value per node
    for i in range(args.num_nodes):
        for j in trange:
            model_data_gt_new_util[i, j] = m.getVarByName(f'gt_new_util_node_{i+1}[{j}]')
            model_data_gt_existing_util[i, j] = m.getVarByName(f'gt_existing_util_node_{i+1}[{j}]')
            model_data_biofuel[i, j] = m.getVarByName(f'biofuel_util_node_{i+1}[{j}]')
            model_data_imports[i, j] = m.getVarByName(f'elec_import_node_{i+1}[{j}]')


        model_data_offshore_cap[i] = m.getVarByName(f'offshore_cap_node_{i+1}')
        model_data_eheating_ratio[i] = m.getVarByName(f'eheating_rate_node_{i+1}')
        model_data_ev_ratio[i] = m.getVarByName(f'ev_rate_node_{i+1}')

    # Limit offshore wind to the total allowable capacity
    m.addConstr(quicksum(model_data_offshore_cap[i] for i in range(args.num_nodes)) <=
                args.offshore_cap_total_limit_mw)


    # add the const make all electrification rate be equal
    if args.same_nodal_elecfx_rates_boolean:
        for i in range(args.num_nodes - 1):
            m.addConstr(model_data_eheating_ratio[i] == model_data_eheating_ratio[i + 1])
            m.addConstr(model_data_ev_ratio[i] == model_data_ev_ratio[i + 1])

    m.update()
    ## Collect ff heating and transport demands for the electrification constraint

    # Get the full heating ff load
    full_ff_heating_load_nodal_avg = np.mean(full_ff_heating_load_hourly_mw, axis=0)
    full_dss50_ff_heating_load_nodal_avg = np.mean(full_ff_dss50_hourly_mw, axis=0)
    ff_heating_load_avg = quicksum(full_ff_heating_load_nodal_avg[i] *
                                      (1-model_data_eheating_ratio[i]) for i in range(args.num_nodes))

    # Find the average eheating and EV load
    eheating_mean_load_region = np.mean(full_heating_load_hourly_mw, axis=0)
    eheating_load_avg = quicksum(eheating_mean_load_region[i] * model_data_eheating_ratio[i]
                              for i in range(args.num_nodes))
    ev_load_avg = quicksum(full_ev_avg_load_hourly_mw[i] * model_data_ev_ratio[i]
                              for i in range(args.num_nodes))

    # Weighted FF EV electrification ratio based on the EV load distribution
    weighted_ev_elecfx_ratio = quicksum(args.icv_load_dist[i] * model_data_ev_ratio[i] for i in range(
        args.num_nodes))

    # Constrain the electrification fractions to be either == to >= what's specified by elec_ratio
    # The electrification ratio is applied to the amounts of electrified load, eheating and vehicle
    # We constrain the ff heating load with (1-elec_ratio) a
    if model_config == 0 or model_config == 2:
        if args.elecfx_constraint_ge:
            m.addConstr(ff_heating_load_avg - np.sum(full_ff_heating_load_nodal_avg) * (1-elec_ratio) >= 0)
            m.addConstr(weighted_ev_elecfx_ratio - elec_ratio >= 0)

        else:
            m.addConstr(ff_heating_load_avg - np.sum(full_ff_heating_load_nodal_avg) * (1-elec_ratio) == 0)
            m.addConstr(weighted_ev_elecfx_ratio - elec_ratio == 0)

    # Collect all the the all btmpv generation
    btmpv_cf = np.mean(btmpv_pot_hourly, axis=0)
    btmpv_avg_gen = np.sum([btmpv_cap_mw[k] * btmpv_cf[k] for k in range(args.num_nodes)])

    # Collect the total demand for LCT/RGT constraint
    full_demand_sum_mwh  = np.sum(baseline_demand_hourly_mw[0:T]) + eheating_load_avg * T + \
                           ev_load_avg * T - (1 - args.btmpv_count_re) * btmpv_avg_gen * T

    # Collect the GT utilization and import data for the LCT/RGT constraint
    full_gt_new_sum_mwh = quicksum(model_data_gt_new_util[i, j] for j in trange for i in range(args.num_nodes))
    full_gt_existing_sum_mwh = quicksum(model_data_gt_existing_util[i, j] for j in trange for i in range(args.num_nodes))
    full_hq_imports_sum_mwh = quicksum(model_data_imports[i, j] for j in trange for i in range(args.num_nodes))

    # Collect the nuclear + biofuel generation for LCT/RGT constraint
    full_nuclear_sum_mwh = np.sum(nuc_gen_mw) * T
    full_biofuel_sum_mwh = quicksum(model_data_biofuel[i, j] for j in trange for i in range(args.num_nodes))

    m.update()

    # Low-carbon or renewable constraint
    if model_config == 0 or model_config == 1:
        frac_netload = 1 - lowc_target
        # Scale to avoid numerical issues on the quadratic constraint
        numer_scale = 1e6
        if args.rgt_boolean:  # Apply RGT constaint
            m.addConstr((full_gt_new_sum_mwh + full_gt_existing_sum_mwh +
                         full_nuclear_sum_mwh + full_biofuel_sum_mwh)/numer_scale -
                        (full_demand_sum_mwh - full_hq_imports_sum_mwh) * frac_netload/numer_scale <= 0)
        else: # Apply LCT constraint
            m.addConstr((full_gt_new_sum_mwh + full_gt_existing_sum_mwh + full_biofuel_sum_mwh)/numer_scale -
                        (full_demand_sum_mwh - full_hq_imports_sum_mwh) * frac_netload/numer_scale <= 0)
    m.update()

    ### Emissions accounting and constraint application ###

    # Find electricity sector emissions -- 1e6 converts from t to MMt
    elec_emissions = (full_gt_new_sum_mwh / args.new_gt_efficiency + full_gt_existing_sum_mwh /
                      args.existing_gt_efficiency) * args.ng_e_factor_t_mwh / (1e6 * args.num_years)

    # Heating emissions
    heating_emissions = quicksum((args.flex_space_heating_emissions_mmt[i] + args.flex_const_heating_emissions_mmt[i])
                                 * (1 - model_data_eheating_ratio[i]) for i in range(args.num_nodes))

    # Accounting for heating emissions from DSS
    heating_emissions_dss = quicksum(int(args.dss_synthetic_ts) *
                         args.flex_space_heating_emissions_mmt[i] * model_data_eheating_ratio[i] *
                         full_dss50_ff_heating_load_nodal_avg[i] / full_ff_heating_load_nodal_avg[i]
                         for i in range(args.num_nodes))

    # Find transport emissions
    trans_emissions = args.flex_trans_emissions_mmt * quicksum((1 - model_data_ev_ratio[i]) * args.icv_load_dist[i]
                                                         for i in range(args.num_nodes))

    # Sum total emissions and constrain to the ghg_target
    m.addConstr((elec_emissions +
                 heating_emissions +
                 heating_emissions_dss +
                 trans_emissions +
                 args.fixed_trans_emissions_mmt +
                 args.fixed_ind_emissions_mmt +
                 waste_emissions_mmt) == (1 - ghg_target) * args.baseline_emissions_mmt,
                name='ghg_emissions_constraint')

    m.update()
    

    if model_config == 3:
        ## Model Modifications for LCOE Minimization
        ## The approach here modifies the base model to a linear-fractional program and employs the Charnes-Cooper
        ## transformation to linearize the problem

        # Compute constant costs independent of decision variables
        const_costs_total = calculate_constant_costs(args)

        ## Adjust scale to avoid very small transform decision variable and associated numerical issues
        all_vars_init = m.getVars()
        obj_coeffs_init = m.getAttr('Obj')
        const_costs_total = const_costs_total / 1e9
        for i in range(len(all_vars_init)):
            var_it = all_vars_init[i]
            var_it.setAttr('Obj', obj_coeffs_init[i] / 1e9)

        ## Create C-C transform variable and add to objective function with const_costs_total as coefficient
        cc_transform = m.addVar(lb=0, obj=const_costs_total, name='cc_transform')
        m.update()

        ## Under transform, the negative of the RHS vector becomes the constraint coefficients for cc_transform and the RHS becomes zero
        cc_b = m.getAttr('RHS')
        cc_constr_list = m.getConstrs()

        for i in range(len(cc_constr_list)):
            m.chgCoeff(cc_constr_list[i], m.getVarByName('cc_transform'), -cc_b[i])
        m.setAttr('RHS', cc_constr_list, 0)

        # Add constraint reflecting transformed denominator of linear-fractional objective (i.e. total electricity demand)
        m.addConstr((eheating_load_avg + ev_load_avg) * T + np.sum(baseline_demand_hourly_mw[0:T]) *
                    cc_transform == 1)
        m.update()

    return m
