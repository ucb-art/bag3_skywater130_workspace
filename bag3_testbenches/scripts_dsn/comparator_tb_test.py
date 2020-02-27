from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

from bag.core import BagProject
from bag.simulation.core import DesignManager
from bag.io import read_yaml

from bag3_testbenches.measurement.comparator.helper import get_db, get_vb_crossing_g_p, \
    get_vb_crossing_g_n, get_vb_crossing_n


def _calc_Vb(specs: Dict[str, Any], vdd: float) -> float:
    nch_db = get_db('nch', specs)
    pch_db = get_db('pch', specs)
    doc_seg = specs['schematic_params']['comp_out_params']['comp_params']['doc_params']['seg_dict']
    Vb, VDD_g = get_vb_crossing_g_p(nch_db, pch_db, doc_seg['gm2n'], doc_seg['gm2p'],
                                    doc_seg['gm1'], doc_seg['pow_cross'], doc_seg['pow_in'], vdd)
    print(f'Vb = {Vb} V')

    tbs = specs['measurements'][0]['testbenches']
    for key, val in tbs.items():
        val['sim_params']['Vb'] = Vb
    return Vb


def _calc_Vb2(specs: Dict[str, Any], vdd: float, ratio: Optional[float] = None) -> float:
    if ratio is None:
        nch_db = get_db('nch', specs)
        pch_db = get_db('pch', specs)
        doc_params = specs['schematic_params']['comp_out_params']['comp_params']['doc_params']
        doc_seg = doc_params['seg_dict']
        Vb, VSS_g = get_vb_crossing_g_n(nch_db, pch_db, doc_seg['gm2n'], doc_seg['gm2p'],
                                        doc_seg['gm1'], doc_seg['pow'], vdd)
        print(f'VSS_g = {VSS_g} V')
    else:
        Vb = vdd * ratio

    print(f'Vb = {Vb} V')

    tbs = specs['measurements'][0]['testbenches']
    for key, val in tbs.items():
        val['sim_params']['Vb'] = Vb
    return Vb


def _calc_Vb3(specs: Dict[str, Any], vdd: float, ratio: Optional[float] = None) -> float:
    if ratio is None:
        nch_db = get_db('nch', specs)
        pch_db = get_db('pch', specs)
        doc_params = specs['schematic_params']['comp_out_params']['comp_params']['doc_params']
        doc_seg = doc_params['seg_dict']
        Vb = get_vb_crossing_n(nch_db, pch_db, doc_seg['gm2n'], doc_seg['gm2p'], doc_seg['gm1'],
                               vdd)
    else:
        Vb = vdd * ratio

    print(f'Vb = {Vb} V')

    tbs = specs['measurements'][0]['testbenches']
    for key, val in tbs.items():
        val['sim_params']['Vb'] = Vb
    return Vb


if __name__ == '__main__':

    # --- Topology 1: pmos input, Vb --- #
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'P_Vb',
    #                    'char_diff.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'P_Vb',
    #                    'supply_diff.yaml')

    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'P_Vb',
    #                    'char_single.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'P_Vb',
    #                    'supply_single.yaml')

    # --- Topology 2: pmos input, self biased --- #
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'P_self',
    #                    'char_diff.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'P_self',
    #                    'supply_diff.yaml')

    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'P_self',
    #                    'char_diff_assym.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'P_self',
    #                    'supply_diff_assym.yaml')

    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'P_self',
    #                    'char_single.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'P_self',
    #                    'supply_single.yaml')

    # --- Topology 3: nmos input, self biased, assymmetric --- #
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_assym',
    #                    'char_single.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_assym',
    #                    'supply_single.yaml')

    # config_file = Path('specs_design', 'bag3_testbenches', 'comparatorN_char.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'comparatorN_tb.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'comparatorN_pss.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'comparatorN_pac.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'comparatorN_pnoise.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'overdriveN_tb.yaml')

    # --- Topology 4: nmos input, self biased, symmetric --- #
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_sym',
    #                    'char_diff.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_sym',
    #                    'supply_diff.yaml')

    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_sym',
    #                    'char_diff_assym.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_sym',
    #                    'supply_diff_assym.yaml')

    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_sym',
    #                    'char_single.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_sym',
    #                    'supply_single.yaml')

    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_sym',
    #                    'char_single_sym.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_sym',
    #                    'supply_single_sym.yaml')

    # --- Topology 5: pmos input, split gain and regeneration stages --- #
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'P_split',
    #                    'char_diff.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'P_split',
    #                    'supply_diff.yaml')

    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'P_split',
    #                    'char_single.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'P_split',
    #                    'supply_single.yaml')

    # --- Topology 6: nmos input, split gain and regeneration stages --- #
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_split',
    #                    'char_diff.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_split',
    #                    'supply_diff.yaml')

    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_split',
    #                    'char_single.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_split',
    #                    'supply_single.yaml')

    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_split',
    #                    'char_single_sym.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_split',
    #                    'supply_single_sym.yaml')

    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_split',
    #                    'char_single_n_sw.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_split',
    #                    'supply_single.yaml')

    # --- Topology 7: inverter input, split gain and regeneration stages --- #
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'inv_split',
    #                    'char_single.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'inv_split',
    #                    'supply_single.yaml')

    # --- Topology 0: auto-zeroing inverter amplifier --- #
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'inv_az',
    #                    'char.yaml')

    # --- Topology 8: nmos input, Vb --- #
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_Vb',
    #                    'char_diff.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_Vb',
    #                    'supply_diff.yaml')

    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_Vb',
    #                    'char_single.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_Vb',
    #                    'supply_single.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_Vb',
    #                    'comp_tran_noise.yaml')

    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_Vb',
    #                    'char_single2.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_Vb',
    #                    'supply_single2.yaml')

    config_file = Path('data', 'bag3_testbenches', 'specs_dsn', 'unit_tests_comparator', 'N_Vb',
                       'char_single3.yaml')
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_Vb',
    #                    'supply_single3.yaml')

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()
    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    specs = read_yaml(config_file)

    calc_Vb: bool = specs.get('calc_Vb', False)
    gen_sch = False

    vdd_list = specs.get('vdd_list', None)

    if vdd_list is not None:
        thres_list, vb_list, hyst_list, fall_list, rise_list = [], [], [], [], []
        tbs = specs['measurements'][0]['testbenches']
        vdd_nom = tbs['overdrive']['sim_params']['vdd']
        # Vb_nom = _calc_Vb2(specs, vdd_nom)
        Vb_nom = _calc_Vb3(specs, vdd_nom)
        for vdd in vdd_list:
            for key, val in tbs.items():
                val['sim_params']['vdd'] = vdd
                val['sim_params']['vclk2'] = vdd
            if calc_Vb:
                # vb_list.append(_calc_Vb(specs, vdd))
                # vb_list.append(_calc_Vb2(specs, vdd, ratio=Vb_nom/vdd_nom))
                vb_list.append(_calc_Vb3(specs, vdd, ratio=Vb_nom/vdd_nom))
            sim = DesignManager(bprj, spec_dict=specs)
            sim.characterize_designs(generate=True, gen_sch=gen_sch, measure=True,
                                     load_from_file=False, mismatch=True)
            for dsn_name in sim.info.dsn_name_iter():
                result = sim.get_result(dsn_name)
                thres_list.append(result['comparator_tran']['mean_thres'])
                hyst_list.append(result['comparator_tran']['hysteresis'])
                fall_list.append(result['comparator_tran']['rst_fall_time'])
                rise_list.append(result['comparator_tran']['rst_rise_time'])
        print('Threshold variation:', thres_list)
        print('Hysteresis variation:', hyst_list)
        print('Reset fall time variation:', fall_list)
        print('Reset rise time variation:', rise_list)
        plt.subplot(211)
        plt.plot(vdd_list, thres_list, label='switching threshold')
        if calc_Vb:
            plt.plot(vdd_list, vb_list, label='Vb')
        plt.legend()
        plt.xlabel('VDD (in V)')
        plt.ylabel('Voltage (in V)')

        plt.subplot(212)
        plt.plot(vdd_list, fall_list, label='Reset fall time')
        plt.plot(vdd_list, rise_list, label='Reset rise time')
        plt.legend()
        plt.xlabel('VDD (in V)')
        plt.ylabel('Time (in sec)')
        plt.show()
    else:
        if calc_Vb:
            tbs = specs['measurements'][0]['testbenches']
            vdd = next(iter(tbs.values()))['sim_params']['vdd']
            # _calc_Vb(specs, vdd)
            # _calc_Vb2(specs, vdd)
            _calc_Vb3(specs, vdd)

        sim = DesignManager(bprj, spec_dict=specs)
        sim.characterize_designs(generate=True, gen_sch=gen_sch, measure=True,
                                 load_from_file=False, mismatch=True)
