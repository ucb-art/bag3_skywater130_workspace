from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any
import numpy as np

from bag.core import BagProject
from bag.simulation.core import DesignManager
from bag.io import read_yaml

from bag3_testbenches.measurement.comparator.helper import get_db, get_vb_crossing_g


def _calc_Vb(specs: Dict[str, Any], vdd: float) -> float:
    nch_db = get_db('nch', specs)
    pch_db = get_db('pch', specs)
    doc_seg = specs['schematic_params']['comp_out_params']['comp_params']['doc_params']['seg_dict']
    Vb, VDD_g = get_vb_crossing_g(nch_db, pch_db, doc_seg['gm2n'], doc_seg['gm2p'],
                                  doc_seg['gm1'], doc_seg['pow_cross'], doc_seg['pow_in'], vdd)
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
    #                    'ac_diff.yaml')

    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_split',
    #                    'ac_single.yaml')

    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_split',
    #                    'char_single_sym.yaml')

    # --- Topology 7: inverter input, split gain and regeneration stages --- #
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'inv_split',
    #                    'ac_diff.yaml')

    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'inv_split',
    #                    'ac_single.yaml')

    # --- Topology 0: auto-zeroing inverter amplifier --- #
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'inv_az',
    #                    'ac.yaml')

    # --- Topology 8: nmos input, Vb --- #
    # config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_Vb',
    #                    'ac_diff.yaml')

    config_file = Path('specs_design', 'bag3_testbenches', 'unit_tests_comparator', 'N_Vb',
                       'ac_single.yaml')

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()
    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    specs = read_yaml(config_file)

    calc_Vb: bool = specs.get('calc_Vb', False)
    gen_sch = True

    sim = DesignManager(bprj, spec_dict=specs)
    sim.characterize_designs(generate=True, gen_sch=gen_sch, measure=True, load_from_file=False)
