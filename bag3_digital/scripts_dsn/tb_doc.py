# -*- coding: utf-8 -*-
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from bag.core import BagProject
from bag.io import read_yaml
from bag.simulation.core import DesignManager

from bag3_digital.design.sampler.doc_dsn import get_db, get_vb_crossing


def run_main(bprj: BagProject) -> None:
    # fname = 'tb_doc.yaml'
    fname = 'tb4_doc.yaml'
    specs = read_yaml(Path('specs_design', 'bag3_digital', fname))

    # Calculate bias point for certain schematic params
    nch_db = get_db(bprj, 'nch', specs)
    pch_db = get_db(bprj, 'pch', specs)

    sch_params = specs['schematic_params']
    doc_seg_dict = sch_params['doc_wrap_params']['doc_params']['seg_dict']
    vdd = specs['measurements'][0]['testbenches']['overdrive']['vdd']
    Vb = get_vb_crossing(nch_db, pch_db, doc_seg_dict['gm2n'], doc_seg_dict['gm2p'],
                         doc_seg_dict['gm1'], vdd)
    specs['measurements'][0]['testbenches']['overdrive']['Vb'] = Vb

    # generate and run simulation
    cc_arr = np.linspace(3.0e-15, 20.0e-15, num=18)
    tau_rst_list, tau_regen_list, vop_list, vom_list = [], [], [], []
    for cc in cc_arr:
        specs['measurements'][0]['testbenches']['overdrive']['cc'] = cc
        sim = DesignManager(bprj, spec_dict=specs)
        sim.characterize_designs(generate=True, measure=True, load_from_file=False)
        for dsn in sim.get_dsn_name_iter():
            result = sim.get_result(dsn)['overdrive_recovery']
            tau_rst_list.append(result['tau_rst'])
            tau_regen_list.append(result['tau_regen'])
            vop_list.append(result['vop'])
            vom_list.append(result['vom'])

    tau_regen_arr, tau_rst_arr = np.asarray(tau_regen_list), np.asarray(tau_rst_list)
    vop_arr, vom_arr = np.asarray(vop_list), np.asarray(vom_list)
    k_list = [0.25, 0.5, 0.75]
    plt.subplot(2, 1, 1)
    for k in k_list:
        tau_arr = k * tau_rst_arr + (1 - k) * tau_regen_arr
        plt.plot(cc_arr * 1.0e15, tau_arr, label=f'k={k}')
    plt.legend()
    plt.xlabel('Cc (in fF)')

    plt.subplot(2, 1, 2)
    plt.plot(cc_arr * 1.0e15, vop_arr)
    plt.plot(cc_arr * 1.0e15, vom_arr)
    plt.xlabel('Cc (in fF)')
    plt.ylabel('V')
    plt.legend(['VoP', 'VoM'])
    plt.show()


if __name__ == "__main__":
    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()
    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    run_main(bprj)
