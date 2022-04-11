# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, Any

import numpy as np

from bag.core import BagProject
from bag.io import read_yaml
from bag.simulation.core import DesignManager

from bag3_testbenches.measurement.mos.query import MOSDBDiscrete
from bag3_testbenches.measurement.monte.sim import MismatchTBM


def get_db(mos_type: str, dsn_specs: Dict[str, Any]) -> MOSDBDiscrete:
    mos_specs = dsn_specs[mos_type]

    spec_file = mos_specs['spec_file']
    interp_method = mos_specs.get('interp_method', 'spline')
    sim_env = mos_specs.get('sim_env', 'tt_25')

    db = MOSDBDiscrete([spec_file], interp_method=interp_method)
    db.env_list = [sim_env]
    db.set_dsn_params(intent=mos_specs['intent'])

    return db


def get_ibias_list(specs: Dict[str, Any]) -> None:
    monte = specs['measurements'][0]['testbenches']['monte']
    vstar_list = monte['vstar_list']

    sim_params = monte['sim_params']
    vbody = sim_params['vbody']
    vcm = sim_params['vcm']

    sch_params = specs['schematic_params']
    nf = sch_params['nf']
    mos_type = sch_params['mos_type']
    sign = -1 if mos_type == 'pch' else 1

    mos_db = get_db(mos_type, specs)
    ibias_list = []
    for vstar in vstar_list:
        mos_op = mos_db.query(vds=vcm-vbody, vbs=0.0, vstar=vstar)
        ibias_list.append(mos_op['ibias'] * nf * sign)
    monte['ibias_list'] = ibias_list
    sim_params['ibias'] = ibias_list[0]


if __name__ == "__main__":
    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()
    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    # fname = 'mismatch_char_n.yaml'
    fname = 'mismatch_char_p.yaml'
    specs = read_yaml(Path('data', 'bag3_testbenches', 'specs_dsn', fname))

    get_ibias_list(specs)

    sim = DesignManager(bprj, spec_dict=specs)
    sim.characterize_designs(generate=True, gen_sch=True, measure=True, load_from_file=False)
    offset_var_arr = None
    for dsn_name in sim.info.dsn_name_iter():
        result = sim.get_result(dsn_name)
        offset_var_list = result['mismatch_char']['offset_var_list']
        if offset_var_arr is None:
            offset_var_arr = np.asarray(offset_var_list)
        else:
            arr = np.asarray(offset_var_list)
            offset_var_arr = np.stack((offset_var_arr, arr))

    sch_params = specs['schematic_params']
    monte = specs['measurements'][0]['testbenches']['monte']
    vstar_list = monte['vstar_list']

    mismatch_dict = MismatchTBM.get_mismatch_parameters(vstar_list, offset_var_arr, sch_params)

    print(f'Avt={mismatch_dict["Avt_list"]}')
    print(f'Abeta={mismatch_dict["Abeta_list"]}')
