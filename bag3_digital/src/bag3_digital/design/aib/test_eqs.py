
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import numpy as np
from bag3_digital.design.digital_db.db import DigitalDB, DigitalMeasType
from bag.core import BagProject

import pdb


def run_main(prj: BagProject):

    stack2_config_file = 'specs_digital_char/digitalDB_specs_lvl_stack2.yaml'
    stack1_config_file = 'specs_digital_char/digitalDB_specs_lvl_stack1.yaml'

    input_db = DigitalDB(prj, stack2_config_file, dut_type='nmos', force_sim=True)
    p_db = DigitalDB(prj, stack1_config_file, dut_type='pmos', force_sim=True)
    n_db = DigitalDB(prj, stack1_config_file, dut_type='nmos', force_sim=True)

    params_dict = dict(
        lch=36,
        w_p=2,
    )
    input_params = input_db.query(params_dict, DigitalMeasType.DELAY_L2H, env='tt_25')
    inv_pmos_params = p_db.query(params_dict, DigitalMeasType.DELAY_H2L, vdd=0.8, env='tt_25')
    inv_nmos_params = n_db.query(params_dict, DigitalMeasType.DELAY_H2L, vdd=0.8, env='tt_25')
    rst_nmos_params = n_db.query(params_dict, DigitalMeasType.DELAY_H2L, vdd=0.8, env='tt_25')
    xcoupled_params = p_db.query(params_dict, DigitalMeasType.DELAY_H2L, vdd=1, env='tt_25')

    rst_ratio = 2
    fout = 4
    alpha = 1  # fudge factor coefficient for the fight between the right nmos and xcoupled pmos
    beta = 1  # fudge factor coefficient for the fight between the left nmos and xcoupled pmos
    rn = input_params['res']
    rp = xcoupled_params['res']
    cd_in = input_params['cd']
    cg_in = input_params['cg']
    cd_rst = rst_nmos_params['cd']
    cg_xp = xcoupled_params['cg']
    cd_xp = xcoupled_params['cd']

    cg_inv = inv_nmos_params['cg'] + inv_pmos_params['cg']
    cd_inv = inv_nmos_params['cd'] + inv_pmos_params['cd']
    rp_inv = inv_pmos_params['res']

    def t1_l2h(w1, w2):
        td = 1 / (w1 / rn - 1 / beta * w2 / rp) * \
             (w1 * cd_in + rst_ratio * w1 * cd_rst + w2 * (cg_xp + cd_xp))
        return td

    def t2_l2h(w1, w2, cload):
        td = rp / w2 * (w2 * (cg_xp + cd_xp) + w1 * cd_in + rst_ratio * w1 * cd_rst + cload)
        return td

    def tdl2h(w1, w2, cload):
        td1 = t1_l2h(w1, w2)
        td2 = t2_l2h(w1, w2, cload)
        print('l2h: td1 = ', td1)
        print('l2h: td2 = ', td2)
        _tdl2h = td1 + td2
        return _tdl2h

    def t1_h2l(w1, winv):
        td = rp_inv / winv * (winv * cd_inv + w1 * cg_in)
        return td

    def t2_h2l(w1, w2, cload):
        td = 1 / (w1 / rn - 1 / alpha * w2 / rp) * \
              (w2 * (cg_xp + cd_xp) + w1 * cd_in + rst_ratio * w1 * cd_rst + cload)
        return td

    def tdh2l(w1, w2, winv, cload):
        td1 = t1_h2l(w1, winv)
        td2 = t2_h2l(w1, w2, cload)
        _tdh2l = td1 + td2
        print('h2l: td1 = ', td1)
        print('h2l: td2 = ', td2)
        return _tdh2l

    winv = 1
    k = rp / rn * alpha / (alpha + 1)

    def fzero(w1):
        return t1_l2h(w1, k * w1) - t1_h2l(w1, winv)

    w1_arr = np.linspace(winv, 20 * winv, 100)
    t1 = t1_l2h(w1_arr, k * w1_arr)
    t2 = t1_h2l(w1_arr, winv)
    plt.plot(w1_arr, t1, label='t1')
    plt.plot(w1_arr, t2, label='t2')
    plt.legend()
    # plt.show()

    w1 = float(brentq(fzero, winv, 20 * winv))
    w2 = k * w1
    cin = (w1 * cg_in + winv * cg_inv)
    cload = fout * cin

    print(f'tdl2h = {tdl2h(w1, w2, cload)}')
    print(f'tdh2l = {tdh2l(w1, w2, winv, cload)}')

    rpu = rp / w2
    rpd = 1 / (w1 / rn - 1 / alpha * w2 / rp)

    print(f'rpu = {rpu}, rpd = {rpd}')
    le = rpu * cin / inv_pmos_params['res'] / cg_inv
    print(f'le = {le}')

    coef = 3
    print(f'winv = {winv * coef}')
    print(f'w1 = {w1 * coef}')
    print(f'w2 = {w2 * coef}')
    print(f'cload = {cload * coef}')
    print(f'inv2 = {cin / 4 / cg_inv * coef}')
    print(f'inv1 = {cin / 16 / cg_inv * coef}')
    pdb.set_trace()


if __name__ == '__main__':

    local_dict = locals()
    if 'prj' not in local_dict:
        print('creating bag project')
        prj = BagProject()
    else:
        print('loading bag project')
        prj = local_dict['prj']

    run_main(prj)

    pdb.set_trace()