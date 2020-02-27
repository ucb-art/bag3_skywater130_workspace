# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, cast

import numpy as np
import scipy.optimize as opt

from bag.util.search import FloatBinaryIterator

from bag3_testbenches.measurement.mos.query import MOSDBDiscrete


def get_self_bias_inv(nch_db: MOSDBDiscrete, pch_db: MOSDBDiscrete, seg_dict: Dict[str, int],
                      vdd: float) -> Dict[str, float]:
    seg_n = seg_dict['n']
    seg_p = seg_dict['p']

    def zero_fun(vb_test):
        nch_op = nch_db.query(vgs=vb_test, vds=vb_test)
        pch_op = pch_db.query(vgs=vb_test-vdd, vds=vb_test-vdd)
        return nch_op['ibias'] * seg_n - pch_op['ibias'] * seg_p

    vb = cast(float, opt.brentq(zero_fun, 0, vdd))
    return dict(vb=vb)


def get_self_bias_clk_rx(nch_db: MOSDBDiscrete, pch_db: MOSDBDiscrete, seg_dict: Dict[str, int],
                         vdd: float) -> Dict[str, float]:
    seg_ngm = seg_dict['gmn']
    seg_pgm = seg_dict['gmp']
    seg_ntail = seg_dict['tailn']
    seg_ptail = seg_dict['tailp']

    tol_v = 1.0e-3
    tol_i = 1.0e-6

    def zero_fun_n(vn_test, vg_test):
        ngm_op = nch_db.query(vgs=vg_test-vn_test, vds=vg_test-vn_test, vbs=-vn_test)
        ntail_op = nch_db.query(vgs=vg_test, vds=vn_test)
        return ngm_op['ibias'] * seg_ngm - ntail_op['ibias'] * (seg_ntail // 2)

    def zero_fun_p(vp_test, vg_test):
        pgm_op = pch_db.query(vgs=vg_test-vp_test, vds=vg_test-vp_test, vbs=vdd-vp_test)
        ptail_op = pch_db.query(vgs=vg_test-vdd, vds=vp_test-vdd)
        return pgm_op['ibias'] * seg_pgm - ptail_op['ibias'] * (seg_ptail // 2)

    vg_bin = FloatBinaryIterator(0.0, vdd, tol=tol_v)
    while vg_bin.has_next():
        vg_cur = vg_bin.get_next()
        vn = cast(float, opt.brentq(zero_fun_n, 0, vg_cur, args=(vg_cur,)))
        ntail_op = nch_db.query(vgs=vg_cur, vds=vn)
        i_n = ntail_op['ibias'] * (seg_ntail // 2)

        vp = cast(float, opt.brentq(zero_fun_p, vg_cur, vdd, args=(vg_cur,)))
        ptail_op = pch_db.query(vgs=vg_cur-vdd, vds=vp-vdd)
        i_p = ptail_op['ibias'] * (seg_ptail // 2)

        save_info = dict(
            vg=vg_cur,
            vn=vn,
            vp=vp,
        )

        if np.abs(i_n - i_p) <= tol_i:
            vg_bin.save_info(save_info)
            break
        if i_n > i_p:
            vg_bin.down()
        else:
            vg_bin.up()

    fin_info = vg_bin.get_last_save_info()
    return fin_info



