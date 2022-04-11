# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Union, Sequence, Tuple, Any, Dict

import os
import math
from pathlib import Path

import numpy as np
import scipy.optimize as opt

from bag.core import BagProject
from bag.io import read_yaml
from bag.util.search import FloatBinaryIterator

from bag3_testbenches.measurement.mos.query import MOSDBDiscrete


def get_db(mos_type: str, dsn_specs: Dict[str, Any]) -> MOSDBDiscrete:
    mos_specs = dsn_specs[mos_type]

    spec_file = mos_specs['spec_file']
    interp_method = mos_specs.get('interp_method', 'spline')
    sim_env = mos_specs.get('sim_env', 'tt_25')

    db = MOSDBDiscrete([spec_file], interp_method=interp_method)
    db.env_list = [sim_env]
    db.set_dsn_params(intent=mos_specs['intent'])

    return db


def get_vb_crossing(nch_db: MOSDBDiscrete, pch_db: MOSDBDiscrete, nf_n2: int, nf_p2: int,
                    nf_p1: int, vdd: float) -> float:
    def zero_fun(vb_test):
        nch_op = nch_db.query(vgs=vb_test, vds=vb_test)
        pch_op = pch_db.query(vgs=vb_test-vdd, vds=vb_test-vdd)
        return nch_op['ibias'] * nf_n2 - pch_op['ibias'] * (nf_p1 + nf_p2)
    vb = opt.brentq(zero_fun, 0, vdd)
    return vb


def get_vb_crossing_g(nch_db: MOSDBDiscrete, pch_db: MOSDBDiscrete, nf_n2: int, nf_p2: int,
                      nf_p1: int, nf_pow: int, vdd: float) -> Tuple[float, float]:
    def zero_fun(vb_test, vd_test):
        nch_op = nch_db.query(vgs=vb_test, vds=vb_test)
        pch_op = pch_db.query(vgs=vb_test-vd_test, vds=vb_test-vd_test)
        return nch_op['ibias'] * nf_n2 - pch_op['ibias'] * (nf_p1 + nf_p2)

    tol_v = 0.001
    tol_i = 1.0e-6
    vdd_bin_iter = FloatBinaryIterator(low=0.0, high=vdd, tol=tol_v)
    while vdd_bin_iter.has_next():
        vdd_g = vdd_bin_iter.get_next()
        vb = opt.brentq(zero_fun, 0, vdd_g, args=(vdd_g,))
        nch_op = nch_db.query(vgs=vb, vds=vb)
        i_n = nch_op['ibias'] * nf_n2
        pch_op = pch_db.query(vgs=-vdd, vds=vdd_g-vdd)
        i_p = pch_op['ibias'] * nf_pow / 2
        if np.abs(i_n - i_p) <= tol_i:
            break
        else:
            if i_n > i_p:
                vdd_bin_iter.down()
            else:
                vdd_bin_iter.up()
    return vb, vdd_g

