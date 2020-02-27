# SPDX-License-Identifier: Apache-2.0
# Copyright 2019 Blue Cheetah Analog Design Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This package contains design class for comparators."""

from typing import Optional, Tuple, Dict, Any, Sequence, Union, List, cast

import math
import importlib
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import scipy.interpolate as interp
import scipy.optimize as sciopt
from scipy.stats import norm
import matplotlib.pyplot as plt

from bag.simulation.data import (
    SimNetlistInfo, netlist_info_from_dict, SimData, AnalysisData, AnalysisType
)
from bag.simulation.base import SimAccess
from bag.simulation.core import MeasurementManager, TestbenchManager
from bag.simulation.hdf5 import save_sim_data_hdf5
from bag.math.interpolate import LinearInterpolator
from bag.util.search import FloatBinaryIterator

from .helper import get_db


class ComparatorDesign:
    def __init__(self, specs: Dict[str, Any]) -> None:
        self._specs = specs
        self._sch_params = specs['schematic_params']
        self._sim_params = specs['measurements'][0]['testbenches']['overdrive']['sim_params']
        self.nch_db = get_db('nch', specs)
        self.pch_db = get_db('pch', specs)

    @property
    def specs(self) -> Dict[str, Any]:
        return self._specs

    @property
    def sch_params(self) -> Dict[str, Any]:
        return self._sch_params

    @property
    def sim_params(self) -> Dict[str, Any]:
        return self._sim_params

    def get_switch_nf(self, sw_type: str, gm_in: float, vbias: float, vdd: float, vg: float,
                      target_val: float = 0.25) -> int:
        """
        Calculate number of fingers of switches to have gm_in * R_sw = target_val
        :param sw_type: switch is 'nch' or 'pch'
        :param gm_in: combined gm of active devices at the switching node
        :param vbias: bias voltage at switching node
        :param vdd: supply voltage
        :param vg: gate node voltage of switch when turned on
        :param target_val: Target value of gm_in * R_sw
        :return: required number of fingers of switch
        """
        if sw_type == 'nch':
            sw_op = self.nch_db.query(vgs=vg - vbias, vds=0.0, vbs=- vbias)
        elif sw_type == 'pch':
            sw_op = self.pch_db.query(vgs=vg - vbias, vds=0.0, vbs=vdd - vbias)
        else:
            raise ValueError(f'Unknown mos_type={sw_type}. Supported values are "nch" and "pch".')

        gds_targ = gm_in / target_val
        nf_sw = np.rint(gds_targ / sw_op['gds'])
        return int(-(- nf_sw // 2) * 2)

    def get_pow_gate_nf(self, mos_type: str, vs: float, vg: float, vd: float, vb: float,
                        ibias: float) -> int:
        """
        Calculate number of fingers of power gate device
        :param mos_type: power gating device is 'nch' or 'pch'
        :param vs: source node voltage of power gate
        :param vg: gate node voltage of power gate
        :param vd: target drain node voltage of power gate
        :param vb: body node voltage of power gate
        :param ibias: max drain current flowing through power gate
        :return: required number of fingers of power gate
        """
        if mos_type == 'nch':
            pow_op = self.nch_db.query(vgs=vg - vs, vds=vd - vs, vbs=vb - vs)
        elif mos_type == 'pch':
            pow_op = self.pch_db.query(vgs=vg - vs, vds=vd - vs, vbs=vb - vs)
        else:
            raise ValueError(f'Unknown mos_type={mos_type}. Supported values are "nch" and "pch".')

        nf_pow = np.rint(ibias / pow_op['ibias'])
        return int(-(- nf_pow // 2) * 2)

    def design_n_split(self, vbias: float, vss_g: float, mode: str = 'single') -> None:
        """
        Design methodology for nmos input, split gain and regeneration topology
        :param vbias: Target self-bias voltage during reset phase
        :param vss_g: Target power gated VSS during reset
        :param mode: 'diff' or 'single'
        :return:
        """
        doc_params = self.sch_params['comp_out_params']['comp_params']['doc_params']

        # query operating point of pch devices of second stage in reset phase
        vdd = self.sim_params['vdd']
        pch_op = self.pch_db.query(vgs=vbias-vdd, vds=vbias-vdd, vbs=0.0)
        gmp = pch_op['gm']
        cgp = pch_op['cgg']
        gdsp = pch_op['gds']

        # query operating point of nch devices of second stage in reset phase
        nch_op = self.nch_db.query(vgs=vbias-vss_g, vds=vbias-vss_g, vbs=-vss_g)
        gmn = nch_op['gm']
        cgn = nch_op['cgg']
        gdsn = nch_op['gds']
        print(f'gdsp={gdsp}, gdsn={gdsn}')

        # vbias is high so that we can use pch switches. Hence pch of inverters are larger than nch.
        # Design flow: keep nch sizes from specs constant, and design pch sizes
        seg_dict = doc_params['seg_dict']
        nf_1 = seg_dict['gm1']  # input nch
        nf_3n = seg_dict['gm2n']  # regeneration nch, in negative feedback mode during gain

        # To balance current, (nf_1 + nf_3n) * ibias_n = (nf_2 + nf_3p) * ibias_p = ibias_one
        ibias_one = (nf_1 + nf_3n) * nch_op['ibias']
        # Double of ibias_one flows through power gate nch
        nf_pow = self.get_pow_gate_nf('nch', vs=0.0, vg=vdd, vd=vss_g, vb=0.0, ibias=2 * ibias_one)

        # In gain phase, negative feedback gm should be close to positive feedback gm.
        # Negative feedback gm is nf_2 * gm_p
        # Positive feedback gm is (nf_3p * gm_p + nf_3n * gm_n) * Cc / (Cc + Cpar)
        # where Cpar = nf_3p * Cgg_p + nf_3n * Cgg_n + nf_sw * Cdd_sw

        nf_sw = seg_dict['sw12']
        sw_op_off = self.pch_db.query(vgs=vdd-vbias, vds=0.0, vbs=vdd-vbias)  # The switch is off.
        cdsw = sw_op_off['cdd']

        cc = self.sim_params['cc']

        # The 2 equations we need to solve are the current balance and gm balance
        nfp_tot = ibias_one / pch_op['ibias']
        eps = 0.00

        def fun2(nf: Union[float, np.ndarray]):
            return (nf * gmp + nf_3n * gmn) * cc / (cc + nf * cgp + nf_3n * cgn + nf_sw * cdsw)

        def fun(nf: Union[float, np.ndarray]):
            return (gmp * (nfp_tot - nf))**2 + eps - fun2(nf) * (nf_1 * gmn * cc / (cc + nf_1 *
                                                                 cgn + nf_sw * cdsw) + fun2(nf))

        def fun3(nf: Union[float, np.ndarray]):
            return gmp * (nfp_tot - nf) + eps - fun2(nf)

        # nf_list = np.linspace(1, nfp_tot, num=nfp_tot)
        # plt.plot(nf_list, fun(nf_list))
        # plt.show()
        if mode == 'single':
            nf_3p = sciopt.brentq(fun, 1, nfp_tot)
        elif mode == 'diff':
            nf_3p = sciopt.brentq(fun3, 1, nfp_tot)
        else:
            raise ValueError(f'Unknown mode = {mode}. Use "single" or "diff".')
        nf_2 = nfp_tot - nf_3p

        nf_3p = int(-(- nf_3p // 2) * 2)
        nf_2 = int(-(- nf_2 // 2) * 2)

        # check gm_in * R_sw
        sw_op_on = self.pch_db.query(vgs=-vbias, vds=0.005, vbs=vdd-vbias)  # The switch is on.
        gds_sw = sw_op_on['gds']
        R_sw = 1 / (nf_sw * gds_sw)
        gm_in = (nf_1 + nf_3n) * gmn + (nf_2 + nf_3p) * gmp
        print(f'gm_in * R_sw = {gm_in * R_sw}')

        print(f'nf_pow={nf_pow}  nf_2={nf_2}  nf_3p={nf_3p}')

        seg_dict['pow'] = nf_pow
        seg_dict['gm2p'] = nf_2  # regeneration pch, in negative feedback mode during gain
        seg_dict['gm2p2'] = nf_3p  # regeneration pch, in positive feedback mode during gain

        if mode == 'single':
            gain = nf_1 * gmn / (- nf_2 * gmp + 0.82 * fun2(nf_3p) * (nf_1 * gmn * cc / (cc + nf_1 *
                                 cgn * 0.93 + nf_sw * cdsw) + 0.82 * fun2(nf_3p)) / (nf_2 * gmp))
        elif mode == 'diff':
            gain = nf_1 * gmn / (- nf_2 * gmp + fun2(nf_3p))
        else:
            raise ValueError(f'Unknown mode = {mode}. Use "single" or "diff".')
        print(f'Expected gain is {gain}')

        print(f'k1 = {cc / (cc + nf_1 * cgn + nf_sw * cdsw)}')
        print(f'k3 = {cc / (cc + nf_3p * cgp + nf_3n * cgn + nf_sw * cdsw)}')
