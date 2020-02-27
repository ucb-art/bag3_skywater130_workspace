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

from typing import Dict, Any, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from bag.util.search import BinaryIterator
from bag.data.lti import LTICircuit

from .helper import get_db


class ComparatorDesignLTI:
    def __init__(self, specs: Dict[str, Any]) -> None:
        self._specs = specs
        self._sch_params = specs['schematic_params']
        self._sim_params = specs['measurements'][0]['testbenches']['ac']['sim_params']
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

    def get_switch_vals(self, sw_type: str, gm_in: float, vbias: float, vdd: float, vg_on: float,
                        vg_off: float, target_val: float = 0.25) -> Dict[str, Union[float, int]]:
        """
        Calculate number of fingers of switches to have gm_in * R_sw = target_val
        :param sw_type: switch is 'nch' or 'pch'
        :param gm_in: combined gm of active devices at the switching node
        :param vbias: bias voltage at switching node
        :param vdd: supply voltage
        :param vg_on: gate node voltage of switch when turned on
        :param vg_off: gate node voltage of switch when turned off
        :param target_val: Target value of gm_in * R_sw
        :return: required number of fingers of switch
        """
        if sw_type == 'nch':
            sw_op_on = self.nch_db.query(vgs=vg_on - vbias, vds=0.005, vbs=- vbias)
            sw_op_off = self.nch_db.query(vgs=vg_off - vbias, vds=0.005, vbs=- vbias)
        elif sw_type == 'pch':
            sw_op_on = self.pch_db.query(vgs=vg_on - vbias, vds=-0.005, vbs=vdd - vbias)
            sw_op_off = self.pch_db.query(vgs=vg_off - vbias, vds=-0.005, vbs=vdd - vbias)
        else:
            raise ValueError(f'Unknown mos_type={sw_type}. Supported values are "nch" and "pch".')

        gds_targ = gm_in / target_val
        nf_sw = np.rint(gds_targ / sw_op_on['gds'])

        results = dict(nf=int(-(- nf_sw // 2) * 2),
                       cdd_on=sw_op_on['cgd'],
                       cdd_off=sw_op_off['cgd'])

        return results

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

    def design_n_split(self, vbias: float, vss_g: float, cap_ratio: float, mode: str = 'single'
                       ) -> None:
        """
        Design methodology for nmos input, split gain and regeneration topology
        :param vbias: Target self-bias voltage during reset phase
        :param vss_g: Target power gated VSS during reset
        :param cap_ratio: Ratio of AC coupling cap to gate cap
        :param mode: 'diff' or 'single'
        :return:
        """
        doc_params = self.sch_params['comp_out_params']['comp_params']['doc_params']

        # query operating point of pch devices of second stage in reset phase
        vdd = self.sim_params['vdd']
        pch_op = self.pch_db.query(vgs=vbias-vdd, vds=vbias-vdd, vbs=0.0)
        gmp = pch_op['gm']
        cgp = pch_op['cgg']

        # query operating point of nch devices of second stage in reset phase
        nch_op = self.nch_db.query(vgs=vbias-vss_g, vds=vbias-vss_g, vbs=-vss_g)
        gmn = nch_op['gm']
        cgn = nch_op['cgg']

        # vbias is high so that we can use pch switches. Hence pch of inverters are larger than nch.
        # Design flow: keep nch sizes from specs constant, and design pch sizes
        seg_dict = doc_params['seg_dict']
        nf_1 = seg_dict['gm1']  # input nch
        nf_3n = seg_dict['gm2n']  # regeneration nch, in negative feedback mode during gain

        # To balance current, (nf_1 + nf_3n) * ibias_n = (nf_2 + nf_3p) * ibias_p = ibias_one
        ibias_one = (nf_1 + nf_3n) * nch_op['ibias']
        nfp_tot = ibias_one / pch_op['ibias']
        nfp_tot = int(-(- nfp_tot // 2) * 2)
        # Double of ibias_one flows through power gate nch
        nf_pow = self.get_pow_gate_nf('nch', vs=0.0, vg=vdd, vd=vss_g, vb=0.0, ibias=2 * ibias_one)
        print(nf_pow)

        # find size of switch across input nmos
        swc12_res = self.get_switch_vals('pch', gmn * nf_1, vbias, vdd, 0.0, vdd, target_val=0.5)
        # nf_swc12 = swc12_res['nf']
        nf_swc12 = int(-(- (nf_1 / 2) // 2) * 2)
        cdc12 = swc12_res['cdd_off'] * nf_swc12  # This switch is off.

        nf_3p_arr = np.arange(2, nfp_tot - 2, 2)
        gain_arr = np.empty(nf_3p_arr.shape)

        for idx, nf_3p in enumerate(nf_3p_arr):
            nf_2 = nfp_tot - nf_3p
            # Use LTICircuit
            cir = LTICircuit()

            # input transistors
            cir.add_transistor(nch_op, 'compM', 'acP', 'gnd_g', 'gnd', nf_1)
            cir.add_transistor(nch_op, 'compP', 'acM', 'gnd_g', 'gnd', nf_1)

            # switches for input transistors
            # cir.add_transistor(swc12_op_off, 'compM', 'gnd', 'acP', 'gnd', nf_swc12)
            # cir.add_transistor(swc12_op_off, 'compP', 'gnd', 'acM', 'gnd', nf_swc12)
            cir.add_cap(cdc12, 'compM', 'gnd')
            cir.add_cap(cdc12, 'acP', 'gnd')
            cir.add_cap(cdc12, 'compP', 'gnd')
            cir.add_cap(cdc12, 'acM', 'gnd')

            # regeneration nch
            cir.add_transistor(nch_op, 'compM', 'intP', 'gnd_g', 'gnd', nf_3n)
            cir.add_transistor(nch_op, 'compP', 'intM', 'gnd_g', 'gnd', nf_3n)

            # positive pch
            cir.add_transistor(pch_op, 'compM', 'intP', 'gnd', 'gnd', nf_3p)
            cir.add_transistor(pch_op, 'compP', 'intM', 'gnd', 'gnd', nf_3p)

            # switches for positive feedback
            # cir.add_transistor(sw12_op_off, 'compM', 'gnd', 'intP', 'gnd', nf_sw12)
            # cir.add_transistor(sw12_op_off, 'compP', 'gnd', 'intM', 'gnd', nf_sw12)
            sw12_res = self.get_switch_vals('pch', gmn * nf_3n + gmp * nf_3p, vbias, vdd, 0.0, vdd,
                                            target_val=0.5)
            # nf_sw12 = sw12_res['nf']
            nf_sw12 = int(-(- ((nf_3p + nf_3n * gmn / gmp) / 2) // 2) * 2)
            cd12 = sw12_res['cdd_off'] * nf_sw12  # This switch is off.
            cir.add_cap(cd12, 'compM', 'gnd')
            cir.add_cap(cd12, 'intP', 'gnd')
            cir.add_cap(cd12, 'compP', 'gnd')
            cir.add_cap(cd12, 'intM', 'gnd')

            # negative pch
            cir.add_transistor(pch_op, 'compM', 'compM', 'gnd', 'gnd', nf_2)
            cir.add_transistor(pch_op, 'compP', 'compP', 'gnd', 'gnd', nf_2)

            # switches for negative feedback
            # cir.add_transistor(sw34_op_on, 'compM', 'gnd', 'int2P', 'gnd', nf_sw34)
            # cir.add_transistor(sw34_op_on, 'compP', 'gnd', 'int2M', 'gnd', nf_sw34)
            sw34_res = self.get_switch_vals('pch', gmp * nf_2, vbias, vdd, 0.0, vdd, target_val=0.5)
            # nf_sw34 = sw34_res['nf']
            nf_sw34 = int(-(- (nf_2 / 2) // 2) * 2)
            cd34 = sw34_res['cdd_on'] * nf_sw34  # This switch is on.
            cir.add_cap(cd34, 'compM', 'gnd')
            cir.add_cap(cd34, 'int2P', 'gnd')
            cir.add_cap(cd34, 'compP', 'gnd')
            cir.add_cap(cd34, 'int2M', 'gnd')

            # power gate
            pow_op = self.nch_db.query(vgs=vdd, vds=vss_g, vbs=0.0)
            cir.add_transistor(pow_op, 'gnd_g', 'gnd', 'gnd', 'gnd', nf_pow)

            # caps
            cc_n = cap_ratio * (cgp * nf_2 + cd34)
            # cc_n = 10.0e-15
            cir.add_cap(cc_n, 'compM', 'compP')
            cir.add_cap(cc_n, 'compP', 'compM')
            cc_p = cap_ratio * (cgp * nf_3p + cgn * nf_3n + cd12)
            # cc_p = 10.0e-15
            cir.add_cap(cc_p, 'intP', 'compP')
            cir.add_cap(cc_p, 'intM', 'compM')
            cpre = cap_ratio * (cgn * nf_1 + cdc12)
            # cpre = 10.0e-15
            cir.add_cap(cpre, 'acP', 'sampP')
            if mode == 'diff':
                cir.add_cap(cpre, 'acM', 'gnd')
            elif mode == 'single':
                cir.add_cap(cpre, 'acM', 'compM')
            else:
                raise ValueError(f'Unknown mode = {mode}. Use "single" or "diff"')

            tf = cir.get_transfer_function('sampP', 'compM')
            w = 2 * np.pi * 1.0e9
            gain_arr[idx] = np.poly1d(tf.num)(w) / np.poly1d(tf.den)(w)

        plt.plot(nf_3p_arr, gain_arr)
        plt.show()
        opt_idx = np.argmin(gain_arr)  # looking for max negative gain
        gain = gain_arr[opt_idx-1]
        nf_3p = nf_3p_arr[opt_idx-1]
        # sw12_res = self.get_switch_vals('pch', gmn * nf_3n + gmp * nf_3p, vbias, vdd, 0.0, vdd,
        #                                 target_val=0.3)
        # nf_sw12 = sw12_res['nf']
        nf_sw12 = int(-(- ((nf_3p + nf_3n * gmn / gmp) / 2) // 2) * 2)
        nf_2 = nfp_tot - nf_3p
        # sw34_res = self.get_switch_vals('pch', gmp * nf_2, vbias, vdd, 0.0, vdd, target_val=0.3)
        # nf_sw34 = sw34_res['nf']
        nf_sw34 = int(-(- (nf_2 / 2) // 2) * 2)

        cc_n = cap_ratio * (cgp * nf_2)
        cc_p = cap_ratio * (cgp * nf_3p + cgn * nf_3n)
        cpre = cap_ratio * (cgn * nf_1)

        print(f'Expected gain = {gain}')
        print(f'nf_pow={nf_pow}  nf_2={nf_2}  nf_3p={nf_3p} cc_n={cc_n} cc_p={cc_p} cpre={cpre} '
              f'nf_swc12={nf_swc12} nf_sw12={nf_sw12} nf_sw34={nf_sw34}')

        seg_dict['pow'] = nf_pow
        seg_dict['gm2p'] = nf_2  # regeneration pch, in negative feedback mode during gain
        seg_dict['gm2p2'] = nf_3p  # regeneration pch, in positive feedback mode during gain
        self.sim_params['cpre'] = cpre
        self.sim_params['cc_p'] = cc_p
        self.sim_params['cc_n'] = cc_n

    def design_inv_split(self, vbias: float, cap_ratio: float, mode: str = 'single') -> None:
        """
        Design methodology for nmos input, split gain and regeneration topology
        :param vbias: Target self-bias voltage during reset phase
        :param cap_ratio: Ratio of AC coupling cap to gate cap
        :param mode: 'diff' or 'single'
        :return:
        """
        # doc_params = self.sch_params['comp_out_params']['comp_params']['doc_params']

        # query operating point of pch devices
        vdd = self.sim_params['vdd']
        pch_op1 = self.pch_db.query(vgs=vbias-vdd, vds=vbias-vdd, vbs=0.0)
        ibias_p = pch_op1['ibias']

        # query operating point of nch devices
        nch_op1 = self.nch_db.query(vgs=vbias, vds=vbias, vbs=0.0)
        ibias_n = nch_op1['ibias']

        pn_ratio = ibias_n / ibias_p
        # Strategy:
        # 1. Keep negative gm inverter fixed at min size
        # 2. Sweep positive gm inverter size to get max negative gain
        # 3. Size up input inverter to get gain > targ_gain

        if pn_ratio >= 1:
            nf_n = 2
            nf_p = -(- (nf_n * pn_ratio) // 2) * 2
        else:
            nf_p = 2
            nf_n = -(- (nf_p / pn_ratio) // 2) * 2
        nf_inv = np.array([nf_p, nf_n])  # unit inverter with even fingers for both n and p

        # recompute bias point and operating point with quantized number of fingers
        def fun_zero(vb: float):
            p_op = self.pch_db.query(vgs=vb-vdd, vds=vb-vdd, vbs=0.0)
            n_op = self.nch_db.query(vgs=vb, vds=vb, vbs=0.0)
            return (nf_p * p_op['ibias'] - nf_n * n_op['ibias']) * 1.0e6

        vbias_new = brentq(fun_zero, 0.0, vdd)
        print(f'New vbias = {vbias_new} V')
        pch_op = self.pch_db.query(vgs=vbias_new - vdd, vds=vbias_new - vdd, vbs=0.0)
        cgp = pch_op['cgg']
        print(f'pch current is {pch_op["ibias"] * nf_p}')

        # query operating point of nch devices
        nch_op = self.nch_db.query(vgs=vbias_new, vds=vbias_new, vbs=0.0)
        cgn = nch_op['cgg']
        print(f'nch current is {nch_op["ibias"] * nf_n}')

        cg_inv = cgn * nf_n + cgp * nf_p

        # 1: input inverter
        # 2: positive gm inverter
        # 3: negative gm inverter
        nf_1, nf_3 = 1, 2
        nf_2_bin_iter = BinaryIterator(1, None)

        def _make_circuit(nf1: int, nf2: int, nf3: int) -> Tuple[float, float, float, float, float]:
            # Use LTICircuit
            cir = LTICircuit()

            # input inverter
            cir.add_transistor(nch_op, 'compM', 'acP', 'gnd', 'gnd', nf1 * nf_inv[1])
            cir.add_transistor(pch_op, 'compM', 'acP', 'gnd', 'gnd', nf1 * nf_inv[0])

            cir.add_transistor(nch_op, 'compP', 'acM', 'gnd', 'gnd', nf1 * nf_inv[1])
            cir.add_transistor(pch_op, 'compP', 'acM', 'gnd', 'gnd', nf1 * nf_inv[0])

            # switches for input inverter
            # cir.add_cap(cdc12, 'compM', 'gnd')
            # cir.add_cap(cdc12, 'acP', 'gnd')
            # cir.add_cap(cdc12, 'compP', 'gnd')
            # cir.add_cap(cdc12, 'acM', 'gnd')

            # negative gm inverter
            cir.add_transistor(nch_op, 'compM', 'compM', 'gnd', 'gnd', nf3 * nf_inv[1])
            cir.add_transistor(pch_op, 'compM', 'compM', 'gnd', 'gnd', nf3 * nf_inv[0])

            cir.add_transistor(nch_op, 'compP', 'compP', 'gnd', 'gnd', nf3 * nf_inv[1])
            cir.add_transistor(pch_op, 'compP', 'compP', 'gnd', 'gnd', nf3 * nf_inv[0])

            # switches for negative gm
            # sw34_res = self.get_switch_vals('pch', gmp * nf_2, vbias, vdd, 0.0, vdd,
            #                                 target_val=0.5)
            # # nf_sw34 = sw34_res['nf']
            # nf_sw34 = int(-(- (nf_2 / 2) // 2) * 2)
            # cd34 = sw34_res['cdd_on'] * nf_sw34  # This switch is on.
            # cir.add_cap(cd34, 'compM', 'gnd')
            # cir.add_cap(cd34, 'int2P', 'gnd')
            # cir.add_cap(cd34, 'compP', 'gnd')
            # cir.add_cap(cd34, 'int2M', 'gnd')

            # positive gm inverter
            cir.add_transistor(nch_op, 'compM', 'intP', 'gnd', 'gnd', nf2 * nf_inv[1])
            cir.add_transistor(pch_op, 'compM', 'intP', 'gnd', 'gnd', nf2 * nf_inv[0])

            cir.add_transistor(nch_op, 'compP', 'intM', 'gnd', 'gnd', nf2 * nf_inv[1])
            cir.add_transistor(pch_op, 'compP', 'intM', 'gnd', 'gnd', nf2 * nf_inv[0])

            # switches for positive feedback
            # sw12_res = self.get_switch_vals('pch', gmn * nf_3n + gmp * nf_3p, vbias, vdd, 0.0,
            #                                 vdd, target_val=0.5)
            # # nf_sw12 = sw12_res['nf']
            # nf_sw12 = int(-(- ((nf_3p + nf_3n * gmn / gmp) / 2) // 2) * 2)
            # cd12 = sw12_res['cdd_off'] * nf_sw12  # This switch is off.
            # cir.add_cap(cd12, 'compM', 'gnd')
            # cir.add_cap(cd12, 'intP', 'gnd')
            # cir.add_cap(cd12, 'compP', 'gnd')
            # cir.add_cap(cd12, 'intM', 'gnd')

            # caps
            cc_n = cap_ratio * (cg_inv * nf3)
            # cc_n = cap_ratio * (cg_inv * nf_3 + cd34)
            # cc_n = 10.0e-15
            cir.add_cap(cc_n, 'compM', 'compP')
            cir.add_cap(cc_n, 'compP', 'compM')

            cc_p = cap_ratio * (cg_inv * nf2)
            # cc_p = cap_ratio * (cg_inv * nf_2 + cd12)
            # cc_p = 10.0e-15
            cir.add_cap(cc_p, 'intP', 'compP')
            cir.add_cap(cc_p, 'intM', 'compM')

            cpre = cap_ratio * (cg_inv * nf1)
            # cpre = cap_ratio * (cg_inv * nf_1 + cdc12)
            # cpre = 10.0e-15
            cir.add_cap(cpre, 'acP', 'sampP')
            if mode == 'diff':
                cir.add_cap(cpre, 'acM', 'gnd')
            elif mode == 'single':
                cir.add_cap(cpre, 'acM', 'compM')
            else:
                raise ValueError(f'Unknown mode = {mode}. Use "single" or "diff"')

            tf = cir.get_transfer_function('sampP', 'compM')
            w = 2 * np.pi * 10.0e6
            gain_w = np.abs(np.poly1d(tf.num)(1j * w) / np.poly1d(tf.den)(1j * w))
            gain_real = np.real(np.poly1d(tf.num)(1j * w) / np.poly1d(tf.den)(1j * w))

            tf2 = cir.get_transfer_function('sampP', 'acP')
            coup = np.abs(np.poly1d(tf2.num)(1j * w) / np.poly1d(tf2.den)(1j * w))
            print(f'gain = {gain_w}, coupling={coup}')

            return gain_w, gain_real, cpre, cc_n, cc_p

        while nf_2_bin_iter.has_next():
            nf_2 = nf_2_bin_iter.get_next()

            gain_w, gain_real, cpre, cc_n, cc_p = _make_circuit(nf_1, nf_2, nf_3)

            last_save_info = nf_2_bin_iter.get_last_save_info()
            if gain_real < 0:
                if last_save_info is None or np.abs(gain_w) > np.abs(last_save_info['gain']):
                    save_info = dict(
                        gain=gain_w,
                        cpre=cpre,
                        cc_p=cc_p,
                        cc_n=cc_n,
                    )
                    nf_2_bin_iter.save_info(save_info)
                nf_2_bin_iter.up()
            else:
                nf_2_bin_iter.down()

        nf_2 = nf_2_bin_iter.get_last_save()
        last_save_info = nf_2_bin_iter.get_last_save_info()
        targ_gain = 4

        if np.abs(last_save_info['gain']) >= np.abs(targ_gain):
            pass
        else:
            nf_1_bin_iter = BinaryIterator(2, None)

            while nf_1_bin_iter.has_next():
                nf_1 = nf_1_bin_iter.get_next()

                gain_w, gain_real, cpre, cc_n, cc_p = _make_circuit(nf_1, nf_2, nf_3)

                last_save_info = nf_1_bin_iter.get_last_save_info()
                if np.abs(gain_w) > np.abs(targ_gain):
                    if last_save_info is None:
                        save_info = dict(
                            gain=gain_w,
                            cpre=cpre,
                            cc_p=cc_p,
                            cc_n=cc_n,
                        )
                        nf_1_bin_iter.save_info(save_info)
                    nf_1_bin_iter.down()
                else:
                    nf_1_bin_iter.up()

            nf_1 = nf_1_bin_iter.get_last_save()
            last_save_info = nf_1_bin_iter.get_last_save_info()

        print(f'Expected gain = {last_save_info["gain"]}')
        print(f'nf_1={nf_1}  nf_2={nf_2}  nf_3={nf_3}  nf_inv=({nf_inv[0]}, {nf_inv[1]})')
        print(f'cpre={last_save_info["cpre"]}  cc_p={last_save_info["cc_p"]}  '
              f'cc_n={last_save_info["cc_n"]}')

        # seg_dict['pow'] = nf_pow
        # seg_dict['gm2p'] = nf_2  # regeneration pch, in negative feedback mode during gain
        # seg_dict['gm2p2'] = nf_3p  # regeneration pch, in positive feedback mode during gain
        # self.sim_params['cpre'] = cpre
        # self.sim_params['cc_p'] = cc_p
        # self.sim_params['cc_n'] = cc_n
