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


from typing import Dict, Any, Union, List, cast, Mapping, Tuple, Sequence
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from bag.simulation.design import DesignerBase
from bag.util.search import BinaryIterator
from bag.env import get_tech_global_info
from bag.simulation.cache import DesignInstance
from bag.concurrent.util import GatherHelper

from bag3_testbenches.measurement.tran.digital import DigitalTranTB
from bag3_testbenches.measurement.data.tran import EdgeType
from bag3_testbenches.measurement.dc.base import DCTB

from bag3_analog.layout.amplifier.diffamp import DiffAmpSelfBiasedGuardRing

from xbase.layout.mos.top import GenericWrapper


def r_even(num: float) -> int:
    """Round to next positive even integer."""
    num0 = int(np.rint(num))
    return max(num0 + (num0 & 1), 2)


class DiffampSelfBiasedDesigner(DesignerBase):

    @staticmethod
    def design_inv_chain_fo4(c_load: float) -> Dict[str, Any]:
        """Design inverter chain with 2 inverters and FO4, with common mode of vdd / 2 because
        diffamp will have almost rail-to-rail output"""
        c_in = c_load / 16

        tech_info = get_tech_global_info('aib_ams')
        cin_inv_seg = tech_info['cin_inv']['cin_per_seg']
        inv_beta = tech_info['inv_beta']

        size_n = c_in / cin_inv_seg
        size_p = size_n * inv_beta

        return dict(
            seg_n=[int(np.rint(size_n)), int(np.rint(size_n * 4))],
            seg_p=[int(np.rint(size_p)), int(np.rint(size_p * 4))],
        )

    def design_diffamp_eqn(self, target: Mapping[str, Any]) -> Dict[str, Any]:
        c_load = target['c_load']

        # design the inverter chain first
        inv_chain_des = self.design_inv_chain_fo4(c_load)

        seg_dict = dict(
            tail_p=4,
            gm_p=2,
            gm_n=2,
            tail_n=4,
            invp=inv_chain_des['seg_p'],
            invn=inv_chain_des['seg_n'],
        )

        return dict(
            seg_dict=seg_dict,
        )

    @classmethod
    def _modify_nf(cls, cur_des: Dict[str, Any], upsize_p: int = 0, upsize_n: int = 0) -> None:
        seg_dict = cur_des['seg_dict']
        seg_dict['gm_p'] += upsize_p
        seg_dict['tail_p'] = 2 * seg_dict['gm_p']
        seg_dict['gm_n'] += upsize_n
        seg_dict['tail_n'] = 2 * seg_dict['gm_n']

    @staticmethod
    def _update_params(pinfo: Mapping[str, Any], seg_dict: Dict[str, Union[int, List[int]]]
                       ) -> Mapping[str, Any]:
        dut_params = dict(
            pinfo=pinfo,
            seg_dict=dict(
                tail_p=seg_dict['tail_p'],
                gm_p=seg_dict['gm_p'],
                gm_n=seg_dict['gm_n'],
                tail_n=seg_dict['tail_n']
            ),
            segp_list=seg_dict['invp'],
            segn_list=seg_dict['invn'],
            show_pins=True,
            draw_taps='BOTH',
            export_mid=True,
        )

        return dut_params

    @staticmethod
    def _get_wrapper_params() -> Mapping[str, Any]:
        # dut_lib and dut_cell get added by SimulationDB
        wrapper_params = dict(
            export_mid=True,
        )

        return wrapper_params

    @staticmethod
    def _get_tbm_specs(tbm_specs: Mapping[str, Any], sim_envs: Sequence[str],
                       env_params: Mapping[str, Any], target: Mapping[str, Any]
                       ) -> Mapping[str, Any]:

        return dict(
            dut_pins=['VDD', 'VSS', 'clk', 'clkb', 'clk_out', 'v_mid'],
            pulse_list=[dict(pin='clk',
                             tper='tper',
                             tpw='tper/2',
                             trf='trf',
                             pos=True,
                             ),
                        dict(pin='clkb',
                             tper='tper',
                             tpw='tper/2',
                             trf='trf',
                             pos=False,
                             )],
            load_list=[dict(pin='clk_out',
                            type='cap',
                            value='c_load',
                            )],
            sup_values=dict(VSS=0,
                            VDD=env_params['vdd'],
                            VSS_in=target['v_cm'] - target['v_amp'],
                            VDD_in=target['v_cm'] + target['v_amp'],
                            ),
            pwr_domain=dict(clk=('VSS_in', 'VDD_in'),
                            clkb=('VSS_in', 'VDD_in'),
                            clk_out=('VSS', 'VDD'),
                            v_mid=('VSS', 'VDD'),
                            ),
            pin_values={},
            sim_envs=sim_envs,
            save_outputs=['clk', 'clkb', 'clk_out', 'v_mid'],
            **tbm_specs,
        )

    @staticmethod
    def _get_dc_tbm_specs(dc_tbm_specs: Mapping[str, Any], sim_envs: Sequence[str],
                          env_params: Mapping[str, Any], target: Mapping[str, Any],
                          ) -> Mapping[str, Any]:
        v_cm = target['v_cm']
        v_amp = target['v_amp']
        return dict(
            dut_pins=['VDD', 'VSS', 'clk', 'clkb', 'clk_out', 'v_mid'],
            src_list=[dict(type='vdc',
                           lib='analogLib',
                           value='v_sweep',
                           conns=dict(PLUS='clk', MINUS='VCM'),
                           ),
                      dict(type='vdc',
                           lib='analogLib',
                           value='v_sweep',
                           conns=dict(PLUS='VCM', MINUS='clkb'),
                           )],
            load_list=[dict(pin='clk_out',
                            type='cap',
                            value='c_load',
                            )],
            sup_values=dict(VSS=0,
                            VDD=env_params['vdd'],
                            VCM=v_cm,
                            ),
            pwr_domain=dict(clk=('VSS', 'VDD'),
                            clkb=('VSS', 'VDD'),
                            clk_out=('VSS', 'VDD'),
                            ),
            pin_values={},
            sim_envs=sim_envs,
            save_outputs=['clk', 'clkb', 'clk_out', 'v_mid'],
            dc_options=dict(hysteresis='yes'),
            sweep_var='v_sweep',
            sweep_options=dict(
                type='LINEAR',
                start=-v_amp,
                stop=v_amp,
                num=100,
            ),
            **dc_tbm_specs,
        )

    async def meas_cur_design(self, cur_des: Mapping[str, Any], pinfo: Mapping[str, Any],
                              tbm_specs: Mapping[str, Any], target: Mapping[str, Any]
                              ) -> Tuple[float, float, float, DesignInstance]:
        # Create DUT
        dut_params = self._update_params(pinfo, cur_des['seg_dict'])

        gen_params = dict(
            cls_name=DiffAmpSelfBiasedGuardRing.get_qualified_name(),
            params=dut_params,
        )

        seg_dict = cur_des['seg_dict']
        dsn_num = f'{seg_dict["tail_n"]}_{seg_dict["gm_n"]}_{seg_dict["gm_p"]}_{seg_dict["tail_p"]}'
        dut = await self.async_new_dut(f'dut{dsn_num}', GenericWrapper, gen_params)
        self.log(f'DUT name is dut{dsn_num}')

        # setup simulation across corners
        tech_info = get_tech_global_info('aib_ams')
        dsn_envs = tech_info['dsn_envs']

        helper = GatherHelper()
        corner_names = ['center']
        for name in corner_names:
            corner = dsn_envs[name]['env']
            vdd = dsn_envs[name]['vdd']
            helper.append(self._meas_corner(dut, target, tbm_specs, corner, vdd, 'design'))

        results = await helper.gather_err()
        duty_cycle_mean = 0.0
        tr_max = 0.0
        tf_max = 0.0
        num = len(corner_names)
        for idx, name in enumerate(corner_names):
            tr, tf, duty_cycle = results[idx]
            tr_max = max(tr_max, tr)
            tf_max = max(tf_max, tf)
            duty_cycle_mean += duty_cycle
        duty_cycle_mean /= num
        return tr_max, tf_max, duty_cycle_mean, dut

    async def meas_cur_dut(self, dut: DesignInstance, tbm: DigitalTranTB, name: str,
                           sim_type: str,) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tbm_params = dict(
            dut_lib='bag3_analog',
            dut_cell='clk_rx_wrapper',
            dut_params=self._get_wrapper_params(),
        )

        sim_results = await self.async_simulate_tbm_obj(f'sim_{name}', dut, tbm, tbm_params)

        sim_data = sim_results.data

        if sim_type == 'tran':
            tr = tbm.calc_trf(sim_data, 'v_mid', True)
            tf = tbm.calc_trf(sim_data, 'v_mid', False)
            self.log(f'Current design is: {name}_{dut.cell_name}')
            self.log(f'Rise time: {tr}')
            self.log(f'Fall time: {tf}')
            if tr[0] is np.inf or tf[0] is np.inf:
                self.error('Check simulation setup.')

            tdr = tbm.calc_delay(sim_data, 'clk', 'clk_out', EdgeType.RISE, EdgeType.RISE)
            tdf = tbm.calc_delay(sim_data, 'clk', 'clk_out', EdgeType.FALL, EdgeType.FALL)

            tper = tbm.specs['sim_params']['tper']
            tpw = tper / 2
            tpw_out = tpw + tdf - tdr
            duty_cycle = tpw_out / tper * 100
            self.log(f'Duty cycle: {duty_cycle}')

            return np.max(tr), np.max(tf), np.mean(duty_cycle)
        elif sim_type == 'dc':
            v_in = sim_data['clk']
            v_inb = sim_data['clkb']
            v_out = sim_data['v_mid']
            return v_in, v_inb, v_out
        else:
            self.error(f'Unknown sim_type = {sim_type}.')

    async def meas_vtc(self, dut: DesignInstance, dc_tbm_specs: Mapping[str, Any],
                       target: Mapping[str, Any]) -> None:
        # setup simulation across corners
        tech_info = get_tech_global_info('aib_ams')
        dsn_envs = tech_info['dsn_envs']

        helper = GatherHelper()
        corner_names = ['slow_io', 'center']
        for name in corner_names:
            corner = dsn_envs[name]['env']
            vdd = dsn_envs[name]['vdd']
            helper.append(self._meas_corner(dut, target, dc_tbm_specs, corner, vdd, 'vtc'))

        results = await helper.gather_err()

        num = len(corner_names)
        for idx, name in enumerate(corner_names):
            plt.subplot(num, 1, idx + 1)
            v_in, v_inb, v_out = results[idx]
            plt.plot(v_in[0], v_out[0], label='clk_out')
            plt.plot(v_in[0][:-1], np.diff(v_out[0]) / np.diff(v_in[0]), label='slope_clk_out')
            plt.plot(v_in[0], v_inb[0], label='clk_inb')
            plt.xlabel('clk_in (in V)')
            plt.ylabel('Magnitude (in V)')
            plt.legend()
        plt.show()

    async def signoff_dut(self, dut: DesignInstance, target: Mapping[str, Any],
                          tbm_specs: Mapping[str, Any]
                          ):
        self.log('--- Beginning sign off across corners ---')
        tech_info = get_tech_global_info('aib_ams')
        all_corners = tech_info['signoff_envs']['all_corners']

        helper = GatherHelper()
        for corner in all_corners['envs']:
            vdd = all_corners['vdd'][corner]
            helper.append(self._meas_corner(dut, target, tbm_specs, corner, vdd, 'signoff'))

        results = await helper.gather_err()
        duty_cycle_mean = 0
        num = len(all_corners['envs'])
        for idx, corner in enumerate(all_corners['envs']):
            tr, tf, duty_cycle = results[idx]
            if max(tr, tf) > target['trf']:
                self.error(f'trf spec not met at corner {corner}')
            duty_cycle_mean += duty_cycle
        duty_cycle_mean /= num
        if np.abs(duty_cycle_mean - 50.0) > target['duty_cycle_tol_signoff']:
            self.error(f'mean duty cycle spec not met')
        self.log('All corners signed off.')

    async def _meas_corner(self, dut: DesignInstance, target: Mapping[str, Any],
                           tbm_specs: Mapping[str, Any], corner: Union[str, List[str]], vdd: float,
                           mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # setup TBM
        if isinstance(corner, str):
            corner = [corner]
        if mode == 'vtc':
            tbm_specs_full = self._get_dc_tbm_specs(tbm_specs, corner, {'vdd': vdd}, target)
            sim_type = 'dc'
            tbm = cast(DCTB, self.make_tbm(DCTB, tbm_specs_full))
        else:
            tbm_specs_full = self._get_tbm_specs(tbm_specs, corner, {'vdd': vdd}, target)
            sim_type = 'tran'
            tbm = cast(DigitalTranTB, self.make_tbm(DigitalTranTB, tbm_specs_full))

        # simulate and measure
        return await self.meas_cur_dut(dut, tbm, f'{mode}_{corner[0]}', sim_type)

    async def async_design(self, target: Mapping[str, Any], pinfo: Mapping[str, Any],
                           tbm_specs: Mapping[str, Any], dc_tbm_specs: Mapping[str, Any],
                           ) -> Mapping[str, Any]:
        # Step 1: Get initial design in tt corner
        cur_des = self.design_diffamp_eqn(target)

        # Step 2: setup binary iterators
        p_bin_iter = BinaryIterator(2, 100, 2)
        n_bin_iter = BinaryIterator(2, 100, 2)
        targ_trf = target['trf']

        # Step 3: Find p size to meet rise time
        p_bin_iter.set_current(2)
        tr = 0
        tf = 0
        duty_cycle = 0
        while p_bin_iter.has_next():
            cur_p = p_bin_iter.get_next()
            cur_des['seg_dict']['gm_p'] = cur_p
            cur_des['seg_dict']['tail_p'] = 2 * cur_p
            tr, tf, duty_cycle, dut = await self.meas_cur_design(cur_des, pinfo, tbm_specs, target)
            if tr > targ_trf:
                self.log('1st binary search for p: rise time too slow, upsize p')
                p_bin_iter.up()
            else:
                p_bin_iter.save()
                p_bin_iter.save_info(dut)
                if duty_cycle < 50.0:
                    self.log('1st binary search for p: rise time met, duty cycle < 50, breaking')
                    break
                self.log('1st binary search for p: rise time met, duty cycle > 50, downsize p')
                p_bin_iter.down()
        cur_p = p_bin_iter.get_last_save()
        cur_dut = p_bin_iter.get_last_save_info()
        cur_des['seg_dict']['gm_p'] = cur_p
        cur_des['seg_dict']['tail_p'] = 2 * cur_p

        # Step 4: Find n size to meet fall time
        if tf > targ_trf:
            n_bin_iter.set_current(4)
            while n_bin_iter.has_next():
                cur_n = n_bin_iter.get_next()
                cur_des['seg_dict']['gm_n'] = cur_n
                cur_des['seg_dict']['tail_n'] = 2 * cur_n
                tr, tf, duty_cycle, dut = await self.meas_cur_design(cur_des, pinfo, tbm_specs,
                                                                     target)
                if tf > targ_trf:
                    self.log('1st binary search for n: fall time too slow, upsize n')
                    n_bin_iter.up()
                else:
                    n_bin_iter.save()
                    n_bin_iter.save_info(dut)
                    if duty_cycle > 50.0:
                        self.log('1st binary search for n: fall time met, duty cycle > 50, '
                                 'breaking')
                        break
                    self.log('1st binary search for n: fall time met, duty cycle < 50, downsize n')
                    n_bin_iter.down()
            cur_n = n_bin_iter.get_last_save()
            cur_dut = n_bin_iter.get_last_save_info()
            cur_des['seg_dict']['gm_n'] = cur_n
            cur_des['seg_dict']['tail_n'] = 2 * cur_n

        # Step 5: Find p size again since n side may have changed a lot
        if tr > targ_trf:
            p_bin_iter = BinaryIterator(cur_p + 2, 100, 2)
            p_bin_iter.set_current(cur_p)
            while p_bin_iter.has_next():
                cur_p = p_bin_iter.get_next()
                cur_des['seg_dict']['gm_p'] = cur_p
                cur_des['seg_dict']['tail_p'] = 2 * cur_p
                tr, tf, duty_cycle, dut = await self.meas_cur_design(cur_des, pinfo, tbm_specs,
                                                                     target)
                if tr > targ_trf:
                    self.log('2nd binary search for p: rise time too slow, upsize p')
                    p_bin_iter.up()
                else:
                    p_bin_iter.save()
                    p_bin_iter.save_info(dut)
                    if duty_cycle < 50.0:
                        self.log('2nd binary search for p: rise time met, duty cycle < 50, '
                                 'breaking')
                        break
                    self.log('2nd binary search for p: rise time met, duty cycle > 50, downsize p')
                    p_bin_iter.down()
            cur_p = p_bin_iter.get_last_save()
            cur_dut = p_bin_iter.get_last_save_info()
            cur_des['seg_dict']['gm_p'] = cur_p
            cur_des['seg_dict']['tail_p'] = 2 * cur_p

        # Step 6: Fine tune for duty cycle
        num_iter = 0
        while True:
            if np.abs(duty_cycle - 50.0) <= target['duty_cycle_tol']:
                self.log('Design converged.')
                break
            if num_iter >= 10:
                self.log('Duty cycle constraint too tight: not converging ...')
                break
            if duty_cycle < 50.0:
                if max(cur_des['seg_dict']['tail_p'], cur_des['seg_dict']['gm_p']) > 100:
                    self.log('Device too big: not converging ...')
                    break
                self.log('Linear search: duty cycle < 50, upsize p')
                self._modify_nf(cur_des, upsize_p=2)
            else:
                if max(cur_des['seg_dict']['tail_n'], cur_des['seg_dict']['gm_n']) > 100:
                    self.log('Device too big: not converging ...')
                    break
                self.log('Linear search: duty cycle > 50, upsize n')
                self._modify_nf(cur_des, upsize_n=2)
            num_iter += 1
            tr, tf, duty_cycle, cur_dut = await self.meas_cur_design(cur_des, pinfo, tbm_specs,
                                                                     target)

        # Step 7: more fine tuning with total segments fixed to improve duty cycle
        tune_des = deepcopy(cur_des)
        half_seg = min(tune_des['seg_dict']['gm_p'], tune_des['seg_dict']['gm_n'])
        bin_iter = BinaryIterator(- half_seg + 2, half_seg, 2)
        err = np.abs(duty_cycle - 50.0)
        while bin_iter.has_next():
            self.log('--- More fine tuning ---')
            seg_off = bin_iter.get_next()
            self._modify_nf(tune_des, upsize_p=seg_off, upsize_n=-seg_off)
            tr, tf, duty_cycle, dut = await self.meas_cur_design(tune_des, pinfo, tbm_specs, target)
            if np.abs(duty_cycle - 50.0) <= err:
                bin_iter.save()
                bin_iter.save_info(dut)
            if duty_cycle < 50.0:
                bin_iter.up()
            else:
                bin_iter.down()

        cur_dut = bin_iter.get_last_save_info()
        seg_off = bin_iter.get_last_save()
        self._modify_nf(cur_des, upsize_p=seg_off, upsize_n=-seg_off)
        self.log(f'Before signoff, final dut is {cur_dut.cell_name}')
        self.log(f'Final design is {cur_des}')

        # Step 8: measure VTC
        await self.meas_vtc(cur_dut, dc_tbm_specs, target)

        # Step 9: sign-off across corners
        await self.signoff_dut(cur_dut, target, tbm_specs)

        return cur_des
