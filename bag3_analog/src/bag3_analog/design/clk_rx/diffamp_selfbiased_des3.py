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


from typing import Dict, Any, Union, List, cast, Mapping, Tuple, Sequence, Optional
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from bag.simulation.design import DesignerBase
from bag.util.search import BinaryIterator
from bag.env import get_tech_global_info
from bag.simulation.cache import DesignInstance
from bag.concurrent.util import GatherHelper

from bag3_testbenches.measurement.tran.digital import DigitalTranTB
from bag3_testbenches.measurement.data.tran import EdgeType, get_first_crossings
from bag3_testbenches.measurement.dc.base import DCTB

from bag3_analog.layout.amplifier.diffamp import DiffAmpSelfBiasedBufferGuardRing

from xbase.layout.mos.top import GenericWrapper


class DiffampSelfBiasedDesigner(DesignerBase):
    def __init__(self, *args, **kwargs):
        DesignerBase.__init__(self, *args, **kwargs)
        self._n_lo = 2
        self._n_hi = 50
        self._p_lo = 2
        self._p_hi = 50

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

    async def meas_duty_cycle(self, cur_des: Dict[str, Any], pinfo: Mapping[str, Any],
                              tbm_specs: Mapping[str, Any], target: Mapping[str, Any]
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DesignInstance]:
        # Create DUT
        dut = await self._make_dut(cur_des, pinfo)

        # setup simulation across corners
        tech_info = get_tech_global_info('aib_ams')
        all_corners = tech_info['signoff_envs']['all_corners']

        helper = GatherHelper()
        corner_names = all_corners['envs']
        for corner in corner_names:
            vdd = all_corners['vdd'][corner]
            helper.append(self._meas_corner(dut, target, tbm_specs, corner, vdd, 'duty_cycle'))

        results = await helper.gather_err()
        num = len(corner_names)
        duty_cycle_arr = np.zeros(num)
        tr_arr = np.zeros(num)
        tf_arr = np.zeros(num)

        for idx, name in enumerate(corner_names):
            tr, tf, duty_cycle, _ = results[idx]
            tr_arr[idx] = tr
            tf_arr[idx] = tf
            duty_cycle_arr[idx] = duty_cycle
        return tr_arr, tf_arr, duty_cycle_arr, dut

    async def meas_cur_dut(self, dut: DesignInstance, tbm: DigitalTranTB, name: str,
                           sim_type: str,) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                    Dict[str, Any]]:
        """This function is called from _meas_corner(), so post processing here is per corner"""
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
            tro = tbm.calc_trf(sim_data, 'clk_out', True)
            tfo = tbm.calc_trf(sim_data, 'clk_out', False)
            self.log(f'Current design is: {name}_{dut.cell_name}')
            self.log(f'Rise time: {tr.item()}')
            self.log(f'Fall time: {tf.item()}')
            self.log(f'Rise time out: {tro.item()}')
            self.log(f'Fall time out: {tfo.item()}')
            if tr[0] is np.inf or tf[0] is np.inf:
                self.error('Check simulation setup.')

            tdr = tbm.calc_delay(sim_data, 'clk', 'clk_out', EdgeType.RISE, EdgeType.RISE)
            tdf = tbm.calc_delay(sim_data, 'clk', 'clk_out', EdgeType.FALL, EdgeType.FALL)

            tper = tbm.specs['sim_params']['tper']
            tpw = tper / 2
            tpw_out = tpw + tdf - tdr
            duty_cycle = tpw_out / tper * 100
            self.log(f'Duty cycle: {duty_cycle}')

            return tr, tf, duty_cycle, dict(
                time=sim_data['time'],
                v_mid=sim_data['v_mid'][0],
                clk_out=sim_data['clk_out'][0],
            )
        elif sim_type == 'dc':
            v_in = sim_data['clk'][0]
            v_inb = sim_data['clkb'][0]
            v_out = sim_data['v_mid'][0]

            # there should be no hysteresis, so ignore half of the waveforms
            num_points = len(v_in)
            v_in2 = v_in[:num_points // 2]
            v_out2 = v_out[:num_points // 2]
            slope = np.diff(v_out2) / np.diff(v_in2)
            low_vin = get_first_crossings(v_in2[:-1], slope, 1.0, etype=EdgeType.RISE)
            high_vin = get_first_crossings(v_in2[:-1], slope, 1.0, etype=EdgeType.FALL)
            if np.isinf(low_vin) and np.isinf(high_vin):
                v_cm = tbm.specs['sup_values']['VCM']
                if v_in2[0] < v_cm:
                    margin = dict(
                        low=-1,
                        high=v_in2[-1] - v_in2[0],
                        mid=v_in2[0],
                        window=-1,
                    )
                else:
                    margin = dict(
                        low=v_in2[-1] - v_in2[0],
                        high=-1,
                        mid=v_in2[-1],
                        window=-1,
                    )
            elif np.isinf(low_vin):
                margin = dict(
                    low=-1,
                    high=v_in2[-1] - high_vin,
                    mid=v_in2[0],
                    window=high_vin - v_in2[0],
                )
            elif np.isinf(high_vin):
                margin = dict(
                    low=low_vin - v_in2[0],
                    high=-1,
                    mid=v_in2[-1],
                    window=v_in2[-1] - low_vin,
                )
            else:
                if high_vin < low_vin:
                    # TODO: hack
                    high_vin = get_first_crossings(v_in2[:-1], slope, 1.0, etype=EdgeType.FALL,
                                                   start=low_vin)

                margin = dict(
                    low=low_vin - v_in2[0],
                    high=v_in2[-1] - high_vin,
                    mid=0.5 * (low_vin + high_vin),
                    window=high_vin - low_vin,
                )

            return v_in, v_inb, v_out, margin
        else:
            self.error(f'Unknown sim_type = {sim_type}.')

    async def _make_dut(self, cur_des: Dict[str, Any], pinfo: Mapping[str, Any]) -> DesignInstance:
        dut_params = self._update_params(pinfo, cur_des['seg_dict'])

        gen_params = dict(
            cls_name=DiffAmpSelfBiasedBufferGuardRing.get_qualified_name(),
            params=dut_params,
        )

        seg_dict = cur_des['seg_dict']
        dsn_num = f'{seg_dict["tail_n"]}_{seg_dict["gm_n"]}_{seg_dict["gm_p"]}_{seg_dict["tail_p"]}'
        dut = await self.async_new_dut(f'dut{dsn_num}', GenericWrapper, gen_params)
        self.log(f'DUT name is dut{dsn_num}')
        return dut

    async def meas_vtc(self, cur_des: Dict[str, Any], pinfo: Mapping[str, Any],
                       dc_tbm_specs: Mapping[str, Any], target: Mapping[str, Any],
                       plot_flag: bool = False) -> Tuple[Dict[str, Any], DesignInstance]:
        # create dut
        dut = await self._make_dut(cur_des, pinfo)

        # setup simulation across corners
        tech_info = get_tech_global_info('aib_ams')
        all_corners = tech_info['signoff_envs']['all_corners']

        helper = GatherHelper()
        corner_names = all_corners['envs']
        for corner in corner_names:
            vdd = all_corners['vdd'][corner]
            helper.append(self._meas_corner(dut, target, dc_tbm_specs, corner, vdd, 'vtc'))

        results = await helper.gather_err()

        num = len(corner_names)
        vtc_dict = dict(
            sim_envs=np.chararray(num, itemsize=10),
            window=np.zeros(num),
            low=np.zeros(num),
            high=np.zeros(num),
            mid=np.zeros(num),
        )
        for idx, name in enumerate(corner_names):
            v_in, v_inb, v_out, margin = results[idx]
            vtc_dict['sim_envs'][idx] = name
            vtc_dict['low'][idx] = margin['low']
            vtc_dict['high'][idx] = margin['high']
            vtc_dict['mid'][idx] = margin['mid']
            vtc_dict['window'][idx] = margin['window']

            if plot_flag:
                plt.subplot(num // 2 + num % 2, 2, idx + 1)
                plt.plot(v_in, v_out, label='v_mid')
                # plt.plot(v_in[:-1], np.diff(v_out) / np.diff(v_in), label='slope_clk_out')
                plt.plot(v_in, v_inb, label='clk_inb')
                plt.xlabel('clk_in (in V)')
                plt.ylabel('Magnitude (in V)')
                plt.title(f'Corner: {name}')
                plt.legend()
        if plot_flag:
            plt.tight_layout()
            plt.show()
        return vtc_dict, dut

    async def signoff_dut(self, dut: DesignInstance, target: Mapping[str, Any],
                          tbm_specs: Mapping[str, Any], dc_tbm_specs: Mapping[str, Any]
                          ) -> None:
        self.log('--- Beginning sign off across corners ---')
        tech_info = get_tech_global_info('aib_ams')
        all_corners = tech_info['signoff_envs']['all_corners']

        # Signoff VTC
        helper = GatherHelper()
        for corner in all_corners['envs']:
            vdd = all_corners['vdd'][corner]
            helper.append(self._meas_corner(dut, target, dc_tbm_specs, corner, vdd, 'vtc'))

        results = await helper.gather_err()
        num = len(all_corners['envs'])
        plt.figure()
        for idx, corner in enumerate(all_corners['envs']):
            v_in, v_inb, v_out, margin = results[idx]
            if margin['low'] == -1 or margin['high'] == -1:
                self.error(f'VTC signoff failed at corner={corner}.')
            plt.subplot(num // 2 + num % 2, 2, idx + 1)
            plt.plot(v_in, v_out, label='v_mid')
            plt.plot(v_in, v_inb, label='clk_inb')
            plt.xlabel('clk_in (in V)')
            plt.ylabel('Magnitude (in V)')
            plt.title(f'Corner: {corner}')
            plt.legend()
        plt.tight_layout()
        plt.show()

        # Signoff duty cycle
        helper = GatherHelper()
        for corner in all_corners['envs']:
            vdd = all_corners['vdd'][corner]
            helper.append(self._meas_corner(dut, target, tbm_specs, corner, vdd, 'duty_cycle'))

        results = await helper.gather_err()
        plt.figure()
        for idx, corner in enumerate(all_corners['envs']):
            tr, tf, duty_cycle, signals = results[idx]
            if tr < 0 or tf < 0:
                self.error(f'Negative rise/fall time at corner={corner}.')
            plt.subplot(num // 2 + num % 2, 2, idx + 1)
            plt.plot(signals['time'], signals['clk_out'], label='clk_out')
            plt.plot(signals['time'], signals['v_mid'], label='v_mid')
            plt.xlabel('time (in sec)')
            plt.ylabel('Magnitude (in V)')
            plt.title(f'Corner: {corner}, Duty cycle: {duty_cycle[0]:0.5g}')
            plt.legend()
        plt.tight_layout()
        plt.show()

        self.log('All corners signed off.')

    async def _meas_corner(self, dut: DesignInstance, target: Mapping[str, Any],
                           tbm_specs: Mapping[str, Any], corner: Union[str, List[str]], vdd: float,
                           mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
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

    async def size_for_vtc(self, cur_des: Dict[str, Any], pinfo: Mapping[str, Any],
                           dc_tbm_specs: Mapping[str, Any], target: Mapping[str, Any],
                           mode: str, bin_iter: Optional[BinaryIterator] = None) -> DesignInstance:

        iter_des = deepcopy(cur_des)
        if bin_iter is None:
            bin_iter = BinaryIterator(2, 3, 2)
        best_error = np.inf

        while bin_iter.has_next():
            seg = bin_iter.get_next()
            if mode == 'main':
                pass
            elif mode == 'n_bin':
                iter_des['seg_dict']['gm_n'] = seg
                iter_des['seg_dict']['tail_n'] = 2 * seg
            elif mode == 'p_bin':
                iter_des['seg_dict']['gm_p'] = seg
                iter_des['seg_dict']['tail_p'] = 2 * seg
            else:
                self.error(f'Unknown mode = {mode}.')
            vtc_dict, cur_dut = await self.meas_vtc(iter_des, pinfo, dc_tbm_specs, target)
            min_high = np.min(vtc_dict['high'])
            min_low = np.min(vtc_dict['low'])
            cur_error = min_high - min_low
            if -1 in vtc_dict['low'] and -1 in vtc_dict['high']:
                # case 00: one corner has transition on leftmost edge and other corner has
                #          transition on rightmost edge
                low_idx = np.where(vtc_dict['low'] == -1)[0]
                high_idx = np.where(vtc_dict['high'] == -1)[0]
                self.error(f'VTC has transition at left edge of input range for corner '
                           f'{vtc_dict["sim_envs"][low_idx]}; and right edge of input range for '
                           f'corner {vtc_dict["sim_envs"][high_idx]}.')
            elif -1 in vtc_dict['low'] or cur_error > 0:
                # case 01: one corner has transition on leftmost edge
                low_idx_arr = np.where(vtc_dict['low'] == -1)[0]
                self.log('-'*80)
                if -1 in vtc_dict['low']:
                    self.log(f'VTC has transition at left edge of input range for corner '
                             f'{vtc_dict["sim_envs"][low_idx_arr]};')
                    if mode != 'p_bin':
                        self._n_lo = max(self._n_lo, iter_des['seg_dict']['gm_n'])
                else:
                    if np.abs(cur_error) < best_error:
                        bin_iter.save_info(cur_dut)
                        cur_des['seg_dict'] = deepcopy(iter_des['seg_dict'])
                        best_error = np.abs(cur_error)
                    self.log('All corners have proper VTC transitions, but shifted to left - '
                             'trying to center.')
                if mode == 'main':
                    n_bin_iter = BinaryIterator(4, 100, 2)
                    cur_dut = await self.size_for_vtc(iter_des, pinfo, dc_tbm_specs, target,
                                                      mode='n_bin', bin_iter=n_bin_iter)
                    bin_iter.save_info(cur_dut)
                    cur_des['seg_dict'] = deepcopy(iter_des['seg_dict'])
                    bin_iter.down()
                    self._p_hi = 10
                elif mode == 'n_bin':
                    bin_iter.up()
                elif mode == 'p_bin':
                    bin_iter.down()
                else:
                    self.error(f'Unknown mode = {mode}.')
            elif -1 in vtc_dict['high'] or cur_error < 0:
                # case 02: one corner has transition on rightmost edge
                high_idx_arr = np.where(vtc_dict['high'] == -1)[0]
                self.log('-' * 80)
                if -1 in vtc_dict['high']:
                    self.log(f'VTC has transition at right edge of input range for corner '
                             f'{vtc_dict["sim_envs"][high_idx_arr]};')
                    if mode != 'n_bin':
                        self._p_lo = max(self._p_lo, iter_des['seg_dict']['gm_p'])
                else:
                    if np.abs(cur_error) < best_error:
                        bin_iter.save_info(cur_dut)
                        cur_des['seg_dict'] = deepcopy(iter_des['seg_dict'])
                        best_error = np.abs(cur_error)
                    self.log('All corners have proper VTC transitions, but shifted to right - '
                             'trying to center')
                if mode == 'main':
                    p_bin_iter = BinaryIterator(4, 100, 2)
                    cur_dut = await self.size_for_vtc(iter_des, pinfo, dc_tbm_specs, target,
                                                      mode='p_bin', bin_iter=p_bin_iter)
                    bin_iter.save_info(cur_dut)
                    cur_des['seg_dict'] = deepcopy(iter_des['seg_dict'])
                    bin_iter.down()
                    self._n_hi = 10
                elif mode == 'n_bin':
                    bin_iter.down()
                elif mode == 'p_bin':
                    bin_iter.up()
                else:
                    self.error(f'Unknown mode = {mode}.')
            else:
                # case 03: all corners have proper transitions
                self.log('-' * 80)
                self.log('All corners have proper VTC transitions.')
                bin_iter.save_info(cur_dut)
                cur_des['seg_dict'] = deepcopy(iter_des['seg_dict'])

                bin_iter.down()

        self.log('-' * 80)
        self.log(f'{mode}')
        self.log('Done with VTC sizing')
        if mode == 'main':
            await self.meas_vtc(cur_des, pinfo, dc_tbm_specs, target, plot_flag=True)
        return bin_iter.get_last_save_info()

    async def _duty_cycle_size_helper(self, cur_des: Dict[str, Any], pinfo: Mapping[str, Any],
                                      tbm_specs: Mapping[str, Any],
                                      dc_tbm_specs: Mapping[str, Any], target: Mapping[str, Any],
                                      bin_iter: BinaryIterator, is_pch: bool
                                      ) -> Tuple[DesignInstance, float]:
        iter_des = deepcopy(cur_des)
        best_error = np.inf
        while bin_iter.has_next():
            self.log(f'is_pch = {is_pch}')
            seg = bin_iter.get_next()
            if is_pch:
                iter_des['seg_dict']['gm_p'] = seg
                iter_des['seg_dict']['tail_p'] = 2 * seg
            else:
                iter_des['seg_dict']['gm_n'] = seg
                iter_des['seg_dict']['tail_n'] = 2 * seg
            vtc_dict, _ = await self.meas_vtc(iter_des, pinfo, dc_tbm_specs, target)
            if -1 in vtc_dict['low']:
                self.log('VTC shifted too much to left')
                bin_iter.down() if is_pch else bin_iter.up()
            elif -1 in vtc_dict['high']:
                self.log('VTC shifted too much to right')
                bin_iter.up() if is_pch else bin_iter.down()
            else:
                # VTC ok, check duty cycle error
                tr_arr, tf_arr, duty_cycle_arr, dut = await self.meas_duty_cycle(iter_des, pinfo,
                                                                                 tbm_specs, target)
                num_corners = len(duty_cycle_arr)
                cur_error = (np.max(duty_cycle_arr) + np.min(duty_cycle_arr)) / 2 - 50.0
                if np.abs(cur_error) < best_error:
                    best_error = np.abs(cur_error)
                    bin_iter.save_info(dut)
                    cur_des['seg_dict'] = deepcopy(iter_des['seg_dict'])
                if cur_error > 0:
                    bin_iter.down() if is_pch else bin_iter.up()
                else:
                    bin_iter.up() if is_pch else bin_iter.down()
        self.log('-' * 80)
        self.log(f'Finished binary iteration for is_pch = {is_pch}.')
        return bin_iter.get_last_save_info(), best_error

    async def size_for_duty_cycle(self, cur_des: Dict[str, Any], pinfo: Mapping[str, Any],
                                  tbm_specs: Mapping[str, Any], dc_tbm_specs: Mapping[str, Any],
                                  target: Mapping[str, Any]) -> DesignInstance:
        # initial measurement
        tr_arr, tf_arr, duty_cycle_arr, dut = await self.meas_duty_cycle(cur_des, pinfo, tbm_specs,
                                                                         target)
        num_corners = len(duty_cycle_arr)
        best_error = (np.max(duty_cycle_arr) + np.min(duty_cycle_arr)) / 2 - 50.0

        iter_des1 = deepcopy(cur_des)
        iter_des2 = deepcopy(cur_des)
        if best_error > 0:
            # 1. upsize n
            up_bin_iter = BinaryIterator(cur_des['seg_dict']['gm_n'] + 2, self._n_hi, 2)
            up_dut, up_error = await self._duty_cycle_size_helper(iter_des1, pinfo, tbm_specs,
                                                                  dc_tbm_specs, target,
                                                                  up_bin_iter, is_pch=False)
            # 2. downsize p
            dn_bin_iter = BinaryIterator(self._p_lo, cur_des['seg_dict']['gm_p'] - 2, 2)
            dn_dut, dn_error = await self._duty_cycle_size_helper(iter_des2, pinfo, tbm_specs,
                                                                  dc_tbm_specs, target,
                                                                  dn_bin_iter, is_pch=True)
        else:
            # 1. upsize p
            up_bin_iter = BinaryIterator(cur_des['seg_dict']['gm_p'] + 2, self._p_hi, 2)
            up_dut, up_error = await self._duty_cycle_size_helper(iter_des1, pinfo, tbm_specs,
                                                                  dc_tbm_specs, target,
                                                                  up_bin_iter, is_pch=True)
            # 2. downsize n
            dn_bin_iter = BinaryIterator(self._n_lo, cur_des['seg_dict']['gm_n'] - 2, 2)
            dn_dut, dn_error = await self._duty_cycle_size_helper(iter_des2, pinfo, tbm_specs,
                                                                  dc_tbm_specs, target,
                                                                  dn_bin_iter, is_pch=False)
        if up_error < best_error:
            dut = up_dut
            cur_des['seg_dict'] = deepcopy(iter_des1['seg_dict'])
        if dn_error < best_error:
            dut = dn_dut
            cur_des['seg_dict'] = deepcopy(iter_des2['seg_dict'])
        return dut

    async def async_design(self, target: Mapping[str, Any], pinfo: Mapping[str, Any],
                           tbm_specs: Mapping[str, Any], dc_tbm_specs: Mapping[str, Any],
                           ) -> Mapping[str, Any]:
        # Step 1: Get initial design in tt corner
        cur_des = self.design_diffamp_eqn(target)

        # Step 2: size for VTC
        _ = await self.size_for_vtc(cur_des, pinfo, dc_tbm_specs, target, 'main')

        # Step 3: size for duty cycle
        cur_dut = await self.size_for_duty_cycle(cur_des, pinfo, tbm_specs, dc_tbm_specs, target)

        # Step 4: sign-off
        # cur_des['seg_dict']['gm_p'] = 14
        # cur_des['seg_dict']['tail_p'] = 28
        # cur_dut = await self._make_dut(cur_des, pinfo)
        await self.signoff_dut(cur_dut, target, tbm_specs, dc_tbm_specs)
        return cur_des
