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


from typing import Optional, Dict, Any, Union, List, cast, Mapping, Tuple, Sequence

import numpy as np
import scipy.optimize as sciopt

from bag.simulation.design import DesignerBase

from bag3_testbenches.measurement.mos.query import MOSDBDiscrete, get_db
from bag3_testbenches.measurement.tran.digital import DigitalTranTB
from bag3_testbenches.measurement.data.tran import EdgeType

from bag3_analog.layout.amplifier.diffamp import DiffAmpSelfBiasedGuardRing

from xbase.layout.mos.top import GenericWrapper


def r_even(num: float) -> int:
    """Round to next positive even integer."""
    num0 = int(np.rint(num))
    return max(num0 + (num0 & 1), 2)


class DiffampSelfBiasedDesigner(DesignerBase):
    @staticmethod
    def design_inv_chain_fo4(vdd: float, c_load: float, mos_db_specs: Dict[str, Any],
                             ) -> Dict[str, Any]:
        """Design inverter chain with 2 inverters and FO4, with common mode of vdd / 2 because
        diffamp will have almost rail-to-rail output"""
        c_in = c_load / 16

        ninv_db: MOSDBDiscrete = get_db('ninv', mos_db_specs)
        pinv_db: MOSDBDiscrete = get_db('pinv', mos_db_specs)

        ninv_op = ninv_db.query(vgs=vdd / 2, vds=vdd / 2)
        pinv_op = pinv_db.query(vgs=- vdd / 2, vds=- vdd / 2)

        ibias_n, cg_n = ninv_op['ibias'], ninv_op['cgg']
        ibias_p, cg_p = pinv_op['ibias'], pinv_op['cgg']

        if ibias_n > ibias_p:
            m = ibias_n / ibias_p
            cg_min = cg_n + m * cg_p
            size_n = c_in / cg_min
            size_p = m * size_n
        else:
            m = ibias_p / ibias_n
            cg_min = m * cg_n + cg_p
            size_p = c_in / cg_min
            size_n = m * size_p

        return dict(
            seg_n=[int(np.rint(size_n)), int(np.rint(size_n * 4))],
            seg_p=[int(np.rint(size_p)), int(np.rint(size_p * 4))],
            c_in=int(np.rint(size_p)) * cg_p + int(np.rint(size_n)) * cg_n,
        )

    def design_diffamp_eqn(self, target: Mapping[str, Any], vdd: float,
                           mos_db_specs: Dict[str, Any],) -> Dict[str, Any]:
        v_cm = target['v_cm']
        v_amp = target['v_amp']
        trf = target['trf']
        c_load = target['c_load']

        nch_db: MOSDBDiscrete = get_db('nch', mos_db_specs)
        pch_db: MOSDBDiscrete = get_db('pch', mos_db_specs)

        # Design so that the self biased diffamp goes into large signal mode of operation
        # with v_amp

        # Helper function to find minimum v_ov at which ibias is greater than ibias_min
        def _find_v_ov(v_test: float, mos_db: MOSDBDiscrete,
                       params: Optional[Dict[str, float]] = None) -> float:
            if params is None:
                mos_op = mos_db.query(vgs=v_test, vds=v_test)
            else:
                if 'vs' not in params:
                    # find vs given vd and vg
                    mos_op = mos_db.query(vgs=params['vg']-v_test, vds=params['vd']-v_test)
                elif 'vd' not in params:
                    # find vd given vs and vg
                    mos_op = mos_db.query(vgs=params['vg']-params['vs'], vds=v_test-params['vs'])
                else:  # 'vg' not in params
                    # find vg given vs and vd
                    mos_op = mos_db.query(vgs=v_test-params['vs'], vds=params['vd']-params['vs'])
            # make sure that ibias is greater than ibias_min_fg = 1uA
            ibias_min = 1.0e-6
            return cast(float, mos_op['ibias'] - ibias_min)

        # min vgs for diode connected pmos
        vgs_min_p = cast(float, sciopt.brentq(_find_v_ov, 0, -vdd, args=(pch_db,)))

        # min vgs for diode connected for nmos
        vgs_min_n = cast(float, sciopt.brentq(_find_v_ov, 0, vdd, args=(nch_db,)))

        # design the inverter chain first
        inv_chain_des = self.design_inv_chain_fo4(vdd, c_load, mos_db_specs)

        # operating point of tail during slewing
        vds_on = 10.0e-3

        # gate is connected to gate of switched off 'diode connected' nmos tail, so vg = vgs_min_n
        ptail_op = pch_db.query(vgs=vgs_min_n - vdd, vds=- vds_on)
        # gate is connected to gate of switched off 'diode connected' pmos tail,
        # so vg = vdd + vgs_min_p
        ntail_op = nch_db.query(vgs=vdd + vgs_min_p, vds=vds_on)

        # operating point of input devices when those are on-switches
        vs_p = vdd - vds_on
        vg_p = v_cm - v_amp
        p_in_op = pch_db.query(vgs=vg_p - vs_p, vds=- vds_on, vbs=vdd - vs_p)

        vs_n = vds_on
        vg_n = v_cm + v_amp
        n_in_op = nch_db.query(vgs=vg_n - vs_n, vds=vds_on, vbs=- vs_n)

        # ensure there is sufficient voltage headroom
        assert vg_n > vds_on + vgs_min_n, f'Increase v_cm={v_cm} V above ' \
                                          f'{vds_on + vgs_min_n - v_amp} V'
        assert vg_p < vdd - vds_on + vgs_min_p, f'Decrease v_cm={v_cm} V below' \
                                                f' {vdd - vds_on + vgs_min_p + v_amp} V'

        # find sizes of clock receiver
        c_inv = inv_chain_des['c_in']
        # a) pmos: switched on
        den_p = p_in_op['ibias'] * trf - p_in_op['cdd'] * vdd
        if den_p < 0:
            self.error('Number of fingers for input pmos is negative.')
        seg_gmp = r_even(c_inv * vdd / den_p)

        # b) nmos: switched on
        den_n = n_in_op['ibias'] * trf - n_in_op['cdd'] * vdd
        if den_n < 0:
            self.error('Number of fingers for input nmos is negative.')
        seg_gmn = r_even(c_inv * vdd / den_n)

        # c) p tail: slewing
        seg_tailp = r_even(seg_gmp * p_in_op['ibias'] / ptail_op['ibias'])

        # d) n tail: slewing
        seg_tailn = r_even(seg_gmn * n_in_op['ibias'] / ntail_op['ibias'])

        seg_dict = dict(
            tail_p=seg_tailp,
            gm_p=seg_gmp,
            gm_n=seg_gmn,
            tail_n=seg_tailn,
            invp=inv_chain_des['seg_p'],
            invn=inv_chain_des['seg_n'],
        )
        self.log('--- Design from equations ---')
        self.log(f'{seg_dict}')

        return dict(
            seg_dict=seg_dict,
            p_in_ibias=p_in_op['ibias'],
            n_in_ibias=n_in_op['ibias'],
            p_tail_ibias=ptail_op['ibias'],
            n_tail_ibias=ntail_op['ibias'],
        )

    @classmethod
    def _upsize_nf(cls, cur_des: Dict[str, Any], upsize_p: bool) -> None:
        seg_dict = cur_des['seg_dict']
        if upsize_p:
            seg_dict['gm_p'] += 2
            seg_dict['tail_p'] = r_even(seg_dict['gm_p'] * cur_des['p_in_ibias'] / cur_des[
                'p_tail_ibias'])
        else:
            seg_dict['gm_n'] += 2
            seg_dict['tail_n'] = r_even(seg_dict['gm_n'] * cur_des['n_in_ibias'] / cur_des[
                'n_tail_ibias'])

    @staticmethod
    def _update_params(pinfo: Mapping[str, Any], seg_dict: Dict[str, Union[int, List[int]]],
                       buf_params: Mapping[str, Any]
                       ) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
        dut_params = dict(
            pinfo=pinfo,
            seg_dict=dict(
                tail_p=seg_dict['tail_p'],
                gm_p=seg_dict['gm_p'],
                gm_n=seg_dict['gm_n'],
                tail_n=seg_dict['tail_n']
            ),
            show_pins=True,
            draw_taps='BOTH',
        )

        for idx, par in enumerate(buf_params['inv_params']):
            par['seg_p'] = seg_dict['invp'][idx]
            par['seg_n'] = seg_dict['invn'][idx]

        return dut_params, buf_params

    @staticmethod
    def _get_wrapper_params(buf_params: Mapping[str, Any]) -> Mapping[str, Any]:
        # dut_lib and dut_cell get added by SimulationDB
        wrapper_params = dict(
            buf_params=buf_params,
        )

        return wrapper_params

    @staticmethod
    def _get_tbm_specs(tbm_specs: Mapping[str, Any], sim_envs: Sequence[str],
                       env_params: Mapping[str, Any], target: Mapping[str, Any]
                       ) -> Mapping[str, Any]:

        return dict(
            dut_pins=['VDD', 'VSS', 'clk', 'clkb', 'clk_out'],
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
                            ),
            pin_values={},
            sim_envs=sim_envs,
            **tbm_specs,
        )

    async def async_design(self, target: Mapping[str, Any], pinfo: Mapping[str, Any],
                           sim_envs: Sequence[str], env_params: Mapping[str, Any],
                           tbm_specs: Mapping[str, Any], buf_params: Mapping[str, Any],
                           mos_db_specs: Dict[str, Any],
                           ) -> Mapping[str, Any]:
        # Step 1: Design based on equations in tt corner
        vdd_tt = env_params['vdd']['tt_25']
        cur_des = self.design_diffamp_eqn(target, vdd_tt, mos_db_specs)
        num_iter = 0

        # Step 2: setup tbm
        tbm_specs = self._get_tbm_specs(tbm_specs, sim_envs, env_params, target)
        tbm = cast(DigitalTranTB, self.make_tbm(DigitalTranTB, tbm_specs))

        while True:
            # Step 3: Update schematic params
            dut_params, buf_params = self._update_params(pinfo, cur_des['seg_dict'], buf_params)
            wrapper_params = self._get_wrapper_params(buf_params)

            gen_params = dict(
                cls_name=DiffAmpSelfBiasedGuardRing.get_qualified_name(),
                params=dut_params,
            )

            dut = await self.async_new_dut(f'dut{num_iter}', GenericWrapper, gen_params)

            tbm_params = dict(
                dut_lib='bag3_analog',
                dut_cell='clk_rx_wrapper',
                dut_params=wrapper_params,
            )

            # Step 4: Simulate and post process
            sim_results = await self.async_simulate_tbm_obj(f'sim_{num_iter}', dut, tbm, tbm_params)

            sim_data = sim_results.data
            tr = tbm.calc_trf(sim_data, 'clk_out', True)
            tf = tbm.calc_trf(sim_data, 'clk_out', False)
            self.log(f'Current design is: {cur_des["seg_dict"]}')
            self.log(f'Rise time: {tr}')
            self.log(f'Fall time: {tf}')
            if tr[0] is np.inf or tf[0] is np.inf:
                self.error('Check simulation setup.')

            tdr = tbm.calc_delay(sim_data, 'clk', 'clk_out', EdgeType.RISE, EdgeType.RISE)
            tdf = tbm.calc_delay(sim_data, 'clk', 'clk_out', EdgeType.FALL, EdgeType.FALL)

            tper = tbm_specs['sim_params']['tper']
            tpw = tper / 2
            tpw_out = tpw + tdf - tdr
            duty_cycle = tpw_out / tper * 100
            self.log(f'Duty cycle: {duty_cycle}')
            duty_cycle_mean = np.mean(duty_cycle)
            if np.abs(duty_cycle_mean - 50.0) <= target['duty_cycle_tol']:
                self.log('Design converged')
                break
            if num_iter >= 10:
                self.log('Duty cycle constraint too tight: not converging ...')
                break
            if duty_cycle_mean < 50.0:
                if max(cur_des['seg_dict']['tail_p'], cur_des['seg_dict']['gm_p']) > 100:
                    self.log('Device too big: not converging ...')
                    break
                self._upsize_nf(cur_des, upsize_p=True)
            else:
                if max(cur_des['seg_dict']['tail_n'], cur_des['seg_dict']['gm_n']) > 100:
                    self.log('Device too big: not converging ...')
                    break
                self._upsize_nf(cur_des, upsize_p=False)
            num_iter += 1

        cur_des['rise_time'] = list(tr)
        cur_des['fall_time'] = list(tf)
        cur_des['duty_cycle'] = list(duty_cycle)

        return cur_des
