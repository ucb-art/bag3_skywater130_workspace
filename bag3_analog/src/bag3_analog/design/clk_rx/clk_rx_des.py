"""This package contains design class for clock receiver"""

from typing import Optional, Dict, Any, Union, List, cast
from copy import deepcopy
from pprint import pprint as print
from pathlib import Path

import numpy as np
import scipy.optimize as sciopt

from bag.core import BagProject
from bag.util.immutable import Param
from bag.simulation.hdf5 import load_sim_data_hdf5

from bag3_testbenches.measurement.mos.query import MOSDBDiscrete, get_db

from ...measurement.clk_rx.helper import get_self_bias_inv, get_self_bias_clk_rx
from ...measurement.clk_rx.timing import ClkRXTimingTB


def r_even(num: float) -> int:
    num0 = int(np.rint(num))
    return max(num0 + (num0 & 1), 2)


class ClkRXDesign:
    def __init__(self, specs: Dict[str, Any]) -> None:
        self._specs: Dict[str, Any] = specs
        self._params: Param = specs['params']
        self._lay_class: str = specs.get('lay_class', '')
        if self._lay_class:
            self._lch: int = self._params['params']['pinfo']['tile_specs']['arr_info']['lch']
        else:
            self._lch: int = self._params['lch']
        self._wrapper_params: Param = specs['wrapper_params']
        self._sim_params: Dict[str, Any] = specs['tbm_specs']['sim_params']
        self._target_specs: Dict[str, Any] = specs['target_specs']
        self._sim_envs: List[str] = specs['tbm_specs']['sim_envs']
        self.nch_db: MOSDBDiscrete = get_db('nch', specs)
        self.ninv_db: MOSDBDiscrete = get_db('ninv', specs)
        self.pch_db: MOSDBDiscrete = get_db('pch', specs)
        self.pinv_db: MOSDBDiscrete = get_db('pinv', specs)

    @property
    def specs(self) -> Dict[str, Any]:
        return self._specs

    @property
    def params(self) -> Param:
        return self._params

    @property
    def wrapper_params(self) -> Param:
        return self._wrapper_params

    @property
    def sim_params(self) -> Dict[str, Any]:
        return self._sim_params

    @property
    def target_specs(self) -> Dict[str, Any]:
        return self._target_specs

    @property
    def sim_envs(self) -> List[str]:
        return self._sim_envs

    @property
    def lch(self) -> int:
        return self._lch

    def test_clk_rx(self) -> None:
        vdd = self.sim_params['vdd']
        seg_dict = self.params['seg_dict']

        seg_invp: List[int] = seg_dict['invp']
        seg_invn: List[int] = seg_dict['invn']

        inv_dict = get_self_bias_inv(self.ninv_db, self.pinv_db,
                                     dict(p=seg_invp[0], n=seg_invn[0]), vdd)

        clk_rx_dict = get_self_bias_clk_rx(self.nch_db, self.pch_db, seg_dict, vdd)

        print('---Inverter---')
        print(inv_dict)

        print('---Clk_RX---')
        print(clk_rx_dict)

    def design_inv_chain(self) -> Dict[str, List[int]]:
        vdd = self.sim_params['vdd']
        c_load = self.target_specs['c_load']

        # Design to have 2 inverters in the chain with fan-out of 4. Also, design for common mode
        # of Vdd / 2 because self biased diffamp will have almost rail-to-rail output
        c_in = c_load / 16

        ninv_op = self.ninv_db.query(vgs=vdd / 2, vds=vdd / 2)
        pinv_op = self.pinv_db.query(vgs=- vdd / 2, vds=- vdd / 2)

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

    def design_clk_rx(self) -> Dict[str, Union[int, List[int]]]:
        vdd = self.sim_params['vdd']
        v_cm = self.target_specs['v_cm']
        v_amp = self.target_specs['v_amp']
        tr = self.target_specs['tr']

        # Design so that the self biased diffamp goes into large signal mode of operation
        # with v_amp

        # Helper function to find minimum v_ov at which ibias is greater than ibias_min
        def _find_v_ov(v_test: float, mos_db: MOSDBDiscrete, vgs: Optional[float] = None) -> float:
            if vgs is None:
                mos_op = mos_db.query(vgs=v_test, vds=v_test)
            else:
                mos_op = mos_db.query(vgs=vgs, vds=v_test)
            # make sure that ibias is greater than ibias_min_fg = 1uA
            ibias_min = 1.0e-6
            return cast(float, mos_op['ibias'] - ibias_min)

        # min vgs for diode connected pmos
        vgs_min_p = cast(float, sciopt.brentq(_find_v_ov, 0, -vdd, args=(self.pch_db,)))

        # min vgs for diode connected for nmos
        vgs_min_n = cast(float, sciopt.brentq(_find_v_ov, 0, vdd, args=(self.nch_db,)))

        inv_chain_des = self.design_inv_chain()  # design the inverter chain first

        # operating point of tail during slewing
        vds_on = 10.0e-3

        # gate is connected to gate of switched off 'diode connected' nmos tail, so vg = vgs_min_n
        ptail_op = self.pch_db.query(vgs=vgs_min_n - vdd, vds=- vds_on)
        # gate is connected to gate of switched off 'diode connected' pmos tail,
        # so vg = vdd - vgs_min_p
        ntail_op = self.nch_db.query(vgs=vdd - vgs_min_p, vds=vds_on)

        # operating point of input devices when those are on-switches
        vs_p = vdd - vds_on
        vg_p = v_cm - v_amp
        p_in_op = self.pch_db.query(vgs=vg_p - vs_p, vds=v_cm - vs_p, vbs=vdd - vs_p)

        vs_n = vds_on
        vg_n = v_cm + v_amp
        n_in_op = self.nch_db.query(vgs=vg_n - vs_n, vds=v_cm - vs_n, vbs=- vs_n)

        # ensure there is sufficient voltage headroom
        assert vg_n > vds_on + vgs_min_n, f'Increase v_cm={v_cm} V above ' \
                                          f'{vds_on + vgs_min_n - v_amp} V'
        assert vg_p < vdd - vds_on + vgs_min_p, f'Decrease v_cm={v_cm} V below' \
                                                f' {vdd - vds_on + vgs_min_p + v_amp} V'

        # find sizes of clock receiver
        c_inv = inv_chain_des['c_in']
        # a) pmos: switched on
        seg_gmp = r_even(c_inv * vdd / (p_in_op['ibias'] * tr - p_in_op['cdd'] * vdd))

        # b) nmos: switched on
        seg_gmn = r_even(c_inv * vdd / (n_in_op['ibias'] * tr - n_in_op['cdd'] * vdd))

        # c) p tail: slewing
        seg_tailp = r_even(seg_gmp * p_in_op['ibias'] / ptail_op['ibias'])

        # d) n tail: slewing
        seg_tailn = r_even(seg_gmn * n_in_op['ibias'] / ntail_op['ibias'])

        return dict(
            tailp=seg_tailp,
            gmp=seg_gmp,
            gmn=seg_gmn,
            tailn=seg_tailn,
            invp=inv_chain_des['seg_p'],
            invn=inv_chain_des['seg_n'],
            p_in_ibias=p_in_op['ibias'],
            n_in_ibias=n_in_op['ibias'],
            p_tail_ibias=ptail_op['ibias'],
            n_tail_ibias=ntail_op['ibias'],
        )

    def _update_params(self, seg_dict: Dict[str, Union[int, List[int]]]) -> None:
        # Update Clock receiver segments dictionary
        self.params['seg_dict'] = dict(
            tailp=seg_dict['tailp'],
            gmp=seg_dict['gmp'],
            gmn=seg_dict['gmn'],
            tailn=seg_dict['tailn']
        )

        # Update inverter chain segments
        inv_params = self.wrapper_params['buf_params']['inv_params']
        for idx, par in enumerate(inv_params):
            par['seg'] = (seg_dict['invp'][idx] + seg_dict['invn'][idx]) // 2

        # mismatch parameters
        Avt_n = self.specs['Avt_n']
        Avt_p = self.specs['Avt_p']
        lch = self.lch
        w = 4  # assume 4 fins are used in every row
        if self._lay_class:
            self.sim_params['v__mGMn_right_MM0'] = -3 * Avt_n / np.sqrt(lch * w * seg_dict['gmn'])
            self.sim_params['v__mGMp_right_MM0'] = -3 * Avt_p / np.sqrt(lch * w * seg_dict['gmp'])
            self.sim_params['v__mTailn_MM0'] = -3 * Avt_n / np.sqrt(lch * w * seg_dict['tailn'] * 2)
            self.sim_params['v__mTailp_MM0'] = -3 * Avt_p / np.sqrt(lch * w * seg_dict['tailp'] * 2)
        else:
            self.sim_params['v__XGMn_right'] = -3 * Avt_n / np.sqrt(lch * w * seg_dict['gmn'])
            self.sim_params['v__XGMp_right'] = -3 * Avt_p / np.sqrt(lch * w * seg_dict['gmp'])
            self.sim_params['v__XTailn'] = -3 * Avt_n / np.sqrt(lch * w * seg_dict['tailn'] * 2)
            self.sim_params['v__XTailp'] = -3 * Avt_p / np.sqrt(lch * w * seg_dict['tailp'] * 2)

    @classmethod
    def _update_design(cls, seg_dict: Dict[str, Union[int, List[int]]], upsize_p: bool) -> None:
        if upsize_p:
            seg_dict['gmp'] += 2
            seg_dict['tailp'] = r_even(seg_dict['gmp'] * seg_dict['p_in_ibias'] / seg_dict[
                'p_tail_ibias'])
        else:
            seg_dict['gmn'] += 2
            seg_dict['tailn'] = r_even(seg_dict['gmn'] * seg_dict['n_in_ibias'] / seg_dict[
                'n_tail_ibias'])

    def design_clk_rx_closed_loop(self, bprj: BagProject) -> Dict[str, Union[int, List[int]]]:
        # Step 1: Design based on equations in tt corner
        des_dict = self.design_clk_rx()
        num_iter = 0

        tbm_specs = self.specs['tbm_specs']

        while True:
            # Step 2: Update schematic params
            self._update_params(des_dict)

            # Step 3: Simulate
            hdf5_file = bprj.simulate_cell(self.specs, extract=True, gen_tb=True, simulate=True,
                                           mismatch=True, raw=False)
            sim_data = load_sim_data_hdf5(Path(hdf5_file).resolve())

            print('--- ---')
            rise_time, fall_time = ClkRXTimingTB.get_output_trf(sim_data, tbm_specs, 'clk_out')
            duty_cycle_list = ClkRXTimingTB.get_output_duty_cycle(sim_data, tbm_specs, 'clk',
                                                                  'clk_out')
            print('Current design:')
            print(des_dict)
            print('rise_time:')
            print(rise_time)
            print('fall_time:')
            print(fall_time)
            print('duty_cycle:')
            print(duty_cycle_list)
            duty_cycle_mean = np.mean(duty_cycle_list)
            if np.abs(duty_cycle_mean - 50.0) <= self.target_specs['duty_cycle_tol']:
                print('Design converged')
                break
            if num_iter >= 10:
                print('Duty cycle constraint too tight: not converging ...')
                break
            if des_dict['tailp'] > 100 or des_dict['tailn'] > 100:
                print(f'Device too big: not converging ...')
                break
            if duty_cycle_mean < 50.0:
                self._update_design(des_dict, upsize_p=True)
            else:
                self._update_design(des_dict, upsize_p=False)
            num_iter += 1

        des_dict['rise_time'] = list(rise_time)
        des_dict['fall_time'] = list(fall_time)
        des_dict['duty_cycle'] = list(duty_cycle_list)

        return des_dict
