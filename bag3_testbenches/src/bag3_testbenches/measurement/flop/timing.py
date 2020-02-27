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

"""This package contains flop timing measurement classes."""

from typing import Tuple, Dict, Any, Sequence, Optional, Union, List, cast, Type, Mapping

from pathlib import Path

import numpy as np

from bag.design.database import ModuleDB
from bag.design.module import Module
from bag.io.file import open_file
from bag.util.search import FloatBinaryIterator
from bag.simulation.data import (
    SimData, SimNetlistInfo, netlist_info_from_dict, AnalysisType, swp_info_from_struct
)
from bag.simulation.base import SimAccess
from bag.simulation.core import TestbenchManager, MeasurementManager

from ..data.tran import EdgeType, bits_to_pwl_iter, get_first_crossings


class FlopTimingTB(TestbenchManager):
    """This class sets up the setup/hold time measurement testbench.

    Assumptions:

    1. tper is not swept.
    2. tper is long enough so that tsetup = tper/2, thold = tper/4 is sufficient.
    """

    def __init__(self, sim: SimAccess, work_dir: Path, tb_name: str, impl_lib: str,
                 specs: Dict[str, Any], sim_view_list: Sequence[Tuple[str, str]],
                 env_list: Sequence[str], precision: int = 6) -> None:
        TestbenchManager.__init__(self, sim, work_dir, tb_name, impl_lib, specs, sim_view_list,
                                  env_list, precision=precision)
        self._nper = 5  # reset, rise, reset, fall, reset

    @classmethod
    def get_schematic_class(cls) -> Type[Module]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_testbenches', 'digital_tb_tran')

    def pre_setup(self, sch_params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        specs = self.specs
        thres_lo: float = specs['thres_lo']
        thres_hi: float = specs['thres_hi']

        # generate PWL waveform files
        clk_data = ['0', 'vdd'] * self._nper
        clk_data.append('0')
        trf_scale = f'{thres_hi - thres_lo:.4g}'
        clk_path = self.work_dir / 'flop_clk.txt'
        with open_file(clk_path, 'w') as f:
            for _, s_tb, s_tr, val in bits_to_pwl_iter(clk_data):
                f.write(f'tper*{s_tb}/2+clk_trf*({s_tr})/{trf_scale} {val}\n')

        vin = ['0', '0', 'vdd', 'vdd', '0', '0', 'vdd', 'vdd', '0', '0', 'vdd', 'vdd', '0']
        in_scale = f'{2 * (thres_hi - thres_lo):.4g}'
        tin = ['0',
               f'tper*3/2+tdr-trf/{in_scale}',
               f'tper*3/2+tdr+trf/{in_scale}',
               f'tper*3/2+thr-trf/{in_scale}',
               f'tper*3/2+thr+trf/{in_scale}',
               f'tper*2-trf/{in_scale}',
               f'tper*2+trf/{in_scale}',
               f'tper*7/2+tdf-trf/{in_scale}',
               f'tper*7/2+tdf+trf/{in_scale}',
               f'tper*7/2+thf-trf/{in_scale}',
               f'tper*7/2+thf+trf/{in_scale}',
               f'tper*4-trf/{in_scale}',
               f'tper*4+trf/{in_scale}',
               ]
        in_path = self.work_dir / 'flop_in.txt'
        with open_file(in_path, 'w') as f:
            for t, v in zip(tin, vin):
                f.write(f'{t} {v}\n')

        ans = sch_params.copy()
        ans['in_file_list'] = [('in', str(in_path.resolve()))]
        ans['clk_file_list'] = [('clk', str(clk_path.resolve()))]
        return ans

    def get_netlist_info(self) -> SimNetlistInfo:
        specs = self.specs
        tstep: Optional[float] = specs.get('tstep', None)
        swp_info: Union[List[Tuple[str, Dict[str, Any]]], Dict[str, Any]] = specs['swp_info']
        monte_carlo_params: Dict[str, Any] = specs.get('monte_carlo_params', {})
        sim_params: Dict[str, float] = specs['sim_params']
        tper: float = sim_params['tper']
        sim_options: Mapping[str, Any] = specs.get('sim_options', {})
        save_outputs: List[str] = specs.get('save_outputs', [])

        tper2 = tper / 2
        tsim = self._nper * tper + tper2

        sim_setup = dict(
            sim_envs=self.sim_envs,
            analyses=[dict(type='TRAN',
                           start=0.0,
                           stop=tsim,
                           options={},
                           save_outputs=save_outputs,
                           )
                      ],
            params=sim_params,
            swp_info=swp_info,
            options=sim_options,
            monte_carlo=monte_carlo_params
        )
        if tstep:
            sim_setup['analyses'][0]['strobe'] = tstep
        return netlist_info_from_dict(sim_setup)

    @classmethod
    def get_output_delay(cls, data: SimData, specs: Dict[str, Any],
                         shape: Optional[Tuple[int, ...]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute output delay from simulation data.

        This method only works for simulation data produced by this TestbenchManager,
        as it assumes a certain clk and input data pattern.

        if the output never resolved correctly, infinity is returned.
        """
        tper: float = specs['sim_params']['tper']
        vdd: float = specs['sim_params']['vdd']
        rtol: float = specs.get('rtol', 1e-8)
        atol: float = specs.get('atol', 1e-22)
        swp_info: Union[List[Tuple[str, Dict[str, Any]]], Dict[str, Any]] = specs['swp_info']

        swp_obj = swp_info_from_struct(swp_info)
        if 'vdd' in swp_obj:
            raise ValueError('Currently does not support vdd sweep.')

        data.open_analysis(AnalysisType.TRAN)
        tvec = data['time']
        yvec = data['out']

        tdr = get_first_crossings(tvec, yvec, vdd / 2, start=tper * 1.5, stop=tper * 2.5,
                                  etype=EdgeType.RISE, rtol=rtol, atol=atol, shape=shape)
        tdf = get_first_crossings(tvec, yvec, vdd / 2, start=tper * 3.5, stop=tper * 4.5,
                                  etype=EdgeType.FALL, rtol=rtol, atol=atol, shape=shape)
        tdr -= tper * 1.5
        tdf -= tper * 3.5
        return tdr, tdf

    @classmethod
    def get_output_trf(cls, data: SimData, specs: Dict[str, Any],
                       shape: Optional[Tuple[int, ...]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute output rise/fall time from simulation data.

        This method only works for simulation data produced by this TestbenchManager,
        as it assumes a certain clk and input data pattern.

        if output never crosses the high threshold, infinity is returned.  If output never
        crosses the low threshold, nan is returned.
        """
        tper: float = specs['tper']
        vdd: float = specs['vdd']
        thres_lo: float = specs['thres_lo']
        thres_hi: float = specs['thres_hi']
        rtol: float = specs.get('rtol', 1e-8)
        atol: float = specs.get('atol', 1e-22)
        swp_info: Union[List[Tuple[str, Dict[str, Any]]], Dict[str, Any]] = specs['swp_info']

        swp_obj = swp_info_from_struct(swp_info)
        if 'vdd' in swp_obj:
            raise ValueError('Currently does not support vdd sweep.')

        data.open_analysis(AnalysisType.TRAN)
        tvec = data['time']
        yvec = data['out']

        tr0 = get_first_crossings(tvec, yvec, vdd * thres_lo, start=tper * 1.5, stop=tper * 2.5,
                                  etype=EdgeType.RISE, rtol=rtol, atol=atol, shape=shape)
        tr1 = get_first_crossings(tvec, yvec, vdd * thres_hi, start=tper * 1.5, stop=tper * 2.5,
                                  etype=EdgeType.RISE, rtol=rtol, atol=atol, shape=shape)
        tf0 = get_first_crossings(tvec, yvec, vdd * thres_hi, start=tper * 3.5, stop=tper * 4.5,
                                  etype=EdgeType.FALL, rtol=rtol, atol=atol, shape=shape)
        tf1 = get_first_crossings(tvec, yvec, vdd * thres_lo, start=tper * 3.5, stop=tper * 4.5,
                                  etype=EdgeType.FALL, rtol=rtol, atol=atol, shape=shape)
        tr1 -= tr0
        tf1 -= tf0
        return tr1, tf1


class FlopSetupHold(MeasurementManager):
    """This class measures setup/hold time of a flop.

    setup/hold time is determined using binary search.

    Assumptions:
    1. The baseline delay is computed with tsetup = tper/2, thold = tper/4.
    2. simulation environment is not swept.
    """

    def __init__(self, sim: SimAccess, dir_path: Path, meas_name: str, impl_lib: str,
                 specs: Dict[str, Any], wrapper_lookup: Dict[str, str],
                 sim_view_list: Sequence[Tuple[str, str]], env_list: Sequence[str],
                 precision: int = 6) -> None:
        if len(env_list) > 1:
            raise ValueError(f'{self.__class__.__name__} does not support corner sweep.')

        MeasurementManager.__init__(self, sim, dir_path, meas_name, impl_lib, specs,
                                    wrapper_lookup, sim_view_list, env_list, precision=precision)

        trf_lut: List[List[str]] = self.specs['trf_lut']
        self._clk_trf_vals = np.array(trf_lut[0])
        self._in_trf_vals = np.array(trf_lut[1])

        self._tdr_thres: Optional[np.ndarray] = None
        self._tdf_thres: Optional[np.ndarray] = None
        self._bin_iters: Dict[Tuple[int, int], Tuple[FloatBinaryIterator, FloatBinaryIterator]] = {}
        self._results: Dict[str, np.ndarray] = {}

    def get_initial_state(self) -> str:
        """Returns the initial FSM state."""
        return 'baseline'

    def get_testbench_info(self, state: str, prev_output: Optional[Dict[str, Any]]
                           ) -> Tuple[str, str, Dict[str, Any], Optional[Dict[str, Any]]]:
        tmp = MeasurementManager.get_testbench_info(self, state, prev_output)
        tb_name, tb_type, tb_specs, tb_params = tmp

        if tb_type == 'baseline':
            # construct initial swp_info
            tb_specs['swp_info'] = [
                ('clk_trf', dict(type='LIST', values=self._clk_trf_vals.tolist())),
                ('trf', dict(type='LIST', values=self._in_trf_vals.tolist())),
            ]
        else:
            # load sweep configuration from previous binary search
            tb_specs['swp_info'] = prev_output['swp_info']

        return tb_name, tb_type, tb_specs, tb_params

    def get_testbench_specs(self, tb_type: str) -> Dict[str, Any]:
        """Helper method to get testbench specifications."""
        return self._specs['testbenches']['timing']

    def process_output(self, state: str, data: SimData, tb_manager: FlopTimingTB
                       ) -> Tuple[bool, str, Dict[str, Any]]:
        specs = self.specs

        tb_specs: Dict[str, Any] = tb_manager.specs

        if state == 'baseline':
            tres: float = specs['tres']
            threshold: float = specs['td_threshold']

            tper: float = tb_specs['sim_params']['tper']
            tper2 = tper / 2
            tper4 = tper / 4

            # get nominal delays
            shape = (self._clk_trf_vals.size, self._in_trf_vals.size)
            tdr_nom, tdf_nom = FlopTimingTB.get_output_delay(data, tb_specs, shape=shape)
            self._tdr_thres = tdr_nom * (1 + threshold)
            self._tdf_thres = tdf_nom * (1 + threshold)

            # initialize binary iterators, and construct swp_info
            swp_vals = []
            swp_info = dict(params=['clk_trf', 'trf', 'tdr', 'tdf'], values=swp_vals)
            for clk_idx, clk_trf in enumerate(self._clk_trf_vals):
                for in_idx, trf in enumerate(self._in_trf_vals):
                    iter_r = FloatBinaryIterator(-tper2, tper4 - trf, tol=tres)
                    iter_f = FloatBinaryIterator(-tper2, tper4 - trf, tol=tres)
                    self._bin_iters[(clk_idx, in_idx)] = (iter_r, iter_f)
                    swp_vals.append([clk_trf, trf, iter_r.get_next(), iter_f.get_next()])

            self._results['setup_rise'] = np.full(shape, np.nan)
            self._results['setup_fall'] = np.full(shape, np.nan)
            self._results['hold_rise'] = np.full(shape, np.nan)
            self._results['hold_fall'] = np.full(shape, np.nan)

            return False, 'setup_0', dict(tdr_nom=tdr_nom, tdf_nom=tdf_nom, swp_info=swp_info)
        elif state.startswith('setup') or state.startswith('hold'):
            return self._do_bin_search(tb_specs, data, state)
        else:
            raise ValueError('Unknown state: %s' % state)

    def _do_bin_search(self, tb_specs: Dict[str, Any], data: SimData, state: str
                       ) -> Tuple[bool, str, Dict[str, Any]]:
        rtol: float = tb_specs.get('rtol', 1e-8)
        atol: float = tb_specs.get('atol', 1e-22)

        clk_trf_vec = data['clk_trf']
        trf_vec = data['trf']

        # remove corner axis
        tdr, tdf = FlopTimingTB.get_output_delay(data, tb_specs, shape=(-1,))

        state_split = state.split('_')
        state_bn = state_split[0]
        state_idx = int(state_split[1])

        # update binary iterators, and construct next swp_info
        swp_vals = []
        for idx, (c_trf, i_trf) in enumerate(zip(clk_trf_vec, trf_vec)):
            clk_idx = cast(int, np.argmin(np.abs(self._clk_trf_vals - c_trf)))
            in_idx = cast(int, np.argmin(np.abs(self._in_trf_vals - i_trf)))
            iter_r, iter_f = self._bin_iters[(clk_idx, in_idx)]
            thres_r = self._tdr_thres[clk_idx, in_idx]
            thres_f = self._tdf_thres[clk_idx, in_idx]
            done_r = self._update(tdr[idx], thres_r, iter_r, state_bn, True,
                                  clk_idx, in_idx, rtol, atol)
            done_f = self._update(tdf[idx], thres_f, iter_f, state_bn, False,
                                  clk_idx, in_idx, rtol, atol)
            if not done_r or not done_f:
                next_rise = iter_r.get_last_save() if done_r else iter_r.get_next()
                next_fall = iter_f.get_last_save() if done_f else iter_f.get_next()
                swp_vals.append([c_trf, i_trf, next_rise, next_fall])

        if not swp_vals:
            if state_bn == 'setup':
                # all setup measurements are done, prepare for hold time measurement.
                # initialize binary iterators, and construct swp_info
                tres: float = self.specs['tres']

                tper: float = tb_specs['sim_params']['tper']
                tper2 = tper / 2
                tper4 = tper / 4

                for clk_idx, clk_trf in enumerate(self._clk_trf_vals):
                    for in_idx, trf in enumerate(self._in_trf_vals):
                        iter_r = FloatBinaryIterator(-tper2 + trf, tper4, tol=tres)
                        iter_f = FloatBinaryIterator(-tper2 + trf, tper4, tol=tres)
                        self._bin_iters[(clk_idx, in_idx)] = (iter_r, iter_f)
                        swp_vals.append([clk_trf, trf, iter_r.get_next(), iter_f.get_next()])

                swp_info = dict(params=['clk_trf', 'trf', 'thr', 'thf'], values=swp_vals)
                output = dict(setup_rise=self._results['setup_rise'],
                              setup_fall=self._results['setup_fall'], swp_info=swp_info)
                next_state = 'hold_0'
                done = False
            else:
                # all setup and hold are done
                output = self._results
                next_state = ''
                done = True
        else:
            if state_bn == 'setup':
                params = ['clk_trf', 'trf', 'tdr', 'tdf']
            else:
                params = ['clk_trf', 'trf', 'thr', 'thf']
            swp_info = dict(params=params, values=swp_vals)
            output = dict(swp_info=swp_info)
            key_rise = f'{state_bn}_rise'
            key_fall = f'{state_bn}_fall'
            output[key_rise] = self._results[key_rise]
            output[key_fall] = self._results[key_fall]
            next_state = f'{state_bn}_{state_idx + 1}'
            done = False

        return done, next_state, output

    def _update(self, td: float, thres: float, biter: FloatBinaryIterator, basename: str,
                is_rise: bool, clk_idx: int, in_idx: int, rtol: float, atol: float) -> bool:
        key = f'{basename}_rise' if is_rise else f'{basename}_fall'
        sign = -1 if basename == 'setup' else 1
        if biter.has_next():
            last_delay = biter.get_next()
            if np.isclose(td, thres, rtol=rtol, atol=atol):
                biter.save()
                self._results[key][clk_idx, in_idx] = sign * last_delay
                return True
            elif td > thres:
                biter.down() if sign < 0 else biter.up()
            elif td < thres:
                biter.save()
                biter.up() if sign < 0 else biter.down()
            if not biter.has_next():
                self._results[key][clk_idx, in_idx] = sign * last_delay
                return True
            return False
        return True


class FlopDelay(MeasurementManager):
    """This class measures output delay and output rise/fall of a flop.

    Notes
    -----
    We assume that:

    1. tsetup = tper/2, thold = tper/4.

    2. simulation environments are not swept.  This is so that the generated data matches
       output of FlopSetupHold.
    """

    def __init__(self, sim: SimAccess, dir_path: Path, meas_name: str, impl_lib: str,
                 specs: Dict[str, Any], wrapper_lookup: Dict[str, str],
                 sim_view_list: Sequence[Tuple[str, str]], env_list: Sequence[str],
                 precision: int = 6) -> None:
        if len(env_list) > 1:
            raise ValueError(f'{self.__class__.__name__} does not support corner sweep.')

        MeasurementManager.__init__(self, sim, dir_path, meas_name, impl_lib, specs,
                                    wrapper_lookup, sim_view_list, env_list, precision=precision)

        lut: List[List[str]] = self.specs['delay_lut']
        self._clk_trf_vals = np.array(lut[0])
        self._cload_trf_vals = np.array(lut[1])

    def get_initial_state(self) -> str:
        """Returns the initial FSM state."""
        return 'delay'

    def get_testbench_info(self, state: str, prev_output: Optional[Dict[str, Any]]
                           ) -> Tuple[str, str, Dict[str, Any], Optional[Dict[str, Any]]]:
        tmp = MeasurementManager.get_testbench_info(self, state, prev_output)
        tb_name, tb_type, tb_specs, tb_params = tmp

        if tb_type == 'delay':
            # construct initial swp_info
            tb_specs['swp_info'] = [
                ('clk_trf', dict(type='LIST', values=self._clk_trf_vals.tolist())),
                ('cload', dict(type='LIST', values=self._cload_trf_vals.tolist())),
            ]

        return tb_name, tb_type, tb_specs, tb_params

    def process_output(self, state: str, data: SimData, tb_manager: FlopTimingTB
                       ) -> Tuple[bool, str, Dict[str, Any]]:
        tb_specs = tb_manager.specs

        if state == 'delay':
            # get nominal delays
            shape = (self._clk_trf_vals.size, self._cload_trf_vals.size)
            tdr, tdf = FlopTimingTB.get_output_delay(data, tb_specs, shape=shape)
            trise, tfall = FlopTimingTB.get_output_trf(data, tb_specs, shape=shape)
            return True, '', dict(delay_rise=tdr, delay_fall=tdf,
                                  trf_rise=trise, trf_fall=tfall)
        else:
            raise ValueError('Unknown state: %s' % state)
