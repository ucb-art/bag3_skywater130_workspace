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

from typing import Any, Sequence, Tuple, Set, Mapping, Dict, Iterable, Optional, Union

from pathlib import Path

from pybag.core import get_cdba_name_bits

from bag.io.file import write_yaml

from bag.concurrent.util import GatherHelper
from bag.simulation.core import TestbenchManager
from bag.simulation.measure import MeasurementManager, MeasInfo
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult

from bag3_testbenches.measurement.data.tran import EdgeType
from bag3_testbenches.measurement.digital.comb import CombLogicTimingMM
from bag3_testbenches.measurement.digital.flop.base import (
    FlopTimingBase, FlopInputMode, FlopMeasMode
)


class ControlTimingTB(FlopTimingBase):
    """This class performs transient simulation to measure control signals timing constraints.

    Notes
    -----
    specification dictionary has the following entries in addition to those in FlopTimingBase:

    flop_params : Mapping[str, Any]
        Flop parameters, with the following entries:

        in_pin : str
            the input pin.
        out_pin : str
            the output pin.
        ctrl_pin : str
            the control pin(s).
        setup_rising : bool
            True to measure setup from rising edge.
        hold_rising : bool
            True to measure hold from rising edge.
        out_invert : bool
            Defaults to False.  True if the output is inverted.
        setup_offset : Union[float, Mapping[str, float]]
            Defaults to 0.  The offset to add to setup time.  Can be sim_env dependnent.
        hold_offset : Union[float, Mapping[str, float]]
            Defaults to 0.  The offset to add to hold time.  Can be sim_env dependent.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @property
    def num_cycles(self) -> int:
        return 1

    @property
    def c_load_pins(self) -> Iterable[str]:
        return [self.flop_params['out_pin']]

    @classmethod
    def get_default_flop_params(cls) -> Dict[str, Any]:
        return dict(out_invert=False, setup_offset=0, hold_offset=0)

    @classmethod
    def get_meas_modes(cls, flop_params: Mapping[str, Any]) -> Sequence[FlopMeasMode]:
        setup_rising: bool = flop_params['setup_rising']
        hold_rising: bool = flop_params['hold_rising']

        return [FlopMeasMode(in_mode=FlopInputMode.IN, in_rising=True, setup_rising=setup_rising,
                             hold_rising=hold_rising, meas_setup=True),
                FlopMeasMode(in_mode=FlopInputMode.IN, in_rising=True, setup_rising=setup_rising,
                             hold_rising=hold_rising, meas_setup=False),
                FlopMeasMode(in_mode=FlopInputMode.IN, in_rising=False, setup_rising=setup_rising,
                             hold_rising=hold_rising, meas_setup=True),
                FlopMeasMode(in_mode=FlopInputMode.IN, in_rising=False, setup_rising=setup_rising,
                             hold_rising=hold_rising, meas_setup=False),
                ]

    @classmethod
    def get_output_meas_modes(cls, flop_params: Mapping[str, Any]) -> Sequence[FlopMeasMode]:
        return []

    def get_stimuli(self) -> Tuple[Sequence[Mapping[str, Any]], Dict[str, int], Set[str],
                                   Sequence[str]]:
        mode = self.meas_mode
        flop_params = self.flop_params

        ctrl_pin: str = flop_params['ctrl_pin']
        in_pin: str = flop_params['in_pin']

        pulses = [self.get_clk_pulse(in_pin, mode.is_pos_edge_clk)]

        pos = mode.input_rising
        hold_opposite = mode.hold_opposite_clk
        stimuli_pins = get_cdba_name_bits(ctrl_pin)
        var_setup = 't_setup'
        var_hold = 't_hold'
        var_list = [var_setup, var_hold]
        for pin in stimuli_pins:
            pulses.append(self.get_input_pulse(pin, var_setup, var_hold, pos, cycle_idx=0,
                                               hold_opposite=hold_opposite))

        outputs = set(stimuli_pins)
        outputs.add(in_pin)
        outputs.add(self.flop_params['out_pin'])
        return pulses, {}, outputs, var_list

    def get_output_map(self, output_timing: bool
                       ) -> Mapping[str, Tuple[Mapping[str, Any],
                                               Sequence[Tuple[EdgeType, Sequence[str]]]]]:
        if output_timing:
            return {}

        mode = self.meas_mode
        flop_params = self.flop_params
        out_invert: bool = flop_params['out_invert']
        in_pin: str = flop_params['in_pin']
        ctrl_pin: str = flop_params['ctrl_pin']
        setup_offset: Union[float, Mapping[str, float]] = flop_params['setup_offset']
        hold_offset: Union[float, Mapping[str, float]] = flop_params['hold_offset']

        stimuli_pins = get_cdba_name_bits(ctrl_pin)

        if mode.meas_setup:
            var_name = 't_setup'
            offset = setup_offset
            out_edge = EdgeType.RISE
        else:
            var_name = 't_hold'
            offset = hold_offset
            out_edge = EdgeType.FALL

        diff_grp = self.get_diff_groups(self.flop_params['out_pin'])
        rise_idx = int(not (mode.is_pos_edge_clk ^ out_invert))
        timing_info = self.get_timing_info(mode, stimuli_pins, in_pin, '', True,
                                           inc_delay=not mode.input_rising, offset=offset)
        edge_out_list = [(out_edge, diff_grp[rise_idx]),
                         (out_edge.opposite, diff_grp[rise_idx ^ 1])]
        return {var_name: (timing_info, edge_out_list)}


class DelayCharMM(MeasurementManager):
    """Characterize delay vs corners.

    Notes
    -----
    specification dictionary has the following entries:

    in_pin : str
        the input pin.
    out_pin : str
        the output pin.
    ctrl_pin : str
        the control pin.
    ctrl_0 : int
        the minimum control code.
    ctrl_1 : int
        the maximum control code.
    out_invert : bool
        Defaults to False.  True if the output is inverted.
    tbm_specs : Mapping[str, Any]
        DigitalTranTB related specifications.  The following simulation parameters are required:

            t_rst :
                reset duration.
            t_rst_rf :
                reset rise/fall time.
            t_bit :
                bit value duration.
            t_rf :
                input rise/fall time.
            c_load :
                load capacitance.
    sim_envs_swp_info : Mapping[str, Any]
        corner sweep information list.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._td_specs: Mapping[str, Any] = {}
        self._ctrl: str = ''
        self._sup_list: Sequence[str] = []

        super().__init__(*args, **kwargs)

    def commit(self) -> None:
        super().commit()

        specs = self.specs
        in_pin: str = specs['in_pin']
        out_pin: str = specs['out_pin']
        ctrl_pin: str = specs['ctrl_pin']
        out_invert: bool = specs['out_invert']
        tbm_specs: Mapping[str, Any] = specs['tbm_specs']

        self._ctrl = ctrl_pin
        self._sup_list = sorted(tbm_specs['sup_values'].keys())
        self._td_specs = dict(
            in_pin=in_pin,
            out_pin=out_pin,
            out_invert=out_invert,
            tbm_specs=tbm_specs,
            out_rise=True,
            out_fall=True,
        )

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance]) -> Dict[str, Any]:
        specs = self.specs
        out_pin: str = specs['out_pin']
        sim_envs_swp_info: Mapping[str, Any] = specs['sim_envs_swp_info']
        t_clk_per: float = specs['t_clk_per']
        t_dist_targ: Mapping[str, float] = specs['t_dist_targ']
        t_dlycell: Mapping[str, float] = specs['t_dlycell']

        name_format: str = sim_envs_swp_info['name_format']
        voltage_precision: int = sim_envs_swp_info['voltage_precision']
        voltage_types: Mapping[str, str] = sim_envs_swp_info['voltage_types']
        sim_envs: Sequence[Mapping[str, Any]] = sim_envs_swp_info['sim_envs']

        # make distinct measurement managers
        mm_table = {}
        result_table = {}
        voltage_fmt = '{:.%df}' % voltage_precision
        result_name_list = []
        for env_config in sim_envs:
            sim_env: str = env_config['sim_env']
            voltages: Mapping[str, float] = env_config['voltages']

            # get result name
            vstr_table = {k: voltage_fmt.format(v).replace('.', 'p') for k, v in voltages.items()}
            result_name = name_format.format(sim_env=sim_env, **vstr_table)
            self._make_td_mm(result_name, sim_db, dut, sim_env, voltage_fmt, voltage_types,
                             voltages, False, mm_table, result_table)
            self._make_td_mm(result_name, sim_db, dut, sim_env, voltage_fmt, voltage_types,
                             voltages, True, mm_table, result_table)

            result_name_list.append(result_name)

        # simulate in batch
        gatherer = GatherHelper()
        idx_map = {}
        for idx, (work_dir, mm) in enumerate(mm_table.items()):
            gatherer.append(sim_db.async_simulate_mm_obj(work_dir, sim_dir / work_dir, dut, mm))
            idx_map[work_dir] = idx

        results = await gatherer.gather_err()

        # post-process data
        max_table = {}
        min_table = {}
        rfmax_table = {}
        rfmin_table = {}
        ans = {'td_min': min_table, 'td_max': max_table, 'trf_min': rfmin_table,
               'trf_max': rfmax_table}
        for (result_name, min_delay), work_dir in result_table.items():
            idx = idx_map[work_dir]
            data = results[idx].data['timing_data'][out_pin]
            td_rise = data['cell_rise'].item()
            td_fall = data['cell_fall'].item()
            tr = data['rise_transition'].item()
            tf = data['fall_transition'].item()
            if min_delay:
                if td_rise < td_fall:
                    min_table[result_name] = td_rise
                    rfmin_table[result_name] = tr
                else:
                    min_table[result_name] = td_fall
                    rfmin_table[result_name] = tf
            else:
                if td_rise < td_fall:
                    max_table[result_name] = td_fall
                    rfmax_table[result_name] = tf
                else:
                    max_table[result_name] = td_rise
                    rfmax_table[result_name] = tr

        t_diff = {}
        t_dll_dlyline = {}
        t_dcc_dlyline = {}
        t_dcc_setup_ctrl = {}
        t_dcc_hold_ctrl = {}
        for result_name in result_name_list:
            t_max_pi = max_table[result_name]
            t_min_pi = min_table[result_name]
            t_dlycell_cur = t_dlycell[result_name]

            t_diff[result_name] = t_max_pi - t_min_pi
            t_dll_dlyline[result_name] = t_clk_per / 4 - t_dist_targ[result_name] - t_min_pi
            t_dcc_dlyline[result_name] = t_clk_per / 2 + t_dlycell_cur
            t_dcc_setup_ctrl[result_name] = -(t_clk_per / 2 - t_max_pi + t_min_pi +
                                              t_dlycell_cur) / 2
            t_dcc_hold_ctrl[result_name] = (t_clk_per / 2 + t_dlycell_cur) / 2

        ans['t_diff'] = t_diff
        ans['t_dll_dlyline'] = t_dll_dlyline
        ans['t_dcc_dlyline'] = t_dcc_dlyline
        ans['t_dcc_setup_ctrl'] = t_dcc_setup_ctrl
        ans['t_dcc_hold_ctrl'] = t_dcc_hold_ctrl
        write_yaml(sim_dir / f'{name}.yaml', ans)
        return ans

    def _make_td_mm(self, result_name: str, sim_db: SimulationDB,
                    dut: Optional[DesignInstance], sim_env: str, voltage_fmt: str,
                    voltage_types: Mapping[str, str], voltages: Mapping[str, float],
                    min_delay: bool, mm_table: Dict[str, MeasurementManager],
                    result_table: Dict[Tuple[str, int], str]) -> None:
        ctrl_code = self.specs['ctrl_0' if min_delay else 'ctrl_1']

        # get working directory
        str_list = [f'{k}_{voltage_fmt.format(voltages[voltage_types[k]]).replace(".", "p")}'
                    for k in self._sup_list]
        work_dir = f'td_{dut.cache_name}_{ctrl_code}_{sim_env}_{"_".join(str_list)}'
        if work_dir not in mm_table:
            mm_specs = dict(**self._td_specs)
            mm_specs['tbm_specs'] = tbm_specs = dict(**mm_specs['tbm_specs'])
            tbm_specs['sup_values'] = sup_values = dict(**tbm_specs['sup_values'])
            tbm_specs['pin_values'] = pin_values = dict(**tbm_specs['pin_values'])

            # update measurement specs and create Measurement Manager
            tbm_specs['sim_envs'] = [sim_env]
            for sup_pin in list(sup_values.keys()):
                sup_values[sup_pin] = voltages[voltage_types[sup_pin]]
            pin_values[self._ctrl] = ctrl_code
            mm = sim_db.make_mm(CombLogicTimingMM, mm_specs)

            mm_table[work_dir] = mm
        result_table[(result_name, min_delay)] = work_dir

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        raise RuntimeError('Unused')

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')
