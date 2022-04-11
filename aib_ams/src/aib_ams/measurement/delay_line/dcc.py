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

from typing import Any, Sequence, Tuple, Set, Mapping, Dict, Iterable, Optional, Union, cast

from pathlib import Path

from pybag.core import get_cdba_name_bits

from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult
from bag.simulation.core import TestbenchManager
from bag.simulation.measure import MeasurementManager, MeasInfo

from bag3_testbenches.measurement.data.tran import EdgeType
from bag3_testbenches.measurement.digital.flop.base import (
    FlopTimingBase, FlopInputMode, FlopMeasMode
)


class ControlTimingTB(FlopTimingBase):
    """This class performs transient simulation to measure control signals timing constraints.

    Assumes maximum timing margin is t_clk_per/4.

    Notes
    -----
    specification dictionary has the following entries in addition to those in FlopTimingBase:

    flop_params : Mapping[str, Any]
        Flop parameters, with the following entries:

        out_pin : str
            the delay line output pin.
        clk_pin : str
            the delay line input pin.
        ctrl_pin : str
            the control pin(s).
        ctrl_code : int
            the ctrl code at which to measure setup/hold time.
        setup_offset : Union[float, Mapping[str, float]]
            setup time offset
        hold_offset : Union[float, Mapping[str, float]]
            hold time offset
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._in_list: Sequence[str] = []

        super().__init__(*args, **kwargs)

    def commit(self) -> None:
        # NOTE: initialize in_list before running super's initialization,
        # so get_stimuli() behaves correctly.
        flop_params: Mapping[str, Any] = self.specs['flop_params']
        ctrl_pin: str = flop_params['ctrl_pin']

        self._in_list = get_cdba_name_bits(ctrl_pin)
        self._in_list.reverse()

        super().commit()

    @property
    def num_cycles(self) -> int:
        return 1

    @property
    def c_load_pins(self) -> Iterable[str]:
        return [self.flop_params['out_pin']]

    @classmethod
    def get_default_flop_params(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def get_meas_modes(cls, flop_params: Mapping[str, Any]) -> Sequence[FlopMeasMode]:
        # TODO: hack, to reuse cache, instead of measuring setup_falling and hold_rising,
        # TODO: and have 4 measurements, we have 8 measurements and drop data for 4 of them.
        return [FlopMeasMode(in_mode=FlopInputMode.IN, in_rising=True, setup_rising=True,
                             hold_rising=True, meas_setup=True),
                FlopMeasMode(in_mode=FlopInputMode.IN, in_rising=True, setup_rising=True,
                             hold_rising=True, meas_setup=False),
                FlopMeasMode(in_mode=FlopInputMode.IN, in_rising=False, setup_rising=True,
                             hold_rising=True, meas_setup=True),
                FlopMeasMode(in_mode=FlopInputMode.IN, in_rising=False, setup_rising=True,
                             hold_rising=True, meas_setup=False),
                FlopMeasMode(in_mode=FlopInputMode.IN, in_rising=True, setup_rising=False,
                             hold_rising=False, meas_setup=True),
                FlopMeasMode(in_mode=FlopInputMode.IN, in_rising=True, setup_rising=False,
                             hold_rising=False, meas_setup=False),
                FlopMeasMode(in_mode=FlopInputMode.IN, in_rising=False, setup_rising=False,
                             hold_rising=False, meas_setup=True),
                FlopMeasMode(in_mode=FlopInputMode.IN, in_rising=False, setup_rising=False,
                             hold_rising=False, meas_setup=False),
                ]

    @classmethod
    def get_output_meas_modes(cls, flop_params: Mapping[str, Any]) -> Sequence[FlopMeasMode]:
        return []

    def get_stimuli(self) -> Tuple[Sequence[Mapping[str, Any]], Dict[str, int], Set[str],
                                   Sequence[str]]:
        mode = self.meas_mode
        flop_params = self.flop_params
        clk_pin: str = flop_params['clk_pin']
        out_pin: str = flop_params['out_pin']
        ctrl_code: int = flop_params['ctrl_code']

        ctrl_pin = self._in_list[ctrl_code]

        if mode.meas_setup:
            ctrl = dict(pin=ctrl_pin, tper='2*t_sim', tpw='t_sim', trf='t_rf',
                        td='t_clk_delay-t_setup', pos=mode.input_rising)
            var_list = ['t_setup']
        else:
            ctrl = dict(pin=ctrl_pin, tper='2*t_sim', tpw='t_sim', trf='t_rf',
                        td='t_clk_delay+t_hold', pos=not mode.input_rising)
            var_list = ['t_hold']

        pulses = [self.get_clk_pulse(clk_pin, mode.is_pos_edge_clk), ctrl]
        biases = {self._in_list[v]: 1 for v in range(0, ctrl_code)}
        for v in range(ctrl_code + 1, len(self._in_list)):
            biases[self._in_list[v]] = 0

        outputs = {ctrl_pin, clk_pin, out_pin}
        return pulses, biases, outputs, var_list

    def get_output_map(self, output_timing: bool
                       ) -> Mapping[str, Tuple[Mapping[str, Any],
                                               Sequence[Tuple[EdgeType, Sequence[str]]]]]:
        if output_timing:
            return {}

        mode = self.meas_mode
        flop_params = self.flop_params
        clk_pin: str = flop_params['clk_pin']
        out_pin: str = flop_params['out_pin']
        setup_offset: float = flop_params['setup_offset']
        hold_offset: float = flop_params['hold_offset']

        if mode.meas_setup:
            var_name = 't_setup'
            cons_offset = setup_offset
        else:
            var_name = 't_hold'
            cons_offset = hold_offset

        inc_delay = not mode.input_rising
        out_edge = EdgeType.RISE if mode.is_pos_edge_clk else EdgeType.FALL

        # TODO: caching hack
        if mode.meas_setup ^ mode.is_pos_edge_clk:
            timing_info = self.get_timing_info(mode, self._in_list, clk_pin, '', True,
                                               inc_delay=inc_delay, offset=cons_offset)
            edge_out_list = [(out_edge, [out_pin])]
            return {var_name: (timing_info, edge_out_list)}
        else:
            return {}


class ControlTimingMM(MeasurementManager):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance]) -> Dict[str, Any]:
        specs = self.specs
        tbm_specs_orig: Mapping[str, Any] = specs['tbm_specs']
        flop_params: Mapping[str, Any] = specs['flop_params']
        in_rising: bool = specs['in_rising']
        clk_rising: bool = specs['clk_rising']
        meas_setup: bool = specs['meas_setup']
        swp_specs: Mapping[str, Any] = specs['swp_specs']
        plot: bool = specs.get('plot', False)

        out_pin: str = flop_params['out_pin']

        meas_mode = FlopMeasMode(in_mode=FlopInputMode.IN, in_rising=in_rising,
                                 setup_rising=clk_rising, hold_rising=clk_rising,
                                 meas_setup=meas_setup)
        tbm_specs = dict(**tbm_specs_orig)
        tbm_specs['flop_params'] = flop_params
        tbm_specs['meas_mode'] = meas_mode

        var_name = 't_setup' if meas_mode.meas_setup else 't_hold'
        tbm_specs['swp_info'] = [(var_name, swp_specs)]
        tbm_specs['dut_pins'] = list(dut.sch_master.pins.keys())
        tbm_specs['load_list'] = [dict(pin=out_pin, type='cap', value='c_load')]
        tbm = cast(ControlTimingTB, self.make_tbm(ControlTimingTB, tbm_specs))
        result = await sim_db.async_simulate_tbm_obj('td_ctrl', sim_dir / 'td_ctrl', dut,
                                                     tbm, {})
        data = result.data
        calc = tbm.get_calculator(data)

        xvec = calc.eval(var_name)

        out_edge = EdgeType.RISE if meas_mode.is_pos_edge_clk else EdgeType.FALL
        td = tbm.calc_clk_to_q(data, out_pin, out_edge)

        if plot:
            from matplotlib import pyplot as plt
            plt.figure(1)
            plt.plot(xvec.flatten(), td.flatten())
            plt.show()

        return dict(xvec=xvec, yvec=td)

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        raise RuntimeError('Unused')

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')
