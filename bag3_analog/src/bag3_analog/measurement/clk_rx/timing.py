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

"""This package contains clock receiver timing measurement classes."""

from typing import Tuple, Dict, Any, Sequence, Optional, Union, List, cast

from pathlib import Path

import numpy as np

from bag.io.file import open_file
from bag.simulation.data import (
    SimData, SimNetlistInfo, netlist_info_from_dict, AnalysisType, swp_info_from_struct
)
from bag.simulation.base import SimAccess
from bag.simulation.core import MeasurementManager

from bag3_testbenches.measurement.data.tran import bits_to_pwl_iter
from bag3_testbenches.measurement.digital.timing import CombLogicTimingTB


class ClkRXTimingTB(CombLogicTimingTB):
    """This class sets up the rise/fall time and duty cycle measurement testbench.

    """

    def __init__(self, sim: SimAccess, work_dir: Path, tb_name: str, impl_lib: str,
                 specs: Dict[str, Any], sim_view_list: Sequence[Tuple[str, str]],
                 env_list: Sequence[str], precision: int = 6) -> None:
        CombLogicTimingTB.__init__(self, sim, work_dir, tb_name, impl_lib, specs, sim_view_list,
                                   env_list, precision=precision)

    def pre_setup(self, sch_params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Set up PWL waveform files."""

        specs = self.specs
        thres_lo: float = specs['thres_lo']
        thres_hi: float = specs['thres_hi']
        v_cm: float = specs['sim_params']['v_cm']
        v_amp: float = specs['sim_params']['v_amp']

        v_lo = v_cm - v_amp
        v_hi = v_cm + v_amp

        # generate PWL waveform files
        clk_data = [v_lo, v_hi, v_lo]
        clkb_data = [v_hi, v_lo, v_hi]

        trf_scale = f'{thres_hi - thres_lo:.4g}'
        clk_path = self.work_dir / 'clk_pwl.txt'
        with open_file(clk_path, 'w') as f:
            for _, s_tb, s_tr, val in bits_to_pwl_iter(clk_data):
                f.write(f'tbit*{s_tb}+trf*({s_tr})/{trf_scale} {val}\n')
        clkb_path = self.work_dir / 'clkb_pwl.txt'
        with open_file(clkb_path, 'w') as f:
            for _, s_tb, s_tr, val in bits_to_pwl_iter(clkb_data):
                f.write(f'tbit*{s_tb}+trf*({s_tr})/{trf_scale} {val}\n')

        ans = sch_params.copy()
        ans['in_file_list'] = [('clk', str(clk_path.resolve())), ('clkb', str(clkb_path.resolve()))]
        ans['clk_file_list'] = []
        return ans

    @classmethod
    def get_output_duty_cycle(cls, data: SimData, specs: Dict[str, Any], in_name: str,
                              out_name: str, shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """Compute output duty cycle from simulation data.

        This method only works for simulation data produced by this TestbenchManager,
        as it assumes a certain input data pattern.

        if the output never resolved correctly, infinity is returned.
        """
        tdr, tdf = cls.get_output_delay(data, specs, in_name=in_name, out_name=out_name,
                                        out_invert=False)
        tbit = specs['sim_params']['tbit']
        tper = 2 * tbit
        tbit_out = tbit + tdf - tdr
        out_duty_cycle = tbit_out / tper * 100

        return out_duty_cycle
