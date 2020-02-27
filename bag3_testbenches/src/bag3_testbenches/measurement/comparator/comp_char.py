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

"""This package contains measurement class for comparators."""

from typing import Optional, Tuple, Dict, Any, Union, cast, Mapping

from pathlib import Path
from copy import deepcopy

from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult
from bag.simulation.measure import MeasurementManager, MeasInfo
from bag.simulation.core import TestbenchManager

from .tran import OverdriveMM
from .pss import CompPSSMM


class CompCharMM(MeasurementManager):
    """This class performs comparator characterization based on overdrive recovery test, PSS,
    PAC, PNoise"""

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        raise RuntimeError('Unused')

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance]) -> Dict[str, Any]:
        """Run OverdriveMM, pass results (switching threshold and hysteresis) to CompPSSMM,
        return final results"""
        # Overdrive recovery measurement
        specs_0 = deepcopy(self.specs)
        specs_0['sim_params']['t_rst'] = specs_0['sim_params']['tper']
        mm0 = cast(OverdriveMM, sim_db.make_mm(OverdriveMM, specs_0))
        mm0_results = await sim_db.async_simulate_mm_obj('OverdriveMM', sim_dir, dut, mm0)

        # PSS measurement
        specs_1 = deepcopy(self.specs)
        specs_1['sim_params']['t_rst'] = 0
        specs_1['tran_results'] = mm0_results.data
        mm1 = cast(CompPSSMM, sim_db.make_mm(CompPSSMM, specs_1))
        mm1_results = await sim_db.async_simulate_mm_obj('PSSMM', sim_dir, dut, mm1)

        return mm1_results.data
