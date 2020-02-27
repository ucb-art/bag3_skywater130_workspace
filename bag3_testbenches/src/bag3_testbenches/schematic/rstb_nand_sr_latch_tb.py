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

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_testbenches__rstb_nand_sr_latch_tb(Module):
    """Module for library bag3_testbenches cell rstb_nand_sr_latch_tb.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'rstb_nand_sr_latch_tb.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            dut_lib='DUT library',
            dut_cell='DUT cellname'
        )

    def design(self, dut_lib: str, dut_cell: str) -> None:
        self.replace_instance_master('XDUT', dut_lib, dut_cell, static=True, keep_connections=True)
