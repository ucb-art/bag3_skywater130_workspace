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
class aib_ams__aib_ff_chain(Module):
    """Module for library aib_ams cell aib_ff_chain.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'aib_ff_chain.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            ff_params='FF parameters',
            nbits='Number of FFs chained together',
        )

    def design(self, ff_params: Dict[str, Any], nbits: int) -> None:

        suffix = f'<{nbits-1}:0>'
        self.rename_pin('out', f'out{suffix}')
        self.rename_pin('outb', f'outb{suffix}')
        self.instances['Xflop'].design(**ff_params)

        term_list, name_list = [], []
        for i in range(nbits):
            name_list.append(f'XFF<{i}>')
            term_dict = {'out': f'out<{i}>', 'outb': f'outb<{i}>', 'in': f'out<{i-1}>'}
            if i == 0:
                term_dict['in'] = 'in'
            term_list.append(term_dict)

        self.array_instance('Xflop', name_list, term_list)
