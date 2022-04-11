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

from typing import Any, Dict, List, Optional, Type

from pybag.enum import MinLenMode

from bag.util.immutable import Param
from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.layout.routing.base import WireArray
from bag.layout.template import TemplateDB

from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase, SupplyColumnInfo

from bag3_digital.layout.stdcells.levelshifter import LevelShifter


class PORVcclBuffer(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('aib_ams', 'aib_por_vccl_buffer')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return LevelShifter.get_params_info()

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo, flip_tile=True)

        lv_params: Param = self.params['lv_params']
        in_buf_params: Param = self.params['in_buf_params']

        lv_params = lv_params.copy(append=dict(dual_output=False))
        lv_master = self.new_template(LevelShifter, params=dict(pinfo=pinfo, lv_params=lv_params,
                                                                in_buf_params=in_buf_params))
        ridx_p = lv_master.ridx_p
        ridx_n = lv_master.ridx_n

        # add instance and supply columns
        hm_layer = self.conn_layer + 1
        top_layer = lv_master.top_layer
        sup_info = self.get_supply_column_info(top_layer)
        lay_range = range(hm_layer, top_layer + 1)
        vdd_io_table = {lay: [] for lay in lay_range}
        vdd_core_table = {lay: [] for lay in lay_range}
        vss_table = {lay: [] for lay in lay_range}

        cur_col = self._draw_supply_column(0, sup_info, vdd_io_table, vdd_core_table, vss_table,
                                           ridx_p, ridx_n, False)
        lv = self.add_tile(lv_master, 0, cur_col)
        cur_col += lv_master.num_cols + (self.sub_sep_col // 2)
        cur_col = self._draw_supply_column(cur_col, sup_info, vdd_io_table, vdd_core_table,
                                           vss_table, ridx_p, ridx_n, True)
        self.set_mos_size(cur_col)

        # connect supplies
        vss = vdd_io = vdd_core = None
        for lay in range(hm_layer, top_layer + 1, 2):
            vss = vss_table[lay]
            vdd_io = vdd_io_table[lay]
            vdd_core = vdd_core_table[lay]
            if lay == hm_layer:
                vss.append(lv.get_pin('VSS'))
                vdd_io.append(lv.get_pin('VDD'))
                vdd_core.append(lv.get_pin('VDD_in'))
            vss = self.connect_wires(vss)[0]
            vdd_io = self.connect_wires(vdd_io)[0]
            vdd_core = self.connect_wires(vdd_core)[0]

        self.add_pin('VDDCore', vdd_core)
        self.add_pin('VDDIO', vdd_io)
        self.add_pin('VSS', vss)

        # input/output pins
        yh = self.bound_box.yh
        self.add_pin('por_io', self.extend_wires(lv.get_pin('out'), upper=yh))
        self.add_pin('por_core', self.extend_wires(lv.get_pin('in'), lower=0))

        self.sch_params = lv_master.sch_params

    def _draw_supply_column(self, col: int, sup_info: SupplyColumnInfo,
                            vdd_io_table: Dict[int, List[WireArray]],
                            vdd_core_table: Dict[int, List[WireArray]],
                            vss_table: Dict[int, List[WireArray]],
                            ridx_p: int, ridx_n: int, flip_lr: bool) -> int:
        ncol = sup_info.ncol
        sup_col = col + int(flip_lr) * ncol
        # draw vdd core columns
        self.add_supply_column(sup_info, sup_col, vdd_core_table, vss_table, ridx_p=ridx_p,
                               ridx_n=ridx_n, flip_lr=flip_lr, extend_vdd=False,
                               extend_vss=False, min_len_mode=MinLenMode.MIDDLE)
        # draw vdd_io columns
        self.add_supply_column(sup_info, sup_col, vdd_io_table, vss_table, ridx_p=ridx_p,
                               ridx_n=ridx_n, tile_idx=1, flip_lr=flip_lr,
                               extend_vdd=False, extend_vss=False)
        return col + ncol
