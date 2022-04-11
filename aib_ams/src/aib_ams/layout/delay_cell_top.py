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

from typing import Any, Dict, Optional, Type

from pybag.enum import RoundMode, MinLenMode

from bag.util.immutable import Param
from bag.layout.routing.base import TrackID
from bag.layout.template import TemplateDB
from bag.design.database import Module

from xbase.layout.enum import MOSWireType
from xbase.layout.mos.base import MOSBase

from .delay_line import DelayCellNoFlop


class DelayCellNoFlopTop(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return DelayCellNoFlop.get_schematic_class()

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        ans = DelayCellNoFlop.get_params_info()
        ans['w_min'] = 'minimum width.'
        return ans

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        ans = DelayCellNoFlop.get_default_param_values()
        ans['w_min'] = 0
        return ans

    def draw_layout(self) -> None:
        master: DelayCellNoFlop = self.new_template(DelayCellNoFlop, params=self.params)
        self.draw_base(master.get_draw_base_sub_pattern(0, 5, mirror=False))

        w_min: int = self.params['w_min']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        grid = self.grid
        tr_manager = self.tr_manager
        vm_w = tr_manager.get_width(vm_layer, 'sig')
        vm_w_sup = tr_manager.get_width(vm_layer, 'sup')
        xm_w = tr_manager.get_width(xm_layer, 'sig')
        xm_w_sup = tr_manager.get_width(xm_layer, 'sup')

        # placement
        # compute total number of columns
        sd_pitch = self.sd_pitch
        ncol_flop = master.num_cols
        w_cur = ncol_flop * sd_pitch
        ncol_tot = -(-max(w_min, w_cur) // sd_pitch)
        ncol_blk = self.arr_info.get_block_ncol(vm_layer, half_blk=False)
        ncol_tot = -(-ncol_tot // ncol_blk) * ncol_blk
        offset = (ncol_tot - ncol_flop) // 2

        # place instances
        row_intvl = master.substrate_row_intvl
        core = self.add_tile(master, 0, offset)
        vss1_warrs = []
        for start, ncol in row_intvl:
            vss1_warrs.append(self.add_substrate_contact(0, start + offset, tile_idx=4, seg=ncol))

        self.set_mos_size(num_cols=ncol_tot)

        vss0_tid = self.get_track_id(0, MOSWireType.DS, 'sup', tile_idx=0)
        vdd_tid = self.get_track_id(0, MOSWireType.DS, 'sup', tile_idx=2)
        vss1_tid = self.get_track_id(0, MOSWireType.DS, 'sup', tile_idx=4)
        y_vss0 = grid.track_to_coord(hm_layer, vss0_tid.base_index)
        y_vdd = grid.track_to_coord(hm_layer, vdd_tid.base_index)
        y_vss1 = grid.track_to_coord(hm_layer, vss1_tid.base_index)
        xm_vss0_tid = TrackID(xm_layer,
                              grid.coord_to_track(xm_layer, y_vss0, mode=RoundMode.LESS_EQ),
                              width=xm_w_sup)
        xm_vss1_tid = TrackID(xm_layer,
                              grid.coord_to_track(xm_layer, y_vss1, mode=RoundMode.GREATER_EQ),
                              width=xm_w_sup)
        xm_vdd_tid = TrackID(xm_layer, grid.coord_to_track(xm_layer, y_vdd, mode=RoundMode.NEAREST),
                             width=xm_w_sup)
        bot_list = tr_manager.place_wires(xm_layer, ['sup', 'sig', 'sig', 'sig', 'sig'],
                                          align_idx=0, align_track=xm_vss0_tid.base_index)[1]
        top_list = tr_manager.place_wires(xm_layer, ['sig', 'sig', 'sup'], align_idx=-1,
                                          align_track=xm_vss1_tid.base_index)[1]

        vss1_warrs.extend(core.port_pins_iter('VSS1'))
        vss0 = self.connect_to_tracks(core.get_all_port_pins('VSS0'), vss0_tid)
        vss1 = self.connect_to_tracks(vss1_warrs, vss1_tid)
        vdd = self.connect_to_tracks(core.get_all_port_pins('VDD'), vdd_tid)

        vm_l = core.get_pin('vm_l')
        vm_r = core.get_pin('vm_r')
        tidx_bk = tr_manager.get_next_track(vm_layer, vm_l.track_id.base_index,
                                            'sig', 'sig', up=False)
        tidxl = tr_manager.get_next_track(vm_layer, tidx_bk, 'sig', 'sup', up=False)
        tidxr = tr_manager.get_next_track(vm_layer, vm_r.track_id.base_index,
                                          'sig', 'sup', up=True)
        vss, vdd = self.connect_differential_tracks([vss0, vss1], vdd, vm_layer, tidxl, tidxr,
                                                    width=vm_w_sup)
        vss0 = self.connect_to_tracks(vss, xm_vss0_tid)
        vss1 = self.connect_to_tracks(vss, xm_vss1_tid)
        vdd = self.connect_to_tracks(vdd, xm_vdd_tid)
        sup_lower = min(vss0.lower, vss1.lower, vdd.lower, self.bound_box.xl)
        sup_upper = max(vss0.upper, vss1.upper, vdd.upper, self.bound_box.xh)
        vss = self.extend_wires([vss0, vss1], lower=sup_lower, upper=sup_upper)
        vdd = self.extend_wires(vdd, lower=sup_lower, upper=sup_upper)
        self.add_pin('VSS', vss)
        self.add_pin('VDD', vdd)

        bk = self.connect_to_tracks(core.get_pin('bk'), TrackID(vm_layer, tidx_bk, width=vm_w),
                                    min_len_mode=MinLenMode.MIDDLE)
        bk = self.connect_to_tracks(bk, TrackID(xm_layer, bot_list[2], width=xm_w),
                                    track_lower=sup_lower)
        self.add_pin('bk', bk)
        out_tid = TrackID(xm_layer, bot_list[4], width=xm_w)
        in_tid = TrackID(xm_layer, top_list[0], width=xm_w)
        outp = self.connect_to_tracks(core.get_pin('out_p'), out_tid, track_lower=sup_lower)
        cip = self.connect_to_tracks(core.get_pin('ci_p'), out_tid, track_upper=sup_upper)
        inp = self.connect_to_tracks(core.get_pin('in_p'), in_tid, track_lower=sup_lower)
        cop = self.connect_to_tracks(core.get_pin('co_p'), in_tid, track_upper=sup_upper)
        self.add_pin('ci_p', cip)
        self.add_pin('out_p', outp)
        self.add_pin('in_p', inp)
        self.add_pin('co_p', cop)

        self.sch_params = master.sch_params
