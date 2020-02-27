# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
# Copyright 2018 Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

from pybag.enum import RoundMode, MinLenMode

from bag.layout.core import PyLayInstance
from bag.util.immutable import Param
from bag.layout.template import TemplateDB
from bag.layout.routing.base import TrackID
from bag.design.database import ModuleDB, Module

from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase
from xbase.layout.enum import MOSWireType

from bag3_digital.layout.stdcells.memory import FlopCore


class FFChainCore(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            ff_params='FF Params according to the docs of memory.FLopCore'
                      '(does not include pinfo, ridxn, and ridxp)',
            nbits='Number of ffs',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            sig_locs='Optional dictionary of user defined signal locations',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            ridx_n=0,
            ridx_p=-1,
            sig_locs=None,
        )

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('aib_ams', 'aib_ff_chain')

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        ridx_n: int = self.params['ridx_n']
        ridx_p: int = self.params['ridx_p']
        nbits: int = self.params['nbits']
        ff_params: Dict[str, Any] = self.params['ff_params']
        sig_locs: Optional[Dict[str, float]] = self.params['sig_locs']

        if not ff_params['resetable']:
            raise ValueError('non-resetable ff_chain is not supported')

        if sig_locs is None:
            sig_locs = {}

        hm_layer = self.conn_layer + 1
        vm_layer = self.conn_layer + 2
        hm2_layer = self.conn_layer + 3

        show_sub_block_pins: bool = False
        ng = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=-1)
        pg = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=0)
        ff_params = dict(pinfo=pinfo, ridx_n=ridx_n, ridx_p=ridx_p, show_pins=show_sub_block_pins,
                         sig_locs=dict(nin=ng, pout=pg),
                         **ff_params)
        ff0_master = self.new_template(FlopCore, params=ff_params)

        ff_params['sig_locs'] = dict(nin=pg, pout=ng)
        ff1_master = self.new_template(FlopCore, params=ff_params)

        sep = max(self.get_hm_sp_le_sep_col(), self.min_sep_col)
        assert ff0_master.num_cols == ff1_master.num_cols, 'Routing difference should not change ' \
                                                           'the width of the master.'
        ff_cols = ff0_master.num_cols
        tot_cols = nbits * (ff_cols + sep)
        self.set_mos_size(tot_cols)

        ff_arr: List[PyLayInstance] = []
        cur_col = 0
        for idx in range(nbits):
            if idx % 2:
                ff_arr.append(self.add_tile(ff1_master, 0, cur_col))
            else:
                ff_arr.append(self.add_tile(ff0_master, 0, cur_col))
            cur_col += sep + ff_cols

        # routing:
        tr_manager = pinfo.tr_manager
        tr_w_v = tr_manager.get_width(vm_layer, 'sig')
        tr_w_h2 = tr_manager.get_width(hm2_layer, 'sig')
        # vdd/vss
        vdd_list, vss_list = [], []
        for inst in ff_arr:
            vdd_list += inst.get_all_port_pins('VDD')
            vss_list += inst.get_all_port_pins('VSS')

        vdd_list = self.connect_wires(vdd_list)
        vss_list = self.connect_wires(vss_list)

        # connect output of i-1 to input of i and connect pins
        for i in range(nbits-1):
            out_prev = ff_arr[i].get_pin('out')
            in_next = ff_arr[i + 1].get_pin('nin')
            hm_out_i = self.connect_to_track_wires(out_prev, in_next)
            self.add_pin(f'nout<{i}>', hm_out_i, hide=True)
            self.add_pin(f'pout<{i}>', hm_out_i, hide=True)
            self.add_pin(f'out<{i}>', [out_prev, ff_arr[i + 1].get_pin('in')])
            self.reexport(ff_arr[i].get_port('outb'), net_name=f'outb<{i}>')
        self.reexport(ff_arr[-1].get_port('out'), net_name=f'out<{nbits - 1}>')
        self.reexport(ff_arr[-1].get_port('pout'), net_name=f'pout<{nbits - 1}>', hide=True)
        self.reexport(ff_arr[-1].get_port('nout'), net_name=f'nout<{nbits - 1}>', hide=True)
        self.reexport(ff_arr[-1].get_port('outb'), net_name=f'outb<{nbits-1}>')

        # connect rst on hm2 layer
        rst_warrs = []
        for i in range(nbits):
            coord = ff_arr[i].bound_box.xh
            tid = self.grid.coord_to_track(vm_layer, coord, RoundMode.GREATER_EQ)
            ff_rst = self.connect_to_tracks(ff_arr[i].get_pin('nrst'),
                                            TrackID(vm_layer, tid, width=tr_w_v),
                                            min_len_mode=MinLenMode.LOWER)
            rst_warrs.append(ff_rst)

        self.add_pin('rst_vm', rst_warrs[0], hide=True)
        coord = self.grid.track_to_coord(hm_layer, ff_arr[0].get_pin('nrst').track_id.base_index)
        tidx = self.grid.coord_to_track(hm2_layer, coord, RoundMode.NEAREST)
        rst_hm2 = self.connect_to_tracks(rst_warrs, TrackID(hm2_layer, tidx, width=tr_w_h2))
        tidx_clk = sig_locs.get('clk', tr_manager.get_next_track(hm2_layer, tidx,
                                                                 'sig', 'sig', up=False))
        clk_list = [ff.get_pin('clk') for ff in ff_arr]
        clk_warr = self.connect_to_tracks(clk_list, TrackID(hm2_layer, tidx_clk, width=tr_w_h2))
        self.add_pin('clk_vm', clk_list[0])

        # add/reexport pins
        self.add_pin('VDD', vdd_list)
        self.add_pin('VSS', vss_list)
        self.add_pin('rst', rst_hm2)
        self.add_pin('clk', clk_warr)
        self.reexport(ff_arr[0].get_port('in'), net_name='in')
        self.reexport(ff_arr[0].get_port('nin'), net_name='nin', hide=True)
        self.reexport(ff_arr[0].get_port('pin'), net_name='pin', hide=True)
        self.reexport(ff_arr[0].get_port('mid_vm_m'), net_name='mid_vm_in', hide=True)
        self.reexport(ff_arr[-1].get_port('mid_vm_s'), net_name='mid_vm_out', hide=True)

        assert ff0_master.sch_params == ff1_master.sch_params, 'Routing difference should not ' \
                                                               'change the schematic parameters ' \
                                                               'of the master.'
        self.sch_params = dict(ff_params=ff0_master.sch_params, nbits=nbits)
