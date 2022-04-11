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

"""This module contains layout generators for differential amplifiers."""

from typing import Any, Dict, Type, Optional, List, Sequence, Mapping, Union

import numpy as np

from pybag.enum import MinLenMode, RoundMode

from bag.typing import TrackType
from bag.util.math import HalfInt
from bag.util.immutable import Param
from bag.layout.template import TemplateDB
from bag.layout.routing.base import TrackID, WireArray
from bag.design.module import Module

from bag.env import get_tech_global_info

from bag3_digital.layout.stdcells.gates import PassGateCore

from xbase.layout.enum import MOSWireType, MOSType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase

from ..enum import IntFlag

from ..schematic.moscap import bag3_analog__moscap
from ..schematic.variablecapchain2 import bag3_analog__variablecapchain2


class MOSCapType(IntFlag):
    NMOS = 1  # Draw all NMOS
    PMOS = 2  # Draw all PMOS
    BOTH = 3  # Draw roughly half NMOS and half PMOS


class MOSCapUnit(MOSBase):
    """Single MOSCap
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

        self._test_seg = 10
        test_params = params.copy(append={'test_seg': self._test_seg})
        test_dut = MOSDUT(temp_db, test_params, **kwargs)
        test_dut.draw_layout()
        self._test_bbox = test_dut.bound_box
        self.seg_row, self.w, self.rows = -1, -1, -1

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return bag3_analog__moscap

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The TileInfoTable object. If None, table will be created dynamically',
            type='MOSCap Flavor',
            seg='transistor segments',
            seg_n='nmos segments',
            seg_p='pmos segments',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            show_pins='True to show pins',
            flip_tile='True to flip all tiles',
            sig_locs='Signal locations for top horizontal metal layer pins',
            vertical_in='True to draw input on vertical metal layer.',
            vertical_sup='True to have supply unconnected on conn_layer.',
            fill_width='width unit constraint to set cap fill',
            fill_height='height unit constraint to set cap fill',
            target_area='target MOS area, in units of total segments * width',
            num_rows='',
            arr_info='',
            g_on_s='',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            type=MOSCapType.BOTH,
            pinfo=None,
            seg=-1,
            seg_n=-1,
            seg_p=-1,
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            show_pins=False,
            flip_tile=False,
            sig_locs={},
            vertical_in=False,
            vertical_sup=False,
            fill_width=-1,
            fill_height=-1,
            target_area=-1,
            num_rows=-1,
            arr_info={},
            g_on_s=False,
        )

    def draw_layout(self) -> None:
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        type: MOSCapType = self.params['type']
        sig_locs: Dict[str, TrackType] = self.params['sig_locs']
        vertical_in: bool = self.params['vertical_in']
        vertical_sup: bool = self.params['vertical_sup']
        g_on_s:bool = self.params['g_on_s']

        pinfo, seg_dict = self._determine_dimensions()
        self.draw_base(pinfo, flip_tile=self.params['flip_tile'])

        vertical_in = vertical_in or len(seg_dict) > 1

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1

        # set is_guarded = True if both rows has same orientation
        is_guarded = False

        # Placement
        max_seg = max(seg_dict.values())
        ports_ds = {}
        ports_g = {}

        num_rows = pinfo.num_rows
        for n in range(num_rows):
            if n not in seg_dict.keys():
                continue
            seg = seg_dict[n]
            loc = (max_seg - seg) // 2

            ports = self.add_mos(n, loc, seg, g_on_s=g_on_s)
            ports_ds[n] = [ports.d, ports.s]
            ports_g[n] = ports.g

        self.set_mos_size()

        # get wire_indices from sig_locs
        tr_manager = self.tr_manager
        tr_w_h = tr_manager.get_width(hm_layer, 'sig')
        tr_w_v = tr_manager.get_width(vm_layer, 'sig')

        # Connect input
        if is_guarded:
            raise ValueError("Guard ring currently not supported")
        else:
            # Connect intermediate tracks
            in_hm_list = []
            for n, g in ports_g.items():
                if n == num_rows - 1 and num_rows % 2:
                    _flip = pinfo.get_row_place_info(num_rows-1).row_info.flip
                    wire_idx = 0 if _flip else -1
                    in_tidx = self.get_track_index(num_rows - 1, MOSWireType.G, wire_name='sig',
                                                   wire_idx=wire_idx)
                else:
                    n_2 = n // 2
                    _flip = pinfo.get_row_place_info(2 * n_2).row_info.flip
                    wire_idx = 0 if _flip else -1
                    in0_tidx = self.get_track_index(2 * n_2, MOSWireType.G, wire_name='sig',
                                                    wire_idx=wire_idx)
                    _flip = pinfo.get_row_place_info(2 * n_2 + 1).row_info.flip
                    wire_idx = 0 if _flip else -1
                    in1_tidx = self.get_track_index(2 * n_2 + 1, MOSWireType.G, wire_name='sig',
                                                    wire_idx=wire_idx)
                    in_tidx = self.grid.get_middle_track(in0_tidx, in1_tidx)
                in_hm_tid = TrackID(hm_layer, in_tidx, width=tr_w_h)
                in_hm_warr = self.connect_to_tracks(g, in_hm_tid)
                in_hm_list.append(in_hm_warr)

            # Draw optional vm and add pin
            if not vertical_in:
                self.add_pin('in', in_hm_warr, connect=True)
            else:
                vm_tidx = self.grid.find_next_track(vm_layer, in_hm_list[0].middle,
                                                    tr_width=tr_w_v, mode=RoundMode.GREATER_EQ)
                vm_tid = TrackID(vm_layer, vm_tidx, width=tr_w_v)
                in_vm_warr = self.connect_to_tracks(in_hm_list, vm_tid)
                self.add_pin('in', in_vm_warr)
                self.add_pin('in_hm', in_hm_list, show=False)

        # Connect Supply
        xr = self.bound_box.xh
        for n, sd_list in ports_ds.items():
            sup_tid = self.get_track_id(n, MOSWireType.DS, wire_name='sup')
            sup = self.connect_to_tracks(sd_list, sup_tid, track_lower=0, track_upper=xr)

            row_type: MOSType = pinfo.get_row_place_info(n).row_info.row_type
            sup_name = 'VSS' if row_type == MOSType.nch else 'VDD'
            self.add_pin(sup_name, sup, connect=True)

        # Set sch params
        rows_n = []
        rows_p = []
        for ridx in range(pinfo.num_rows):
            row_info = self.place_info.get_row_place_info(ridx).row_info
            row_sch_params = dict(
                seg=seg_dict[ridx],
                w=row_info.width,
                th=row_info.threshold,
            )
            if row_info.row_type == MOSType.nch:
                rows_n.append(row_sch_params)
            else:
                rows_p.append(row_sch_params)

        lch = self.place_info.lch
        self.sch_params = dict(
            lch=lch,
            rows_n=rows_n,
            rows_p=rows_p,
        )

    def _determine_dimensions(self):
        """Determines how to draw the moscap in one of the following conditions
        1) pinfo is given and individual cap segments are explicit
        2) pinfo is given but we are only given total cap area
        3) pinfo is not given but we are given total cap area, seg, and num rows
        4) pinfo is not given and we are given total cap area and total height or total width

        Total cap area is defined as total segments * width

        Returns
        -------
        pinfo
        seg_dict: Mapping[int]: maps row index to how many segments in each row
        """
        pinfo = self.params['pinfo']
        seg: int = self.params['seg']
        seg_p: int = self.params['seg_p']
        seg_n: int = self.params['seg_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        type: MOSCapType = self.params['type']
        fill_width: int = self.params['fill_width']
        fill_height: int = self.params['fill_height']
        target_area: int = self.params['target_area']
        num_rows: int = self.params['num_rows']

        seg_dict = {}
        if pinfo is not None:  # pinfo is given
            pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
            if isinstance(pinfo, tuple):
                pinfo = pinfo[0].get_tile_pinfo(0)
            num_rows = pinfo.num_rows
            # TODO: account for type
            if seg_p < 0 or seg_n < 0:
                seg_p = seg
                seg_n = seg

            if seg_p > 0 and seg_n > 0:  # parameters explicitly defined
                for n in range(num_rows):
                    row_type = pinfo.get_row_place_info(n).row_info.row_type
                    if type & 1 and row_type == MOSType.nch:
                        seg_dict[n] = seg_n
                    elif type & 2 and row_type == MOSType.pch:
                        seg_dict[n] = seg_p
            elif target_area > 0:  # pinfo is given, determine parameters
                avail_rows = 0
                for n in range(num_rows):
                    row_type = pinfo.get_row_place_info(n).row_info.row_type
                    if (type & 1 and row_type == MOSType.nch) or (type & 2 and row_type == MOSType.pch):
                        avail_rows += 1

                cap_per_row = target_area / avail_rows
                for n in range(num_rows):
                    row_type = pinfo.get_row_place_info(n).row_info.row_type
                    if (type & 1 and row_type == MOSType.nch) or (type & 2 and row_type == MOSType.pch):
                        row_w = pinfo.get_row_place_info(n).row_info.width
                        seg_dict[n] = int(np.round(cap_per_row / row_w))
            else:
                raise RuntimeError("pinfo given, but could not determine dimensions")

            return pinfo, seg_dict
        else:  # pinfo is not known, build our own
            if target_area < 0:
                raise RuntimeError("No pinfo given, no target area given. Cannot determine cap")

            # some parameters defined
            # we assume if w was defined, then pinfo would have existed
            if num_rows > 0 or seg > 0:
                if num_rows > 0 and seg > 0:
                    w = int(np.round(target_area / num_rows / seg))
                elif num_rows > 0:
                    total_w = target_area / num_rows
                    w = _determine_unit_width(total_w, min=False)
                    seg = int(np.round(total_w / w))
                elif seg > 0:
                    total_w = target_area / seg
                    w = _determine_unit_width(total_w, min=False)
                    num_rows = int(np.round(total_w / w))
            else:
                # Calculate parameters using dimension constraints
                if fill_width > 0 and fill_height > 0:
                    raise RuntimeError("Only one of fill_width and fill_height should be positive")
                if fill_width < 0 and fill_height < 0:
                    raise RuntimeError("One of fill_width and fill_height should be positive")

                if fill_width > 0:
                    seg, w, num_rows = self.fill_cap_by_width(fill_width, target_area)
                if fill_height > 0:
                    seg, w, num_rows = self.fill_cap_by_height(fill_height, target_area)

            # create pinfo
            pinfo = _create_moscap_pinfo_spec(type, num_rows, w, self.params['arr_info'])
            pinfo = MOSBasePlaceInfo.make_place_info(self.grid, pinfo)
            if isinstance(pinfo, tuple):
                pinfo = pinfo[0].get_tile_pinfo(0)
            seg_dict = {}
            for n in range(num_rows):
                seg_dict[n] = seg
            return pinfo, seg_dict

    def fill_cap_by_width(self, width_unit, target_area):
        """Given width and area target, fill area with MOSCap

        Parameters
        ----------
        width_unit: width to fill in grid units
        target_area: target cap area, in units of total_seg * width
        """
        # get unit_width
        width_unit_per_seg =  self._test_bbox.w / self._test_seg
        seg_targ = int(np.floor(width_unit / width_unit_per_seg))
        if seg_targ < 1:
            raise RuntimeError(f"Not enough width to draw MOSCap. Need at least {width_unit_per_seg}")
        w_fill = target_area / seg_targ

        # get target area
        w_targ = _determine_unit_width(w_fill)
        rows_targ = int(np.round(w_fill / w_targ))

        return seg_targ, w_targ, rows_targ

    def fill_cap_by_height(self, height_unit, target_area):
        """Given height and area target, fill area with MOSCap

        Parameters
        ----------
        height_unit: height to fill in grid units
        target_area: target cap area, in units of total_seg * width
        """
        # get unit_width
        height_unit_per_row = self._test_bbox.h
        rows_targ = int(np.floor(height_unit / height_unit_per_row))
        if rows_targ < 1:
            raise RuntimeError(f"Not enough vertical room to draw MOSCap. Need at least {height_unit_per_row}")
        w_fill = target_area / rows_targ

        # get target area
        w_targ = _determine_unit_width(w_fill, min=False)
        seg_targ = int(np.round(w_fill / w_targ))

        return seg_targ, w_targ, rows_targ


def _get_row_info(mos_type, width, flip=False, shared=False):
    global_info = get_tech_global_info('bag3_analog')
    threshold = global_info['thresholds'][0]
    if flip:
        bot_wires = dict(data=['sup', 'sig'])
        if shared:
            bot_wires['shared'] = ['sup']
        top_wires = ['sig']
    else:
        top_wires = dict(data=['sig', 'sup'])
        if shared:
            top_wires['shared'] = ['sup']
        bot_wires = ['sig']
    return dict(
        mos_type=mos_type,
        width=width,
        threshold=threshold,
        bot_wires=bot_wires,
        top_wires=top_wires,
        flip=flip,
    )


# Creates a new dictionary for TileInfoTable
def _create_moscap_pinfo_spec(type: MOSCapType, num_rows: int, w: int, arr_info: Mapping):
    nmos_rinfo = _get_row_info('nch', w, flip=False, shared=True)
    nmos_rinfo_flip = _get_row_info('nch', w, flip=True, shared=True)
    pmos_rinfo = _get_row_info('pch', w, flip=False, shared=True)
    pmos_rinfo_flip = _get_row_info('pch', w, flip=True, shared=True)

    row_specs = []
    for n in range(num_rows):
        # determine to draw n or p
        if type == MOSCapType.NMOS:
            new_row = nmos_rinfo if (n % 2 == 1) else nmos_rinfo_flip
        elif type == MOSCapType.PMOS:
            new_row = pmos_rinfo if (n % 2 == 1) else pmos_rinfo_flip
        else:
            if n % 4 == 0: new_row = nmos_rinfo_flip
            if n % 4 == 1: new_row = pmos_rinfo
            if n % 4 == 2: new_row = pmos_rinfo_flip
            if n % 4 == 3: new_row = nmos_rinfo
        row_specs.append(new_row)
    cap_tile=dict(priority=2, bot_mirror=False, top_mirror=False, row_specs=row_specs)
    place_info = dict(cap_tile=cap_tile)
    tile_specs = dict(arr_info=arr_info, place_info=place_info, abut_list=[])
    tiles=[dict(name='cap_tile')]
    return dict(tiles=tiles, tile_specs=tile_specs)


def _determine_unit_width(w_tot, min=True):
    """Given a total width, determine which unit width is the closest to being a multiple
    If two widths are both multiples or equally close, min flag determines whether to select
    the smallest width or largest width
    """
    global_info = get_tech_global_info('bag3_analog')
    w_minn = global_info['w_minn']
    w_maxn = global_info['w_maxn']  # TODO: What if PDK allows for bigger?

    # get target area
    if min:
        w_range = np.arange(w_minn, w_maxn + 1)
    else:
        w_range = np.flip(np.arange(w_minn, w_maxn + 1))
    w_err = np.round(w_tot / w_range) * w_range - w_tot
    w_opt = w_range[np.argmin(np.abs(w_err))]
    return w_opt


class MOSDUT(MOSBase):
    """A test class for getting the bound box of a single MOS device
    """
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        pass

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            test_seg='test number of segments'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            test_seg=4
        )

    def draw_layout(self) -> None:
        test_seg = self.params['test_seg']

        global_info = get_tech_global_info('bag3_analog')
        w_minn = global_info['w_minn']  # TODO: parametrize which gets used as input?
        lch = global_info['lch_min']
        top_layer = 4
        row_info = [_get_row_info('nch', w_minn, flip=True)]
        pinfo_dict = dict(
            lch=lch,
            top_layer=top_layer,
            row_specs=row_info,
            tr_widths=dict(
                sup={2: 2},
                sig={2: 1},
            ),
            tr_spaces={},
        )
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, pinfo_dict)
        self.draw_base(pinfo)
        self.add_mos(0, 0, seg=test_seg, w=w_minn)
        self.set_mos_size()


class VariableCapChain(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        print("VariableCapChain has no schematic!")
        pass

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            min_seg='min segments for each passgate',
            show_pins='True to show pins',
            flip_tile='True to flip all tiles',
            sig_locs='Signal locations for top horizontal metal layer pins',
            vertical_in='True to draw input on vertical metal layer.',
            vertical_sup='True to have supply unconnected on conn_layer.',
            type='MOSCap Flavor',
            target_cap_area='target MOS area, in units of total segments * width',
            fill_height='height unit constraint to set cap fill',
            cap_tiles='Tile indexes for moscaps',
            logic_tile='Tile index for passgate',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            min_seg=0,
            show_pins=False,
            flip_tile=False,
            sig_locs={},
            vertical_in=False,
            vertical_sup=False,
            type=MOSCapType.BOTH,
        )

    def draw_layout(self) -> None:

        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        ridx_p: int = self.params['ridx_p']  # TODO: pass in properly
        ridx_n: int = self.params['ridx_n']
        min_seg: int = self.params['min_seg']
        sig_locs: Dict[str, TrackType] = self.params['sig_locs']
        vertical_in: bool = self.params['vertical_in']
        vertical_sup: bool = self.params['vertical_sup']
        type: MOSCapType = self.params['type']
        target_cap_area: int = self.params['target_cap_area']
        fill_height: int = self.params['fill_height']
        cap_tiles: List[int] = self.params['cap_tiles']
        logic_tile: int = self.params['logic_tile']

        num_caps = len(cap_tiles)
        if min_seg == 0:
            global_info = get_tech_global_info('bag3_analog')
            min_seg = global_info['seg_min']

        # figure out cap
        # assume fill_height is given for just top or bottom, and is symmetric for top and bottom
        # TODO: this is probably a bad assumption
        fill_caps = num_caps / 2
        targ_height = fill_height / fill_caps

        cap_masters: List[MOSCapUnit] = []
        for tile_i in cap_tiles:
            cap_params=dict(
                pinfo=self.get_tile_pinfo(tile_i),
                target_area=target_cap_area,
            )
            cap_masters.append(self.new_template(MOSCapUnit, params=cap_params))
        max_seg = max([master.sch_params['seg_p'] for master in cap_masters])

        logic_pinfo = self.get_tile_pinfo(logic_tile)
        min_sep = self.min_sep_col
        pg_seg_tot = max_seg - min_sep * (num_caps - 1)
        pg_seg = int(np.round(pg_seg_tot / num_caps))
        pg_seg = max(pg_seg, min_seg)
        # TODO: support different n and p seg?
        pg_params=dict(
            pinfo=logic_pinfo,
            seg=pg_seg,
            w_p=w_p,
            w_n=w_n,
            ridx_p=ridx_p,
            ridx_n=ridx_n,
            vertical_out=True,
        )
        _pg_master = self.new_template(PassGateCore, params=pg_params)
        pg0 = _pg_master.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=-1)
        pg1 = _pg_master.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=0)

        sig_locs_e = dict(s=pg0)
        pg_params = dict(
            pinfo=logic_pinfo,
            seg=pg_seg,
            w_p=w_p,
            w_n=w_n,
            ridx_p=ridx_p,
            ridx_n=ridx_n,
            sig_locs=sig_locs_e,
            vertical_out=True,
        )
        pg_master_e = self.new_template(PassGateCore, params=pg_params)
        sig_locs_o = dict(s=pg1)
        pg_params = dict(
            pinfo=logic_pinfo,
            seg=pg_seg,
            w_p=w_p,
            w_n=w_n,
            ridx_p=ridx_p,
            ridx_n=ridx_n,
            sig_locs=sig_locs_o,
            vertical_out=True,
        )
        pg_master_o = self.new_template(PassGateCore, params=pg_params)

        # TODO: route so that horizontal routing is used to connect?
        cur_col = 0
        pg_insts = []
        for n in range(num_caps):
            pg_master = pg_master_o if n % 2 else pg_master_e
            pg_insts.append(self.add_tile(pg_master, logic_tile, cur_col))
            cur_col += pg_seg + min_sep

        cap_insts = []
        for n, tile_i in enumerate(cap_tiles):
            cap_insts.append(self.add_tile(cap_masters[n], tile_i, 0))

        self.set_mos_size()

        # route caps to passgates
        for i in range(num_caps):
            pg_out = pg_insts[i].get_pin("d")
            cap_in = cap_insts[i].get_pin("in")
            self.connect_to_track_wires(pg_out, cap_in)

        # connect passgates
        for i in range(num_caps - 1):
            if i == 0:
                self.reexport(pg_insts[i].get_port('s'), net_name='in')

            pg_i_out = pg_insts[i].get_pin("d")
            pg_i1_in = pg_insts[i+1].get_pin("s")
            self.connect_to_track_wires(pg_i_out, pg_i1_in)

            self.add_pin(f'en<{i}>', pg_insts[i].get_pin('en'))
            self.add_pin(f'enb<{i}>', pg_insts[i].get_pin('enb'))

        # reexport supplies
        vdd_list = []
        vss_list = []
        for cap_inst in cap_insts:
            vdd_list.extend(cap_inst.port_pins_iter('VDD'))
            vss_list.extend(cap_inst.port_pins_iter('VSS'))
        self.add_pin('VDD', vdd_list, connect=True)
        self.add_pin('VSS', vss_list, connect=True)

        # TODO: schematic parameters


class SplitPassGate(MOSBase):
    """
    Separate NMOS switch and PMOS switch in one logic tile
    Both switches have the same input, but split outputs
    """
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        print("GatedCapUnit has no schematic!")
        pass

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='Number of segments.',
            seg_p='segments of pmos',
            seg_n='segments of nmos',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            show_pins='True to show pins',
            sig_locs='Signal locations for top horizontal metal layer pins',
            g_on_s='',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            seg=-1,
            seg_p=-1,
            seg_n=-1,
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            show_pins=False,
            sig_locs={},
            g_on_s=False,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg: int = self.params['seg']
        seg_p: int = self.params['seg_p']
        seg_n: int = self.params['seg_n']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        g_on_s: bool = self.params['g_on_s']
        sig_locs: Mapping[str, Union[float, HalfInt]] = self.params['sig_locs']

        if seg_p <= 0:
            seg_p = seg
        if seg_n <= 0:
            seg_n = seg
        if seg_p <= 0 or seg_n <= 0:
            raise ValueError('Invalid segments.')

        hm_layer = self.conn_layer + 1

        pports = self.add_mos(ridx_p, 0, seg_p, w=w_p, g_on_s=g_on_s)
        nports = self.add_mos(ridx_n, 0, seg_n, w=w_n, g_on_s=g_on_s)
        self.set_mos_size()

        # VDD/VSS wires
        # xr = self.bound_box.xh
        # ns_tid = self.get_track_id(ridx_n, MOSWireType.DS_GATE, wire_name='sup')
        # ps_tid = self.get_track_id(ridx_p, MOSWireType.DS_GATE, wire_name='sup')
        # vss = self.add_wires(hm_layer, ns_tid.base_index, 0, xr, width=ns_tid.width)
        # vdd = self.add_wires(hm_layer, ps_tid.base_index, 0, xr, width=ps_tid.width)
        # self.add_pin('VDD', vdd)
        # self.add_pin('VSS', vss)

        # input will connect on the track id aligned with transistor's source
        tr_manager = self.tr_manager
        tr_w_h = tr_manager.get_width(hm_layer, 'sig')
        en_tidx = sig_locs.get('en', self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig'))
        enb_tidx = sig_locs.get('enb', self.get_track_index(ridx_p, MOSWireType.G,
                                                            wire_name='sig', wire_idx=-1))

        inn_tidx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=-1)
        inp_tidx = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=0)
        in_tidx = sig_locs.get('in', self.grid.get_middle_track(inn_tidx, inp_tidx))

        en_warr = self.connect_to_tracks(nports.g, TrackID(hm_layer, en_tidx, width=tr_w_h))
        enb_warr = self.connect_to_tracks(pports.g, TrackID(hm_layer, enb_tidx, width=tr_w_h))

        if g_on_s:
            self.add_pin('in', self.connect_to_tracks([nports.d, pports.d],
                                                      TrackID(hm_layer, in_tidx, width=tr_w_h)))
            self.add_pin('outn', nports.s)
            self.add_pin('outp', pports.s)
        else:
            self.add_pin('in', self.connect_to_tracks([nports.s, pports.s],
                                                     TrackID(hm_layer, in_tidx, width=tr_w_h)))
            self.add_pin('outn', nports.d)
            self.add_pin('outp', pports.d)

        self.add_pin('en', en_warr)
        self.add_pin('enb', enb_warr)

        # TODO: schematic parameters


class VariableCapChain2(MOSBase):
    """
    Switch and MOSCap chain with split N and P directions
    Draws chain in a single stack, as narrow as possible
    Number of segments determined by capacitor target area
    """
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return bag3_analog__variablecapchain2

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            min_seg='min segments for each passgate',
            show_pins='True to show pins',
            flip_tile='True to flip all tiles',
            sig_locs='Signal locations for top horizontal metal layer pins',
            type='MOSCap Flavor',
            target_cap_area='target MOS area, in units of total segments * width',
            fill_height='height unit constraint to set cap fill',
            logic_tile='Tile index for passgate',
            num_en='number of enable bits to control',
            g_on_s='Draw gate on source instead of on drain',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            min_seg=0,
            show_pins=False,
            flip_tile=False,
            sig_locs={},
            type=MOSCapType.BOTH,
            num_en=0,
            g_on_s=False,
        )

    def draw_layout(self) -> None:
        """
        Assumptions
        - pinfo has 1 logic tile. The rest of the tiles either single nch tiles or pch tiles
        - The logic tile is used as separate n and p gates
        - The passgate / n/p gate for a given cap will be in the tile directly above on the n-side,
          and directly below on the p-side
        - The first enable bit goes to the caps immediately outside the logic tile. The subsequent
          enable bits connect increasingly outward from the logic tile.
        """

        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        ridx_p: int = self.params['ridx_p']  # TODO: pass in properly
        ridx_n: int = self.params['ridx_n']
        min_seg: int = self.params['min_seg']
        sig_locs: Dict[str, TrackType] = self.params['sig_locs']
        type: MOSCapType = self.params['type']
        target_cap_area: int = self.params['target_cap_area']
        fill_height: int = self.params['fill_height']
        logic_tile: int = self.params['logic_tile']
        num_en: int = self.params['num_en']
        g_on_s: bool = self.params['g_on_s']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1

        # TODO: support min segments
        if min_seg == 0:
            global_info = get_tech_global_info('bag3_analog')
            min_seg = global_info['seg_min']

        # Determine how many caps we will draw
        _pinfo = pinfo[0] if isinstance(pinfo, tuple) else pinfo
        num_tiles = _pinfo.num_tiles
        assert logic_tile < num_tiles
        caps_avail = (((num_tiles - 1) // 2) + 1) // 2
        if num_en > 0 and num_en > caps_avail:
            raise ValueError(f"Cannot draw {4*num_en - 1} tiles in the given {num_tiles} tiles. "
                             f"Passgate drawn next to cap currently not supported")
        if num_en == 0:  # draw as many as we can
            num_en = caps_avail

        # Create masters
        cap_tiles = [logic_tile + (n * 2 + 1) for n in range(num_en)]
        cap_tiles += [logic_tile - (n * 2 + 1) for n in range(num_en)]
        cap_masters: Mapping[int, MOSCapUnit] = {}
        for tile_i in cap_tiles:
            cap_params = dict(
                pinfo=self.get_tile_pinfo(tile_i),
                target_area=target_cap_area,
                g_on_s=g_on_s,
            )
            cap_masters[tile_i] = self.new_template(MOSCapUnit, params=cap_params)
        max_seg = cap_masters[logic_tile - 1].sch_params['rows_n'][0]['seg'] # TODO: fix

        # Verify logic pinfo
        logic_pinfo = self.get_tile_pinfo(logic_tile)
        assert logic_pinfo.num_rows == 2
        _types = [logic_pinfo.get_row_place_info(i).row_info.row_type for i in [0, -1]]
        assert MOSType.pch in _types and MOSType.nch in _types

        pg_seg = max_seg
        # TODO: support different n and p seg?
        pg_params = dict(
            pinfo=logic_pinfo,
            seg=pg_seg,
            w_p=w_p,
            w_n=w_n,
            ridx_p=ridx_p,
            ridx_n=ridx_n,
            vertical_out=True,
            g_on_s=g_on_s,
        )
        pg_master = self.new_template(SplitPassGate, params=pg_params)
        pg_inst = self.add_tile(pg_master, logic_tile, 0)

        # draw instances
        cap_insts = {}
        switch_ports = {}
        for cap_tile_id in cap_tiles:
            cap_insts[cap_tile_id] = self.add_tile(cap_masters[cap_tile_id], cap_tile_id, 0)
            if not (cap_tile_id + 1 == logic_tile or cap_tile_id - 1 == logic_tile):
                switch_ind = cap_tile_id + 1 if cap_tile_id < logic_tile else cap_tile_id - 1
                switch_ports[cap_tile_id] = self.add_mos(0, 0, seg=max_seg, tile_idx=switch_ind,
                                                         g_on_s=g_on_s)

        self.set_mos_size()

        # Routing
        tr_manager = self.tr_manager
        tr_w_h = tr_manager.get_width(hm_layer, 'sig')
        tr_w_v = tr_manager.get_width(vm_layer, 'sig')

        # Connect Logic tile to caps
        self.connect_to_track_wires(pg_inst.get_pin('outn'), cap_insts[logic_tile-1].get_pin('in'))
        self.connect_to_track_wires(pg_inst.get_pin('outp'), cap_insts[logic_tile+1].get_pin('in'))

        # Connect individual gates to caps
        for cap_tile_id, cap_inst in cap_insts.items():
            if cap_tile_id + 1 == logic_tile or cap_tile_id - 1 == logic_tile:
                continue
            gate_ports = switch_ports[cap_tile_id]
            if g_on_s:
                self.connect_to_track_wires(gate_ports.s, cap_inst.get_pin('in'))
                self.connect_to_track_wires(gate_ports.s, cap_inst.get_pin('in'))
            else:
                self.connect_to_track_wires(gate_ports.d, cap_inst.get_pin('in'))
                self.connect_to_track_wires(gate_ports.d, cap_inst.get_pin('in'))

        self.reexport(pg_inst.get_port('in'), net_name='in')

        # Define Enables pins and route between stages
        for idx in range(1, num_en):
            # Define en_i
            gate_tile_id = logic_tile - (idx * 2)
            gate_ports = switch_ports[gate_tile_id - 1]
            en_i_tidx = self.get_track_index(0, MOSWireType.G, 'sig', wire_idx=0,
                                             tile_idx=gate_tile_id)
            en = self.connect_to_tracks(gate_ports.g, TrackID(hm_layer, en_i_tidx, tr_w_h))
            self.add_pin(f'en<{idx}>', en)

            # Connect to previous stage
            # TODO: BAG3 currently does not support dual gates
            in_hm_tidx = self.get_track_index(0, MOSWireType.DS, 'sig', tile_idx=gate_tile_id)
            in_port = gate_ports.d if g_on_s else gate_ports.s
            in_hm = self.connect_to_tracks(in_port, TrackID(hm_layer, in_hm_tidx, tr_w_h))
            in_vm_tidx = self.grid.coord_to_track(vm_layer, in_hm.middle)
            self.connect_to_tracks([in_hm, cap_insts[gate_tile_id + 1].get_pin('in')],
                                   TrackID(vm_layer, in_vm_tidx, tr_w_v))

            # Define enb_i
            gate_tile_id = logic_tile + (idx * 2)
            gate_ports = switch_ports[gate_tile_id + 1]
            enb_i_tidx = self.get_track_index(0, MOSWireType.G, 'sig', wire_idx=-1,
                                              tile_idx=gate_tile_id)
            enb = self.connect_to_tracks(gate_ports.g, TrackID(hm_layer, enb_i_tidx, tr_w_h))
            self.add_pin(f'enb<{idx}>', enb)

            # Connect to previous stage
            # TODO: BAG3 currently does not support dual gates
            in_hm_tidx = self.get_track_index(0, MOSWireType.DS, 'sig', tile_idx=gate_tile_id)
            in_port = gate_ports.d if g_on_s else gate_ports.s
            in_hm = self.connect_to_tracks(in_port, TrackID(hm_layer, in_hm_tidx, tr_w_h))
            in_vm_tidx = self.grid.coord_to_track(vm_layer, in_hm.middle)
            self.connect_to_tracks([in_hm, cap_insts[gate_tile_id - 1].get_pin('in')],
                                   TrackID(vm_layer, in_vm_tidx, tr_w_v))

        self.reexport(pg_inst.get_port('en'), net_name='en<0>')
        self.reexport(pg_inst.get_port('enb'), net_name='enb<0>')

        # reexport supplies
        vdd_list = []
        vss_list = []
        for cap_inst in cap_insts.values():
            vdd_list.extend(cap_inst.port_pins_iter('VDD'))
            vss_list.extend(cap_inst.port_pins_iter('VSS'))
        self.add_pin('VDD', vdd_list, connect=True)
        self.add_pin('VSS', vss_list, connect=True)

        # schematic parameters
        p_gates = []
        n_gates = []
        p_caps = []
        n_caps = []
        p_gates.append(dict(
            seg=pg_seg,
            w=w_p,
            th=logic_pinfo.get_row_place_info(-1).row_info.threshold,
        ))
        n_gates.append(dict(
            seg=pg_seg,
            w=w_n,
            th=logic_pinfo.get_row_place_info(0).row_info.threshold,
        ))
        p_caps.append(cap_masters[logic_tile + 1].sch_params)
        n_caps.append(cap_masters[logic_tile - 1].sch_params)
        idx = 2
        while logic_tile + idx + 1 < num_tiles:
            cap_tile = logic_tile + idx + 1
            gate_row_info = _pinfo.get_tile_pinfo(cap_tile - 1).get_row_place_info(0).row_info
            p_gates.append(dict(
                seg=max_seg,
                w=gate_row_info.width,
                th=gate_row_info.threshold,
            ))
            p_caps.append(cap_masters[cap_tile].sch_params)

            cap_tile = logic_tile - idx - 1
            gate_row_info = _pinfo.get_tile_pinfo(cap_tile + 1).get_row_place_info(0).row_info
            n_gates.append(dict(
                seg=max_seg,
                w=gate_row_info.width,
                th=gate_row_info.threshold,
            ))
            n_caps.append(cap_masters[cap_tile].sch_params)
            idx += 2

        self.sch_params = dict(
            lch=_pinfo.arr_info.lch,
            p_gates=p_gates,
            n_gates=n_gates,
            p_caps=p_caps,
            n_caps=n_caps,
        )
