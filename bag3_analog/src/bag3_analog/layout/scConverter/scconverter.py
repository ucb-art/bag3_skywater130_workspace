"""This module contains layout generators for switch-capacitor converter."""

from typing import Any, Dict, Sequence, cast, Type
import sys

from pybag.enum import MinLenMode, RoundMode, Orientation
from pybag.core import BBox, Transform

from bag.typing import TrackType
from bag.util.immutable import Param
from bag.layout.template import TemplateDB, TemplateType, TemplateBase
from bag.layout.routing.base import TrackID, TrackManager, WDictType, SpDictType
from bag.util.math import HalfInt

from xbase.layout.enum import MOSWireType, MOSPortType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase
from xbase.layout.mos.top import MOSBaseWrapper

from ...enum import DrawTaps


class SCConverterUnit(MOSBase):
    """The core of switch capacitor converter"""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='Dictionary of segments',
            ridx_psw0='index for pmos row with p-switch connect to vdd_core',
            ridx_nsw0='index for nmos row with n-switch connect to ground',
            ridx_nsw1='index for pmos row with n-switch connect to cap top',
            ridx_nsw2='index for pmos row with n-switch connect to cap bottom',
            show_pins='True to show pins',
            flip_tile='True to flip all tiles',
            draw_taps='LEFT or RIGHT or BOTH or NONE',
            sig_locs='Signal locations for top horizontal metal layer pins',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            ridx_psw0=-1,
            ridx_nsw0=2,
            ridx_nsw1=1,
            ridx_nsw2=0,
            show_pins=False,
            flip_tile=False,
            draw_taps='NONE',
            sig_locs={},
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo, flip_tile=self.params['flip_tile'])

        seg_dict: Dict[str, int] = self.params['seg_dict']
        ridx_psw0: int = self.params['ridx_psw0']
        ridx_nsw0: int = self.params['ridx_nsw0']
        ridx_nsw1: int = self.params['ridx_nsw1']
        ridx_nsw2: int = self.params['ridx_nsw2']
        draw_taps: DrawTaps = DrawTaps[self.params['draw_taps']]
        sig_locs: Dict[str, TrackType] = self.params['sig_locs']

        for key, val in seg_dict.items():
            if val % 2:
                raise ValueError(f'This generator does not support odd number of segments '
                                 f'{key} = {val}')

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1

        seg_psw0 = seg_dict['psw0']
        seg_nsw0 = seg_dict['nsw0']
        seg_nsw1 = seg_dict['nsw1']
        seg_nsw2 = seg_dict['nsw2']

        # === calculate taps information ===
        tap_n_cols = self.get_tap_ncol()
        sub_sep = self.sub_sep_col
        num_taps = 0
        tap_offset = 0
        tap_left = tap_right = False

        if draw_taps in DrawTaps.RIGHT | DrawTaps.BOTH:
            num_taps += 1
            tap_right = True
        if draw_taps in DrawTaps.LEFT | DrawTaps.BOTH:
            num_taps += 1
            tap_left = True
            tap_offset += tap_n_cols + sub_sep

        # === get enough space for vertical routing. ===
        num, locs = self.tr_manager.place_wires(vm_layer, ['clk', 'clk', 'clk',
                                                           'sig', 'sig', 'sig'])
        sig_vm_w = self.tr_manager.get_width(vm_layer, 'sig')
        clk_vm_w = self.tr_manager.get_width(vm_layer, 'clk')
        vm_coord = self.grid.get_wire_bounds(vm_layer, locs[-1], width=sig_vm_w)[1]
        vm_col = pinfo.coord_to_col(vm_coord, RoundMode.GREATER_EQ)
        vm_col += vm_col & 1

        # === set total width ===
        seg_max = max(seg_psw0, seg_nsw0, seg_nsw1, seg_nsw2, vm_col)
        seg_tot = seg_max + (tap_n_cols + sub_sep) * num_taps
        self.set_mos_size(seg_tot)  # set total size

        # === Placement ===
        # = transistors
        nsw2_inst = self.add_mos(ridx_nsw2, tap_offset + (seg_max - seg_nsw2) // 2, seg_nsw2)
        _row_info = pinfo.get_row_place_info(ridx_nsw2).row_info
        w_nsw2, th_nsw2 = _row_info.width, _row_info.threshold
        nsw1_inst = self.add_mos(ridx_nsw1, tap_offset + (seg_max - seg_nsw1) // 2, seg_nsw1)
        _row_info = pinfo.get_row_place_info(ridx_nsw1).row_info
        w_nsw1, th_nsw1 = _row_info.width, _row_info.threshold
        nsw0_inst = self.add_mos(ridx_nsw0, tap_offset + (seg_max - seg_nsw0) // 2, seg_nsw0)
        _row_info = pinfo.get_row_place_info(ridx_nsw0).row_info
        w_nsw0, th_nsw0 = _row_info.width, _row_info.threshold
        psw0_inst = self.add_mos(ridx_psw0, tap_offset + (seg_max - seg_psw0) // 2, seg_psw0)
        _row_info = pinfo.get_row_place_info(ridx_psw0).row_info
        w_psw0, th_psw0 = _row_info.width, _row_info.threshold

        # = taps
        tap_vdd_list, tap_vss_list = [], []
        if tap_left:
            self.add_tap(0, tap_vdd_list, tap_vss_list)
        if tap_right:
            self.add_tap(seg_tot, tap_vdd_list, tap_vss_list, flip_lr=True)

        # === Routing ===
        # 1. connect source of psw0 to supply and source of nsw0 to ground
        vdd_tid = self.get_track_id(ridx_psw0, MOSWireType.DS, wire_name='sup')
        vdd = self.connect_to_tracks(psw0_inst[MOSPortType.D], vdd_tid)
        self.connect_to_track_wires(tap_vdd_list, vdd)

        vss_tid = self.get_track_id(ridx_nsw0, MOSWireType.DS, wire_name='sup')
        vss = self.connect_to_tracks(nsw0_inst[MOSPortType.D], vss_tid)
        self.connect_to_track_wires(tap_vss_list, vss)

        # 2. connect gate of switches to different clock phases
        psw0_g_tid = self.get_track_id(ridx_psw0, MOSWireType.G, wire_name='clk')
        clk_p1b_psw0 = self.connect_to_tracks(psw0_inst[MOSPortType.G], psw0_g_tid)
        nsw0_g_tid = self.get_track_id(ridx_nsw0, MOSWireType.G, wire_name='clk')
        clk_p2_nsw0 = self.connect_to_tracks(nsw0_inst[MOSPortType.G], nsw0_g_tid)
        nsw1_g_tid = self.get_track_id(ridx_nsw1, MOSWireType.G, wire_name='clk')
        clk_p2_nsw1 = self.connect_to_tracks(nsw1_inst[MOSPortType.G], nsw1_g_tid)
        nsw2_g_tid = self.get_track_id(ridx_nsw2, MOSWireType.G, wire_name='clk')
        clk_p1_nsw2 = self.connect_to_tracks(nsw2_inst[MOSPortType.G], nsw2_g_tid)

        # 3. connect source/drain of transistors
        # = connect to cap_top/cap_bot
        psw0_d_tid = self.get_track_id(ridx_psw0, MOSWireType.DS, wire_name='sig')
        psw0_cap_top = self.connect_to_tracks(psw0_inst[MOSPortType.S], psw0_d_tid)
        nsw0_d_tid = self.get_track_id(ridx_nsw0, MOSWireType.DS, wire_name='sig')
        nsw0_cap_bot = self.connect_to_tracks(nsw0_inst[MOSPortType.S], nsw0_d_tid)
        nsw1_d_tid = self.get_track_id(ridx_nsw1, MOSWireType.DS, wire_name='sig')
        nsw1_cap_top = self.connect_to_tracks(nsw1_inst[MOSPortType.D], nsw1_d_tid)
        nsw2_d_tid = self.get_track_id(ridx_nsw2, MOSWireType.DS, wire_name='sig')
        nsw2_cap_bot = self.connect_to_tracks(nsw2_inst[MOSPortType.D], nsw2_d_tid)
        # = connect to vout
        nsw1_s_tid = self.get_track_id(ridx_nsw1, MOSWireType.DS, wire_name='sig', wire_idx=1)
        nsw1_vout = self.connect_to_tracks(nsw1_inst[MOSPortType.S], nsw1_s_tid)
        nsw2_s_tid = self.get_track_id(ridx_nsw2, MOSWireType.DS, wire_name='sig', wire_idx=1)
        nsw2_vout = self.connect_to_tracks(nsw2_inst[MOSPortType.S], nsw2_s_tid)

        # 4. connect to vertical layer
        vm_mid = self.grid.coord_to_track(vm_layer, clk_p1b_psw0.middle, mode=RoundMode.NEAREST)
        try:
            loc_mid = (locs[0] + locs[-1]) / 2
        except ValueError:
            loc_mid = (locs[0] + locs[-1]) // 2 + HalfInt(1)
        vm_offset = vm_mid - loc_mid

        clk_p2_idx = locs[0] + vm_offset
        clk_p1_idx = locs[1] + vm_offset
        clk_p1b_idx = locs[2] + vm_offset
        cap_top_idx = locs[-1] + vm_offset
        cap_bot_idx = locs[-2] + vm_offset
        vout_idx = locs[-3] + vm_offset

        cap_top, cap_bot = self.connect_differential_tracks([psw0_cap_top, nsw1_cap_top],
                                                            [nsw0_cap_bot, nsw2_cap_bot], vm_layer,
                                                            cap_top_idx, cap_bot_idx,
                                                            width=sig_vm_w)
        clk_p2, clk_p1, clk_p1b = self.connect_matching_tracks(
            [[clk_p2_nsw0, clk_p2_nsw1], clk_p1_nsw2, clk_p1b_psw0],
            vm_layer, [clk_p2_idx, clk_p1_idx, clk_p1b_idx], width=clk_vm_w)

        vout = self.connect_to_tracks([nsw1_vout, nsw2_vout],
                                      TrackID(vm_layer, vout_idx, width=sig_vm_w))

        # 5. connect to horizontol
        out_idx = sig_locs.get('out', self.grid.coord_to_track(xm_layer, vout.middle,
                                                               mode=RoundMode.NEAREST))
        sig_width_xm = self.tr_manager.get_width(xm_layer, 'sig')
        clk_width_xm = self.tr_manager.get_width(xm_layer, 'clk')
        out = self.connect_to_tracks(vout, TrackID(xm_layer, out_idx, sig_width_xm),
                                     min_len_mode=MinLenMode.UPPER)

        cap_top_idx = sig_locs.get('cap_top', self.tr_manager.get_next_track(xm_layer, out_idx,
                                                                             'sig', 'sig', up=True))
        cap_bot_idx = sig_locs.get('cap_bot', self.tr_manager.get_next_track(xm_layer, cap_top_idx,
                                                                             'sig', 'sig', up=True))

        cap_top, cap_bot = self.connect_differential_tracks(cap_top, cap_bot, xm_layer,
                                                            cap_top_idx, cap_bot_idx,
                                                            width=sig_width_xm)

        clk_p2_idx = sig_locs.get('clk_p2', self.grid.coord_to_track(xm_layer, clk_p2.middle,
                                                                     mode=RoundMode.NEAREST))

        clk_p2 = self.connect_to_tracks(clk_p2, TrackID(xm_layer, clk_p2_idx, width=clk_width_xm),
                                        min_len_mode=MinLenMode.UPPER)

        clk_p1_idx = sig_locs.get('clk_p1', self.tr_manager.get_next_track(xm_layer, clk_p2_idx,
                                                                           'clk', 'clk', up=True))
        clk_p1b_idx = sig_locs.get('clk_p1b', self.tr_manager.get_next_track(xm_layer, clk_p1_idx,
                                                                             'clk', 'clk', up=True))
        clk_p1, clk_p1b = self.connect_differential_tracks(clk_p1, clk_p1b, xm_layer,
                                                           clk_p1_idx, clk_p1b_idx,
                                                           width=clk_width_xm)

        # add pins
        self.add_pin('VDD', vdd, label='VDD:', show=self.show_pins)
        self.add_pin('VSS', vss, label='VSS:', show=self.show_pins)

        self.add_pin('vout', out, show=self.show_pins)
        self.add_pin('cap_top', cap_top, show=self.show_pins)
        self.add_pin('cap_bot', cap_bot, show=self.show_pins)

        self.add_pin('clk_p1', clk_p1, show=self.show_pins)
        self.add_pin('clk_p2', clk_p2, show=self.show_pins)
        self.add_pin('clk_p1b', clk_p1b, show=self.show_pins)

        # set properties
        self.sch_params = dict(
            lch=pinfo.lch,
            seg_dict=dict(
                psw0=seg_psw0,
                nsw0=seg_nsw0,
                nsw1=seg_nsw1,
                nsw2=seg_nsw2,
            ),
            w_dict=dict(
                psw0=w_psw0,
                nsw0=w_nsw0,
                nsw1=w_nsw1,
                nsw2=w_nsw2,
            ),
            th_dict=dict(
                psw0=th_psw0,
                nsw0=th_nsw0,
                nsw1=th_nsw1,
                nsw2=th_nsw2,
            ),
        )


class SCConverterArray(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object for unit cell',
            seg_dict='Dictionary of segments',
            ridx_psw0='index for pmos row with p-switch connect to vdd_core',
            ridx_nsw0='index for nmos row with n-switch connect to ground',
            ridx_nsw1='index for pmos row with n-switch connect to cap top',
            ridx_nsw2='index for pmos row with n-switch connect to cap bottom',
            show_pins='True to show pins',
            flip_tile='True to flip all tiles',
            draw_taps='LEFT or RIGHT or BOTH or NONE',
            sig_locs='Signal locations for top horizontal metal layer pins',
            ncol='Number of column',
            nrow='Number of rows',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            ridx_psw0=-1,
            ridx_nsw0=2,
            ridx_nsw1=1,
            ridx_nsw2=0,
            show_pins=False,
            flip_tile=False,
            draw_taps='NONE',
            sig_locs={},
            ncol=1,
            nrow=1,
        )

    def _connect_wires_to_tracks(self, layer_id, width, tr_list, wire_list, upper=None, lower=None):
        upper_wire_list = []
        for _tidx in tr_list:
            _tid = TrackID(layer_id, _tidx, width=width)
            _wire = self.connect_to_tracks(wire_list, _tid, track_upper=upper,
                                           track_lower=lower)
            upper_wire_list.append(_wire)
        return upper_wire_list

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo, flip_tile=self.params['flip_tile'])

        seg_dict: Dict[str, int] = self.params['seg_dict']
        ridx_psw0: int = self.params['ridx_psw0']
        ridx_nsw0: int = self.params['ridx_nsw0']
        ridx_nsw1: int = self.params['ridx_nsw1']
        ridx_nsw2: int = self.params['ridx_nsw2']
        draw_taps: DrawTaps = DrawTaps[self.params['draw_taps']]
        sig_locs: Dict[str, TrackType] = self.params['sig_locs']
        ncol: int = self.params['ncol']
        nrow: int = self.params['nrow']
        top_layer: int = self.params['pinfo']['top_layer']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1
        show_sub_block_pins: bool = False
        # TODO: modify signal locations and vertical out?
        # pinfo_unit = dict(self.params['pinfo'])
        # pinfo_unit['top_layer'] = 4
        pinfo_unit = self.params['pinfo']
        sc_converter_unit_params = dict(
            pinfo=pinfo_unit,
            seg_dict=seg_dict,
            ridx_psw0=ridx_psw0,
            ridx_nsw0=ridx_nsw0,
            ridx_nsw1=ridx_nsw1,
            ridx_nsw2=ridx_nsw2,
            show_pins=show_sub_block_pins,
            sig_locs=sig_locs,
        )
        tr_manager = pinfo.tr_manager
        unit_master = self.new_template(SCConverterUnit, params=sc_converter_unit_params)
        unit_cell_col = unit_master.num_cols

        #  === Taps ===
        tap_n_cols = self.get_tap_ncol()
        tap_sep_col = self.sub_sep_col
        l_offset = tap_sep_col + tap_n_cols if draw_taps in DrawTaps.LEFT | DrawTaps.BOTH else 0
        r_offset = tap_sep_col + tap_n_cols if draw_taps in DrawTaps.RIGHT | DrawTaps.BOTH else 0
        unit_conn_col = unit_master.num_cols

        # === Calculation: have enough space for routing ===
        # - Calculate seperation needs for unit cell, decide by hm_layer le space or min_sep_col
        tr_w_h_max = max([tr_manager.get_width(hm_layer, sig_type) for sig_type in ['sig', 'clk']])
        sep_conn = max(self.min_sep_col, self.get_hm_sp_le_sep_col(tr_w_h_max))
        num_sup_side = 4  # set to even number
        num_sup_vm, sup_vm_locs = tr_manager.place_wires(vm_layer, ['sup'] * num_sup_side)
        # - Calculate seperation left for middle supply connections
        sup_vm_sep = self.grid.track_to_coord(vm_layer, sup_vm_locs[1]) - self.grid.track_to_coord(
            vm_layer, sup_vm_locs[0])
        sup_vm_col = - (-(num_sup_side * sup_vm_sep) // self.sd_pitch)
        sep_conn = max(sep_conn, sup_vm_col)
        # - Calculate width needs for top layer wire array
        top_vert_layer = top_layer if self.grid.get_direction(top_layer) == 'y' else top_layer - 1

        # - Top vertical layer arrange [vss, cap_top, cap_bot, vout, vdd], cap_top/bot not exported
        num_top_vert, top_vert_locs = tr_manager.place_wires(top_vert_layer, ['sup', 'sig', 'sig',
                                                                              'sig', 'sup', 'sup'])
        top_vert_sep = self.grid.track_to_coord(vm_layer,
                                                top_vert_locs[-1]) - self.grid.track_to_coord(
            vm_layer, top_vert_locs[0])
        top_vert_col = -(-top_vert_sep // self.sd_pitch)
        sep = max(sep_conn, top_vert_col - unit_cell_col)

        # - on the left/right will be 3 vm_layer track
        clk_width_vm = tr_manager.get_width(vm_layer, 'clk')
        sig_width_vm = tr_manager.get_width(vm_layer, 'sig')
        num_vm, locs_vm = tr_manager.place_wires(vm_layer, ['clk', 'clk'])
        vm_sep = self.grid.track_to_coord(vm_layer, locs_vm[-1]) - \
                 self.grid.track_to_coord(vm_layer, locs_vm[0])

        l_offset = max(l_offset, -(- (3 * vm_sep) // self.sd_pitch))
        r_offset = max(r_offset, -(- (3 * vm_sep) // self.sd_pitch))

        # === Placement ===
        # - Add unit cells
        start_col = l_offset + sep // 2
        num_cols = (unit_conn_col + sep) * ncol
        num_cols += l_offset + r_offset
        pitch = sep + unit_conn_col
        inst_arr = []
        for r in range(nrow):
            row_list = []
            cur_col = start_col
            for c in range(ncol):
                inst = self.add_tile(unit_master, r, cur_col)
                cur_col += pitch
                row_list.append(inst)
            inst_arr.append(row_list)
        self.set_mos_size(num_cols, nrow)

        # - Add taps
        tap_vdd_list, tap_vss_list = [], []
        if draw_taps in DrawTaps.LEFT | DrawTaps.BOTH:
            for i in range(nrow):
                self.add_tap(0, tap_vdd_list, tap_vss_list, tile_idx=i)
        if draw_taps in DrawTaps.RIGHT | DrawTaps.BOTH:
            for i in range(nrow):
                self.add_tap(num_cols, tap_vdd_list, tap_vss_list, flip_lr=True, tile_idx=i)

        # === Routing ===
        # - Supply
        vdd_list, vss_list = [], []
        for inst_row in inst_arr:
            for inst in inst_row:
                vdd_list += inst.get_all_port_pins('VDD')
                vss_list += inst.get_all_port_pins('VSS')
        vdd = self.connect_wires(vdd_list)
        vss = self.connect_wires(vss_list)
        vdd = self.connect_to_track_wires(tap_vdd_list, vdd)
        vss = self.connect_to_track_wires(tap_vss_list, vss)
        sup_vm_coord = [self.grid.track_to_coord(vm_layer, tidx) for tidx in sup_vm_locs]
        sup_vm_middle = (sup_vm_coord[0] + sup_vm_coord[-1])
        sup_vm_shift = int(sep * self.sd_pitch - sup_vm_middle) // 2
        mid_tidx_vm = []
        sup_tidx_vm = []
        for col in range(ncol + 1):
            _tidx = self.grid.coord_to_track(vm_layer,
                                             (l_offset - sep // 2 + col * pitch) * self.sd_pitch +
                                             sup_vm_shift, mode=RoundMode.LESS)
            mid_tidx_vm.append(_tidx)

        for tidx in mid_tidx_vm:
            sup_tidx_vm += [_loc + tidx for _loc in sup_vm_locs]

        # - Take all tidx except first and last half
        sup_tidx_vm = sup_tidx_vm[num_sup_side // 2:-num_sup_side // 2]
        vdd_tidx_vm = sup_tidx_vm[::2]
        vss_tidx_vm = sup_tidx_vm[1::2]
        sup_w_vm = tr_manager.get_width(vm_layer, 'sup')
        vdd_list_vm = self._connect_wires_to_tracks(vm_layer, sup_w_vm, vdd_tidx_vm,
                                                    vdd, upper=self.bound_box.yh,
                                                    lower=self.bound_box.yl)
        vss_list_vm = self._connect_wires_to_tracks(vm_layer, sup_w_vm, vss_tidx_vm,
                                                    vss, upper=self.bound_box.yh,
                                                    lower=self.bound_box.yl)

        # === Connect cap_top/cap_bot/vout ===
        # - Pins are on xm_layer
        cap_top_list = [inst.get_pin('cap_top') for row in inst_arr for inst in row]
        cap_bot_list = [inst.get_pin('cap_bot') for row in inst_arr for inst in row]
        vout_list = [inst.get_pin('vout') for row in inst_arr for inst in row]

        clk_p1_list = [inst.get_pin('clk_p1') for row in inst_arr for inst in row]
        clk_p1b_list = [inst.get_pin('clk_p1b') for row in inst_arr for inst in row]
        clk_p2_list = [inst.get_pin('clk_p2') for row in inst_arr for inst in row]

        cap_top_list, cap_bot_list, vout_list = \
            [self.connect_wires(w_list) for w_list in [cap_top_list, cap_bot_list, vout_list]]

        clk_p1_list, clk_p1b_list, clk_p2_list = \
            [self.connect_wires(w_list) for w_list in [clk_p1_list, clk_p1b_list, clk_p2_list]]

        # - connect clk signals
        clk_tidx_start = \
            self.grid.coord_to_track(vm_layer, l_offset * self.sd_pitch - int(vm_sep * 2.5),
                                     mode=RoundMode.NEAREST)
        clk_num_vm, clk_locs_vm = tr_manager.place_wires(vm_layer, ['clk', 'clk', 'clk'],
                                                         clk_tidx_start)
        clk_p1_vm, clk_p1b_vm, clk_p2_vm = \
            self.connect_matching_tracks([clk_p1_list, clk_p1b_list, clk_p2_list], vm_layer,
                                         list(clk_locs_vm), width=clk_width_vm)

        # = connect vout, cap_top/cap_bot signals
        out_tidx_start = self.grid.coord_to_track(vm_layer, (num_cols - r_offset) * self.sd_pitch +
                                                  int(vm_sep * 0.5), mode=RoundMode.NEAREST)
        out_num_vm, out_locs_vm = tr_manager.place_wires(vm_layer, ['sig', 'sig', 'sig'],
                                                         out_tidx_start)
        vout_vm, cap_top_vm, cap_bot_vm = \
            self.connect_matching_tracks([vout_list, cap_top_list, cap_bot_list], vm_layer,
                                         list(out_locs_vm), width=sig_width_vm)

        # - Connect supply to xm_layer
        clk_p1b_list = [y for x in clk_p1b_list for y in x.to_warr_list()]
        vout_list = [y for x in vout_list for y in x.to_warr_list()]
        num_sup_xm, sup_xm_locs = tr_manager.place_wires(xm_layer, ['sup'] * num_sup_side, 0)

        sup_tidx_xm = []
        for _row in range(nrow - 1):
            if _row % 2 == 0:
                sup_rowmid_tidx = clk_p1b_list[_row].track_id.base_index
                sup_xm_width = clk_p1b_list[_row + 1].track_id.base_index - clk_p1b_list[
                    _row].track_id.base_index
            else:
                sup_rowmid_tidx = vout_list[_row].track_id.base_index
                sup_xm_width = vout_list[_row + 1].track_id.base_index - vout_list[
                    _row].track_id.base_index
            if sup_xm_width < num_sup_xm:
                raise ValueError('Does not have enough xm layer supply routing space, '
                                 'reduce num_sup_side')
            start_idx = (sup_xm_width - num_sup_xm) // 2 + sup_rowmid_tidx
            sup_tidx_xm += [_loc + start_idx for _loc in sup_xm_locs]

        vdd_tidx_xm, vss_tidx_xm = sup_tidx_xm[::2], sup_tidx_xm[1::2]
        sup_w_xm = tr_manager.get_width(xm_layer, 'sup')
        vdd_list_xm = self._connect_wires_to_tracks(xm_layer, sup_w_xm, vdd_tidx_xm,
                                                    vdd_list_vm, upper=self.bound_box.xh,
                                                    lower=self.bound_box.xl)
        vss_list_xm = self._connect_wires_to_tracks(xm_layer, sup_w_xm, vss_tidx_xm,
                                                    vss_list_vm, upper=self.bound_box.xh,
                                                    lower=self.bound_box.xl)

        # === Connections from xm to higher ===
        num_top_vert, top_vert_locs = tr_manager.place_wires(top_vert_layer, ['sup', 'sig', 'sig',
                                                                              'sig', 'sup', 'sup'])
        vdd_tidx_top, vss_tidx_top = [], []
        vout_tidx_top = []
        for col in range(ncol):
            start_idx_vert = self.grid.coord_to_track(top_vert_layer,
                                                      (l_offset + col * pitch) * self.sd_pitch,
                                                      mode=RoundMode.NEAREST)
            vdd_tidx_top.append(start_idx_vert + top_vert_locs[0])
            vss_tidx_top.append(start_idx_vert + top_vert_locs[-2])
            vout_tidx_top.append(start_idx_vert + top_vert_locs[-3])

        sup_w_ym = tr_manager.get_width(ym_layer, 'sup')
        vdd_list_ym = self._connect_wires_to_tracks(ym_layer, sup_w_ym, vdd_tidx_top,
                                                    vdd_list_xm, upper=self.bound_box.yh,
                                                    lower=self.bound_box.yl)
        vss_list_ym = self._connect_wires_to_tracks(ym_layer, sup_w_ym, vss_tidx_top,
                                                    vss_list_xm, upper=self.bound_box.yh,
                                                    lower=self.bound_box.yl)
        sup_out_ym = tr_manager.get_width(ym_layer, 'sig')
        out_list_vm = self._connect_wires_to_tracks(ym_layer, sup_out_ym, vout_tidx_top,
                                                    vout_list, upper=self.bound_box.yh,
                                                    lower=self.bound_box.yl)

        # === Add pins ===
        self.add_pin('clk_p1', clk_p1_vm)
        self.add_pin('clk_p1b', clk_p1b_vm)
        self.add_pin('clk_p2', clk_p2_vm)

        self.add_pin('vout', vout_vm)
        # self.add_pin('vout', out_list_vm)
        self.add_pin('cap_top', cap_top_vm)
        self.add_pin('cap_bot', cap_bot_vm)

        self.add_pin('VDD', vdd)
        # self.add_pin('VDD', vdd_list_vm)
        self.add_pin('VSS', vss)
        # self.add_pin('VSS', vss_list_ym)
        self.sch_params = dict(
            ncol=ncol,
            nrow=nrow,
            unit_sch_params=unit_master.sch_params
        )


class SCConverterUnitWrapper(MOSBaseWrapper):
    """Wrapper for SC_Converter Unit Cell """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBaseWrapper.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return SCConverterUnit.get_params_info()

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return SCConverterUnit.get_default_param_values()

    def draw_layout(self) -> None:
        show_pins: bool = self.show_pins
        core_params: Param = self.params

        core_params = core_params.copy(append=dict(show_pins=False))
        master = self.new_template(SCConverterUnit, params=core_params)

        inst = self.draw_boundaries(master, master.top_layer)

        # re-export pins
        for name in inst.port_names_iter():
            self.reexport(inst.get_port(name), show=show_pins)

        # set properties
        self.sch_params = master.sch_params


class SCConverterArrayWrapper(MOSBaseWrapper):
    """Wrapper for SC Converter Array"""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBaseWrapper.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return SCConverterArray.get_params_info()

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return SCConverterArray.get_default_param_values()

    def draw_layout(self) -> None:
        show_pins: bool = self.show_pins
        core_params: Param = self.params

        core_params = core_params.copy(append=dict(show_pins=False))
        master = self.new_template(SCConverterArray, params=core_params)

        inst = self.draw_boundaries(master, master.top_layer)

        # re-export pins
        for name in inst.port_names_iter():
            self.reexport(inst.get_port(name), show=show_pins)

        # set properties
        self.sch_params = master.sch_params
