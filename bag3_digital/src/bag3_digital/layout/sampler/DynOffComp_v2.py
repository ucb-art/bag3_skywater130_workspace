"""This module contains version 2 of layout generators for a DOC comparator with external Vb,
NMOS input and NMOS switches."""

from typing import Any, Dict, Optional, Type

from pybag.enum import MinLenMode, RoundMode, Orientation
from pybag.core import Transform

from bag.util.immutable import Param
from bag.layout.template import TemplateDB
from bag.layout.routing.base import TrackID
from bag.design.module import Module

from xbase.layout.enum import MOSWireType, MOSPortType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase
from xbase.layout.cap.core import MOMCapCore

from ...schematic.DynOffCompVb_core2N import bag3_digital__DynOffCompVb_core2N
from ...schematic.DynOffCompVb_Cap2N import bag3_digital__DynOffCompVb_Cap2N


class DynOffCompCore(MOSBase):
    """Core of DOC comparator with external Vb, NMOS input and NMOS switches.

    Assumes:

    1. 3 rows: nch (flipped), nch, pch.
    2. Row 0: Mc2, M2, M1, Mc1
    3. Row 1: S2, S4, S5, S3, S1
    4. Row 2: EN2, M4, M3, EN1
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag3_digital__DynOffCompVb_core2N

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='List of segments for different devices.',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_p='index for pmos row with gm cells and enable switches.',
            ridx_nsw='index for nmos row with switches.',
            ridx_n='index for nmos row with input devices and gm cells.',
            draw_taps='True to draw substrate taps.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_nsw=1,
            ridx_n=0,
            draw_taps=True,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg_dict: Dict[str, int] = self.params['seg_dict']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        ridx_p: int = self.params['ridx_p']
        ridx_nsw: int = self.params['ridx_nsw']
        ridx_n: int = self.params['ridx_n']
        draw_taps: bool = self.params['draw_taps']

        hm_layer = self.conn_layer + 1

        seg_sw12 = seg_dict['sw12']
        seg_sw34 = seg_dict['sw34']
        seg_sw5 = seg_dict['sw5']
        seg_gm1 = seg_dict['gm1']
        seg_gm2p = seg_dict['gm2p']
        seg_gm2n = seg_dict['gm2n']
        seg_en12 = seg_dict['en12']
        seg_sep = self.min_sep_col

        seg_tot = max((seg_gm2n + seg_gm1) * 2, (seg_sw34 + seg_sw12 + seg_sep) * 2 + seg_sw5,
                      (seg_gm2p + seg_en12 + seg_sep) * 2)

        # taps on left and right
        if draw_taps:
            tap_n_cols = self.get_tap_ncol()
            tap_sep_col = self.sub_sep_col
            seg_tot += 2 * (tap_sep_col + tap_n_cols)

        self.set_mos_size(seg_tot)
        seg_tot2 = seg_tot // 2

        # --- Placement --- #
        #
        # 1. nch row gm2n + gm1n
        # regeneration nmos
        gm2n_2 = self.add_mos(ridx_n, seg_tot2, seg_gm2n, flip_lr=True, w=w_n)
        gm2n_1 = self.add_mos(ridx_n, seg_tot2, seg_gm2n, w=w_n)

        # input nmos
        gm1n_2 = self.add_mos(ridx_n, seg_tot2 - seg_gm2n, seg_gm1, w=w_n, flip_lr=True)
        gm1n_1 = self.add_mos(ridx_n, seg_tot2 + seg_gm2n, seg_gm1, w=w_n)

        _row_info = pinfo.get_row_place_info(ridx_n).row_info
        w_gm2n, th_gm2n = _row_info.width, _row_info.threshold

        # 2. nch row sw12 + sw34 + sw5: break sw5 into 2 halves for symmetry
        if seg_sw5 % 2 != 0:
            raise ValueError('Number of fingers of bridge switch S5 has to be even.')
        seg_sw5_2 = seg_sw5 // 2

        sw5_dum = self.add_mos(ridx_nsw, seg_tot2 - 1, 2, w=w_n)

        sw5_right = self.add_mos(ridx_nsw, seg_tot2 + 1, seg_sw5_2, w=w_n)
        sw5_left = self.add_mos(ridx_nsw, seg_tot2 - 1, seg_sw5_2, w=w_n, flip_lr=True)
        _row_info = pinfo.get_row_place_info(ridx_nsw).row_info
        w_sw, th_sw = _row_info.width, _row_info.threshold
        dum_info = [((_row_info, _row_info.width, pinfo.lch, _row_info.threshold, 'VSS', 'SW_M'),
                     1),
                    ((_row_info, _row_info.width, pinfo.lch, _row_info.threshold, 'VSS', 'SW_P'),
                     1)]

        # place sw3 and sw4
        sw34_col = 1 + seg_sw5_2
        sw4 = self.add_mos(ridx_nsw, seg_tot2 - sw34_col, seg_sw34, flip_lr=True, w=w_n)
        sw3 = self.add_mos(ridx_nsw, seg_tot2 + sw34_col, seg_sw34, w=w_n)

        # place sw1 and sw2
        sw12_col = max(sw34_col + seg_sw34 + self.min_sep_col, seg_gm2n)
        sw2 = self.add_mos(ridx_nsw, seg_tot2 - sw12_col, seg_sw12, flip_lr=True, w=w_n)
        sw1 = self.add_mos(ridx_nsw, seg_tot2 + sw12_col, seg_sw12, w=w_n)

        if (sw12_col - seg_gm2n - seg_sep) % 2:
            in_gate, in_drain = MOSPortType.S, MOSPortType.D
        else:
            in_gate, in_drain = MOSPortType.D, MOSPortType.S

        # 3. pch row for gm2p cells and enable switches
        gm2p_4 = self.add_mos(ridx_p, seg_tot2, seg_gm2p, flip_lr=True, w=w_p)
        gm2p_3 = self.add_mos(ridx_p, seg_tot2, seg_gm2p, w=w_p)

        en2 = self.add_mos(ridx_p, seg_tot2 - seg_gm2p - seg_sep, seg_en12, flip_lr=True, w=w_p,
                           g_on_s=True)
        en1 = self.add_mos(ridx_p, seg_tot2 + seg_gm2p + seg_sep, seg_en12, w=w_p, g_on_s=True)

        _row_info = pinfo.get_row_place_info(ridx_p).row_info
        w_gmp, th_gmp = _row_info.width, _row_info.threshold

        # 4. add taps
        tap_vdd_list, tap_vss_list = [], []
        if draw_taps:
            self.add_tap(0, tap_vdd_list, tap_vss_list)
            self.add_tap(seg_tot, tap_vdd_list, tap_vss_list, flip_lr=True)

        # --- Routing --- #
        #
        # 1. clock signals
        swg_tid = self.get_track_id(ridx_nsw, MOSWireType.G, wire_name='clk', wire_idx=1)
        swdg_tid = self.get_track_id(ridx_nsw, MOSWireType.G, wire_name='clk', wire_idx=0)

        clk = self.connect_to_tracks([sw1.g, sw2.g, sw3.g, sw4.g], swg_tid)
        clkd = self.connect_to_tracks([sw5_left.g, sw5_right.g], swdg_tid,
                                      min_len_mode=MinLenMode.MIDDLE)

        # 2. vdd: s nodes of pch go to VDD; d nodes of pch are actual drain connections
        vdd_tid = self.get_track_id(ridx_p, MOSWireType.DS_MATCH, wire_name='sup')

        vdd = self.connect_to_tracks([gm2p_3.s, gm2p_4.s, en1.s, en2.s], vdd_tid)
        self.connect_to_track_wires(tap_vdd_list, vdd)

        en_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sig')

        rx_en = self.connect_to_tracks([en1.g, en2.g], en_tid)

        # 3. vss: s nodes of gm2n and gm1 go to VSS; d nodes of gm2n are actual drain connections
        vss_tid = self.get_track_id(ridx_n, MOSWireType.DS_MATCH, wire_name='sup')

        vss = self.connect_to_tracks([gm2n_1.s, gm2n_2.s, gm1n_1.s, gm1n_2.s], vss_tid)
        self.connect_to_track_wires(tap_vss_list, vss)
        # short gate and drain of dummy sw5 to vss as well
        self.connect_to_tracks([sw5_dum.d, sw5_dum.g], vss_tid)

        # 4. input to gm1 and drain/source of sw1/2
        in_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sig')
        vip = self.connect_to_tracks([gm1n_2.g, sw2[in_gate]], in_tid,
                                     min_len_mode=MinLenMode.LOWER)
        vim = self.connect_to_tracks([gm1n_1.g, sw1[in_gate]], in_tid,
                                     min_len_mode=MinLenMode.UPPER)

        # 5. drain/source of sw3, sw4 connection to Vb
        vb_tid = self.get_track_id(ridx_nsw, MOSWireType.DS, wire_idx=1, wire_name='sig')
        if (1 + seg_sw5_2) % 2:
            vb_term, int_term = MOSPortType.D, MOSPortType.S
        else:
            vb_term, int_term = MOSPortType.S, MOSPortType.D
        vb = self.connect_to_tracks([sw3[vb_term], sw4[vb_term]], vb_tid)

        # 6. drains/sources of sw1/2
        sw12_term_tid = self.get_track_id(ridx_nsw, MOSWireType.DS, wire_idx=0, wire_name='sig')
        sw2_term = self.connect_to_tracks(sw2[in_drain], sw12_term_tid)
        sw1_term = self.connect_to_tracks(sw1[in_drain], sw12_term_tid)

        # 7. gates of gm2 stage
        # connect gates of gm2n to drain/source of sw5 and int_term of sw3/sw4
        sw5_l_idx = self.get_track_index(ridx_nsw, MOSWireType.DS_MATCH, wire_idx=3,
                                         wire_name='sig')
        sw5_r_idx = self.get_track_index(ridx_nsw, MOSWireType.DS_MATCH, wire_idx=2,
                                         wire_name='sig')
        num_g_min = min(gm2n_1.g.track_id.num, gm2p_3.g.track_id.num)
        left_warr_list, right_warr_list = [sw5_left.s, sw5_right.d, sw4[int_term]], \
                                          [sw5_left.d, sw5_right.s, sw3[int_term]]
        sw5_l, sw5_r = self.connect_differential_tracks(left_warr_list, right_warr_list, hm_layer,
                                                        sw5_l_idx, sw5_r_idx)
        self.connect_differential_tracks(en2.d, en1.d, hm_layer, sw5_l_idx, sw5_r_idx)
        self.connect_differential_tracks(gm2n_2.g[0:num_g_min], gm2n_1.g[0:num_g_min], hm_layer,
                                         sw5_l_idx, sw5_r_idx)
        if num_g_min < gm2n_1.g.track_id.num:
            self.connect_differential_tracks([gm2n_2.g[num_g_min:]], [gm2n_1.g[num_g_min:]],
                                             hm_layer, sw5_l_idx, sw5_r_idx)
        if num_g_min < gm2p_3.g.track_id.num:
            self.connect_differential_tracks([gm2p_4.g[num_g_min:]], [gm2p_3.g[num_g_min:]],
                                             hm_layer, sw5_l_idx, sw5_r_idx)

        # connect gates of gm2n and gm2p using all of the smaller number of gates (sw5 drain/gate
        # gets automatically connected in correct manner)
        for _idx3 in range(num_g_min):
            self.connect_wires([gm2p_3.g[_idx3], gm2n_1.g[_idx3]])
            self.connect_wires([gm2p_4.g[_idx3], gm2n_2.g[_idx3]])

        # 8. outputs
        outp_top_idx = self.get_track_index(ridx_p, MOSWireType.DS_MATCH, wire_idx=0,
                                            wire_name='sig')
        outm_top_idx = self.get_track_index(ridx_p, MOSWireType.DS_MATCH, wire_idx=1,
                                            wire_name='sig')

        vop_top, vom_top = self.connect_differential_tracks(gm2p_3.d, gm2p_4.d, hm_layer,
                                                            outp_top_idx, outm_top_idx)

        outp_bot_idx = self.get_track_index(ridx_n, MOSWireType.DS_MATCH, wire_idx=0,
                                            wire_name='sig')
        outm_bot_idx = self.get_track_index(ridx_n, MOSWireType.DS_MATCH, wire_idx=1,
                                            wire_name='sig')

        vop_bot, vom_bot = self.connect_differential_tracks([gm1n_1.d, gm2n_1.d],
                                                            [gm1n_2.d, gm2n_2.d], hm_layer,
                                                            outp_bot_idx, outm_bot_idx)

        self.add_pin('VOP_top', vop_top, label='VOP:')
        self.add_pin('VOM_top', vom_top, label='VOM:')
        self.add_pin('VOP_bot', vop_bot, label='VOP:')
        self.add_pin('VOM_bot', vom_bot, label='VOM:')
        self.add_pin('VOP_sw', sw1_term, label='VOP:')
        self.add_pin('VOM_sw', sw2_term, label='VOM:')

        # # --- Pins --- #
        self.add_pin('RST', clk)
        self.add_pin('RSTD', clkd)
        self.add_pin('VDD', vdd, label='VDD:')
        self.add_pin('RX_EN', rx_en)
        self.add_pin('VSS', vss, label='VSS:')
        self.add_pin('VIP', vip)
        self.add_pin('VIM', vim)
        self.add_pin('Vb', vb)
        self.add_pin('SW_P', sw5_l)
        self.add_pin('SW_M', sw5_r)

        # set properties
        self.sch_params = dict(
            seg_dict=dict(
                sw12=seg_sw12,
                sw34=seg_sw34,
                sw5=seg_sw5,
                gm1=seg_gm1,
                gm2p=seg_gm2p,
                gm2n=seg_gm2n,
                en12=seg_en12,
            ),
            lch=pinfo.lch,
            w_dict=dict(
                sw12=w_sw,
                sw34=w_sw,
                sw5=w_sw,
                gm1=w_gm2n,
                gm2p=w_gmp,
                gm2n=w_gm2n,
                en12=w_gmp,
            ),
            th_dict=dict(
                sw12=th_sw,
                sw34=th_sw,
                sw5=th_sw,
                gm1=th_gm2n,
                gm2p=th_gmp,
                gm2n=th_gm2n,
                en12=th_gmp,
            ),
            dum_info=dum_info,
        )


class DynOffCompCap(MOSBase):
    """Comparator core with capacitors."""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag3_digital__DynOffCompVb_Cap2N

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            doc_params='Params for doc comparator',
            cap_in_params='Params for input AC coupling MOMCap',
            cap_cross_params='Params for cross MOMCap',
            mode='1 for single-ended, 2 for differential',
            draw_taps='True to draw substrate taps',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            mode=2,
            draw_taps=True,
        )

    def draw_layout(self) -> None:
        # DOC comparator with wrapping
        doc_params: Param = self.params['doc_params'].copy(append=dict(draw_taps=False))
        mode: int = self.params['mode']
        if mode not in [1, 2]:
            raise ValueError(f'Illegal value of mode={mode}. Use 1 for single-ended, '
                             f'2 for differential')
        draw_taps = self.params['draw_taps']

        doc_master = self.new_template(DynOffCompCore, params=doc_params)

        self.draw_base(doc_master.place_info)

        # capacitors
        cap_in_params: Param = self.params['cap_in_params']
        cap_cross_params: Param = self.params['cap_cross_params']

        cap_in_master = self.new_template(MOMCapCore, params=cap_in_params)
        cap_cross_master = self.new_template(MOMCapCore, params=cap_cross_params)

        doc_ncols = doc_master.num_cols
        sd_pitch = doc_master.sd_pitch

        cap_in_w, cap_in_h = cap_in_master.bound_box.w, cap_in_master.bound_box.h
        cap_cross_w, cap_cross_h = cap_cross_master.bound_box.w, cap_cross_master.bound_box.h
        cap_in_ncols, cap_cross_ncols = - (- cap_in_w // sd_pitch), - (- cap_cross_w // sd_pitch)

        sep_ncols = self.get_hm_sp_le_sep_col()

        cap_cols_offset = max(cap_in_ncols, cap_cross_ncols)

        tap_n_cols = self.get_tap_ncol()
        tap_sep_col = self.sub_sep_col

        cap_left_col = tap_n_cols + tap_sep_col if draw_taps else 0
        doc_col = cap_left_col + cap_cols_offset + sep_ncols
        cap_right_col = doc_col + doc_ncols + sep_ncols
        tot_cols = cap_right_col + cap_cols_offset
        if draw_taps:
            tot_cols += tap_n_cols + tap_sep_col

        self.set_mos_size(tot_cols)

        # --- Placement --- #
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1

        doc = self.add_tile(doc_master, 0, doc_col)

        # place capacitors relative to supply tracks of comparator
        # 1. Find TrackID of vss and vdd rails of comparator core instance
        doc_bot_hm = doc.get_pin('VOP_bot', layer=hm_layer)
        avail_yb_idx = doc_bot_hm.track_id.base_index
        doc_top_hm = doc.get_pin('VOM_top', layer=hm_layer)
        avail_yt_idx = doc_top_hm.track_id.base_index

        hm_track_pitch = self.grid.get_track_pitch(hm_layer)

        # 2. Find TrackID ports of cap masters
        port_yb_idx = avail_yt_idx
        for port_name in cap_in_master.port_names_iter():
            port = cap_in_master.get_port(port_name)
            for pin in port.get_pins(hm_layer):
                port_yb_idx = min(pin.track_id.base_index, port_yb_idx)

        port_yt_idx = 0
        for port_name in cap_cross_master.port_names_iter():
            port = cap_cross_master.get_port(port_name)
            for pin in port.get_pins(hm_layer):
                port_yt_idx = max(pin.track_id.base_index, port_yt_idx)

        off_yb = (avail_yb_idx - port_yb_idx) * hm_track_pitch
        off_yt = (avail_yt_idx - port_yt_idx) * hm_track_pitch

        blk_w, blk_h = self.grid.get_block_size(cap_cross_master.top_layer, half_blk_x=True,
                                                half_blk_y=True)
        off_yb = -(-off_yb // blk_h) * blk_h
        off_yt = off_yt // blk_h * blk_h

        if off_yb + cap_in_h >= off_yt:
            raise ValueError('Try decreasing capacitor height')

        cap_in_right = self.add_instance(cap_in_master, xform=Transform(cap_right_col * sd_pitch,
                                                                        off_yb + cap_in_h,
                                                                        Orientation.MX))
        cap_in_left = self.add_instance(cap_in_master, xform=Transform((cap_left_col +
                                                                        cap_cols_offset) * sd_pitch,
                                                                       off_yb + cap_in_h,
                                                                       Orientation.R180))

        cap_cross_right = self.add_instance(cap_cross_master, xform=Transform(cap_right_col *
                                            sd_pitch, off_yt + cap_cross_h, Orientation.MX))
        cap_cross_left = self.add_instance(cap_cross_master, xform=Transform((cap_left_col +
                                           cap_cols_offset) * sd_pitch, off_yt + cap_cross_h,
                                                                             Orientation.R180))

        # 3. add taps:
        tap_vdd_list, tap_vss_list = [], []
        if draw_taps:
            self.add_tap(0, tap_vdd_list, tap_vss_list)
            self.add_tap(tot_cols, tap_vdd_list, tap_vss_list, flip_lr=True)

        # --- Routing --- #

        # 1. connect input cap to VIP and VIN
        vip, vim = doc.get_pin('VIP'), doc.get_pin('VIM')
        vm_tidx_list = []
        for _cap, _vi, _coord in zip([cap_in_left, cap_in_right], [vip, vim], [vip.lower,
                                                                               vim.upper]):
            # get vip/vim on vm_layer
            _vm_tidx = self.grid.coord_to_track(vm_layer, _coord, mode=RoundMode.NEAREST)
            vm_tidx_list.append(_vm_tidx)
            _vi_vm = self.connect_to_tracks(_vi, TrackID(vm_layer, _vm_tidx),
                                            min_len_mode=MinLenMode.LOWER)

            # connect to cap_in
            _plus = _cap.get_pin('plus', layer=xm_layer)
            _vi_xm = self.connect_to_track_wires([_vi_vm], _plus)
            _name = 'VIM' if _vi == vim else 'VIP'
            self.add_pin(_name, _vi_xm)

        vpreP = cap_in_left.get_pin('minus', layer=xm_layer)
        self.add_pin('VpreP', vpreP, label='VpreP:')

        vpreM = cap_in_right.get_pin('minus', layer=xm_layer)
        if mode == 2:
            self.add_pin('VpreM', vpreM, label='VpreM:')
        elif mode == 1:
            self.add_pin('VOM_cap1', vpreM, label='VOM:')

        # get sw5 terminal on vm_layer
        _sw_m, _sw_p = doc.get_pin('SW_M'), doc.get_pin('SW_P')
        left_vm_tidx = self.tr_manager.get_next_track(vm_layer, vm_tidx_list[0], 'sig', 'sig',
                                                      up=True)
        right_vm_tidx = self.tr_manager.get_next_track(vm_layer, vm_tidx_list[1], 'sig', 'sig',
                                                       up=False)
        sw_m, sw_p = self.connect_differential_tracks(_sw_m, _sw_p, vm_layer, left_vm_tidx,
                                                      right_vm_tidx)
        self.add_pin('SW_P', _sw_p)
        self.add_pin('SW_M', _sw_m)

        sw5_list = []
        for _cap, _sw in zip([cap_cross_left, cap_cross_right], [sw_m, sw_p]):
            # connect sw5 to minus port of cap
            _minus = _cap.get_pin('minus', layer=xm_layer)
            sw5_list.append(self.connect_to_track_wires([_sw], _minus))

            # connect vop/vom to plus port of cap
            _plus = _cap.get_pin('plus', layer=xm_layer)
            if _cap == cap_cross_left:
                _name = 'VOM_cap'
                _label = 'VOM:'
            else:
                _name = 'VOP_cap'
                _label = 'VOP:'
            self.add_pin(_name, _plus, label=_label)

        # supplies
        for sup_name, tap_sup_list in zip(['VDD', 'VSS'], [tap_vdd_list, tap_vss_list]):
            sup = doc.get_pin(sup_name)
            self.connect_to_track_wires(tap_sup_list, sup)
            self.add_pin(sup_name, sup, label=f'{sup_name}:')

        # re-export pins
        for name in ['RST', 'RSTD', 'Vb', 'RX_EN', 'VOP_top', 'VOM_top', 'VOP_sw', 'VOM_sw',
                     'VOP_bot', 'VOM_bot']:
            self.reexport(doc.get_port(name))

        # set properties
        self.sch_params = dict(
            mode=mode,
            cap_in_params=cap_in_master.sch_params,
            cap_cross_params=cap_cross_master.sch_params,
            doc_params=doc_master.sch_params,
        )
