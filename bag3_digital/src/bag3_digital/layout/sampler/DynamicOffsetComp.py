"""This module contains layout generators for a DOC comparator."""

from typing import Any, Dict

from pybag.enum import MinLenMode, RoundMode, Orientation
from pybag.core import Transform

from bag.util.immutable import Param
from bag.layout.template import TemplateDB
from bag.layout.routing.base import TrackID

from xbase.layout.enum import MOSWireType, MOSPortType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase
from xbase.layout.mos.top import MOSBaseWrapper
from xbase.layout.cap.core import MOMCapCore


class DynamicOffsetComp_Core(MOSBase):
    """Core of DOC comparator.

    Assumes:

    1. 4 rows: nch (flipped), nch (flipped), nch, pch.
    2. Row 0: M2, M1
    3. Row 1: S2, S5, S1
    4. Row 2: S4, S3
    5. Row 3: Mc2, M4, M3, Mc1
    6. Row 4: power gate Mpow
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='List of segments for different devices.',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_ppow='index for pmos row with power gate',
            ridx_p='index for pmos row with input devices and gm cells.',
            ridx_nsw='index for nmos row with switches.',
            ridx_nsw5='index for nmos row with switch 5.',
            ridx_ngm='index for nmos row with gm cells.',
            show_pins='True to show pins.'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_ppow=-1,
            ridx_p=-2,
            ridx_nsw=2,
            ridx_nsw5=1,
            ridx_ngm=0,
            show_pins=False,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg_dict: Dict[str, int] = self.params['seg_dict']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        ridx_ppow: int = self.params['ridx_ppow']
        ridx_p: int = self.params['ridx_p']
        ridx_nsw: int = self.params['ridx_nsw']
        ridx_nsw5: int = self.params['ridx_nsw5']
        ridx_ngm: int = self.params['ridx_ngm']
        show_pins: bool = self.params['show_pins']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1

        seg_sw12 = seg_dict['sw12']
        seg_sw34 = seg_dict['sw34']
        seg_sw5 = seg_dict['sw5']
        seg_gm1 = seg_dict['gm1']
        seg_gm2p = seg_dict['gm2p']
        seg_gm2n = seg_dict['gm2n']
        seg_pow_cross = seg_dict['pow_cross']
        seg_pow_in = seg_dict['pow_in']
        seg_sep = max(self.min_sep_col,
                      self.get_hm_sp_le_sep_col(self.tr_manager.get_width(hm_layer, 'sup')))
        seg_sep += (seg_sep + seg_pow_in // 2) % 2

        assert seg_pow_cross == 2 * seg_gm2p, f'Power gate {seg_pow_cross} of cross coupled pair ' \
                                              f'must have same number of fingers as both pmos ' \
                                              f'{seg_gm2p} of pair'
        assert seg_pow_in >= seg_gm1, f'Power gate {seg_pow_in} of input pmos must have greater ' \
                                      f'or equal fingers as input pmos {seg_gm1}'

        seg_tot = max(seg_sw34 * 2, seg_gm2n * 2, (seg_sw12 + self.min_sep_col) * 2 + seg_sw5,
                      seg_pow_cross + 2 * (seg_pow_in + seg_sep))
        self.set_mos_size(seg_tot)
        seg_tot2 = seg_tot // 2

        # --- Placement --- #
        # nch row gm2n
        gm2n_2 = self.add_mos(ridx_ngm, seg_tot2, seg_gm2n, flip_lr=True, w=w_n)
        gm2n_1 = self.add_mos(ridx_ngm, seg_tot2, seg_gm2n, abut=True, w=w_n)
        _row_info = pinfo.get_row_place_info(ridx_ngm).row_info
        w_gm2n, th_gm2n = _row_info.width, _row_info.threshold

        # nch row sw12 + sw5: break sw5 into 2 halves for symmetry
        if seg_sw5 % 2 != 0:
            raise ValueError('Number of fingers of bridge switch S5 has to be even.')
        seg_sw5_2 = seg_sw5 // 2

        sw5_dum = self.add_mos(ridx_nsw5, seg_tot2 - 1, 2, w=w_n)
        vss_dum_tid = self.get_track_id(ridx_nsw5, MOSWireType.DS, wire_name='sup')
        vss_dum = self.connect_to_tracks([sw5_dum.d, sw5_dum.g], vss_dum_tid,
                                         min_len_mode=MinLenMode.MIDDLE)

        sw5_right = self.add_mos(ridx_nsw5, seg_tot2 + 1, seg_sw5_2, w=w_n, abut=True)
        sw5_left = self.add_mos(ridx_nsw5, seg_tot2 - 1, seg_sw5_2, w=w_n, abut=True, flip_lr=True)
        _row_info = pinfo.get_row_place_info(ridx_nsw5).row_info
        w_sw5, th_sw5 = _row_info.width, _row_info.threshold
        dum_info = [((_row_info, _row_info.width, pinfo.lch, _row_info.threshold, 'VSS', 'SW_M'),
                     1),
                    ((_row_info, _row_info.width, pinfo.lch, _row_info.threshold, 'VSS', 'SW_P'),
                     1)]

        # place sw1 and sw2
        sw12_col = max(1 + seg_sw5_2 + self.min_sep_col, min(seg_gm2p, seg_gm2n))
        sw2 = self.add_mos(ridx_nsw5, seg_tot2 - sw12_col, seg_sw12, flip_lr=True, w=w_n)
        sw1 = self.add_mos(ridx_nsw5, seg_tot2 + sw12_col, seg_sw12, w=w_n)

        # nch row sw34
        # make sure sw3 and sw4 lines up with gm1p, but gates align with gates of sw1 and sw2
        g_on_s = abs(sw12_col - seg_gm2p) % 2 == 1
        sw4 = self.add_mos(ridx_nsw, seg_tot2 - seg_gm2p, seg_sw34, flip_lr=True, w=w_n,
                           g_on_s=g_on_s)
        sw3 = self.add_mos(ridx_nsw, seg_tot2 + seg_gm2p, seg_sw34, w=w_n, g_on_s=g_on_s)
        _row_info = pinfo.get_row_place_info(ridx_nsw).row_info
        w_sw, th_sw = _row_info.width, _row_info.threshold

        # pch row for gm1 and gm2p cells
        gm2p_4 = self.add_mos(ridx_p, seg_tot2, seg_gm2p, flip_lr=True, w=w_p)
        gm2p_3 = self.add_mos(ridx_p, seg_tot2, seg_gm2p, abut=True, w=w_p)

        offset_gm1p = (seg_pow_in - seg_gm1) // 2
        gm1p_2 = self.add_mos(ridx_p, seg_tot2 - seg_gm2p - seg_sep - offset_gm1p, seg_gm1, w=w_p,
                              flip_lr=True)
        gm1p_1 = self.add_mos(ridx_p, seg_tot2 + seg_gm2p + seg_sep + offset_gm1p, seg_gm1, w=w_p)
        _row_info = pinfo.get_row_place_info(ridx_p).row_info
        w_gmp, th_gmp = _row_info.width, _row_info.threshold

        # pch row for power gating cells
        ppow_cross = self.add_mos(ridx_ppow, seg_tot2 - seg_gm2p, seg_pow_cross, w=w_p)

        if offset_gm1p % 2:
            sup_term, gated_term = MOSPortType.S, MOSPortType.D
            pow_g_on_s = True
        else:
            sup_term, gated_term = MOSPortType.D, MOSPortType.S
            pow_g_on_s = False

        ppow_in_l = self.add_mos(ridx_ppow, seg_tot2 - seg_gm2p - seg_sep, seg_pow_in, w=w_p,
                                 flip_lr=True, g_on_s=pow_g_on_s)
        ppow_in_r = self.add_mos(ridx_ppow, seg_tot2 + seg_gm2p + seg_sep, seg_pow_in, w=w_p,
                                 g_on_s=pow_g_on_s)
        _row_info = pinfo.get_row_place_info(ridx_ppow).row_info
        w_pow, th_pow = _row_info.width, _row_info.threshold

        # --- Routing --- #

        # 1. clock signals
        swg_tid = self.get_track_id(ridx_nsw, MOSWireType.G, wire_name='clk')
        sw5g_tid = self.get_track_id(ridx_nsw5, MOSWireType.G, wire_name='clk')

        clk = self.connect_to_tracks([sw1.g, sw2.g, sw3.g, sw4.g], swg_tid)
        clkd = self.connect_to_tracks([sw5_left.g, sw5_right.g], sw5g_tid,
                                      min_len_mode=MinLenMode.MIDDLE)

        # 2. vdd: s nodes of pch go to VDD_gated; d nodes of pch are actual drain connections
        vdd_gate_tid = self.get_track_id(ridx_p, MOSWireType.DS_MATCH, wire_name='sup')

        # 2a. cross coupled pair is same size as the power gate
        self.connect_to_tracks([gm2p_3.s, gm2p_4.s, ppow_cross.s], vdd_gate_tid)

        # 2b. power gate left and right input
        self.connect_to_tracks([gm1p_2.s, ppow_in_l[gated_term]], vdd_gate_tid)
        self.connect_to_tracks([gm1p_1.s, ppow_in_r[gated_term]], vdd_gate_tid)

        vdd_tid = self.get_track_id(ridx_ppow, MOSWireType.DS_MATCH, wire_name='sup')
        vdd = self.connect_to_tracks([ppow_cross.d, ppow_in_l[sup_term], ppow_in_r[sup_term]],
                                     vdd_tid)

        pow_g0_tid = self.get_track_id(ridx_ppow, MOSWireType.G, wire_name='sig', wire_idx=0)
        pow_g1_tid = self.get_track_id(ridx_ppow, MOSWireType.G, wire_name='sig', wire_idx=1)
        pow_g2_tid = self.get_track_id(ridx_ppow, MOSWireType.G, wire_name='sig', wire_idx=2)

        rx_enb_cross = self.connect_to_tracks([ppow_cross.g], pow_g0_tid)
        rx_enb_in_l = self.connect_to_tracks([ppow_in_l.g], pow_g1_tid)
        rx_enb_in_r = self.connect_to_tracks([ppow_in_r.g], pow_g2_tid)

        # 3. vss: s nodes of gm2n go to VSS; d nodes of gm2n are actual drain connections
        vss_tid = self.get_track_id(ridx_ngm, MOSWireType.DS_MATCH, wire_name='sup')

        vss = self.connect_to_tracks([gm2n_1.s, gm2n_2.s], vss_tid)

        # 4. input to gm1 and drain of sw3/4
        in_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sig')
        vip = self.connect_to_tracks([gm1p_2.g, sw4.d], in_tid, min_len_mode=MinLenMode.LOWER)
        vim = self.connect_to_tracks([gm1p_1.g, sw3.d], in_tid, min_len_mode=MinLenMode.UPPER)

        # 5. drain/source of sw1, sw2 connection to Vb
        vb_tid = self.get_track_id(ridx_nsw5, MOSWireType.DS_MATCH, wire_idx=2, wire_name='sig')
        if (1 + seg_sw5_2) % 2:
            vb_term, int_term = MOSPortType.D, MOSPortType.S
        else:
            vb_term, int_term = MOSPortType.S, MOSPortType.D
        vb = self.connect_to_tracks([sw1[vb_term], sw2[vb_term]], vb_tid)

        # 6. sources of sw3/4
        sw34_s_tid = self.get_track_id(ridx_nsw, MOSWireType.DS, wire_idx=0, wire_name='sig')
        sw4_s = self.connect_to_tracks(sw4.s, sw34_s_tid)
        sw3_s = self.connect_to_tracks(sw3.s, sw34_s_tid)

        # 7. gates of gm2 stage
        # connect gates of gm2n to drain/source of sw5 and int_term of sw1/sw2
        sw5_l_tid = self.get_track_id(ridx_nsw5, MOSWireType.DS_MATCH, wire_idx=0, wire_name='sig')
        sw5_r_idx = self.get_track_index(ridx_nsw5, MOSWireType.DS_MATCH, wire_idx=1,
                                         wire_name='sig')
        num_g_min = min(gm2n_1.g.track_id.num, gm2p_3.g.track_id.num)
        left_warr_list, right_warr_list = [sw5_left.s, sw5_right.d, gm2n_2.g[0:num_g_min-1],
                                           sw2[int_term]], \
                                          [sw5_left.d, sw5_right.s, gm2n_1.g[0:num_g_min-1],
                                           sw1[int_term]]
        sw5_l, sw5_r = self.connect_differential_tracks(left_warr_list, right_warr_list,
                                                        sw5_l_tid.layer_id,
                                                        sw5_l_tid.base_index, sw5_r_idx)
        self.connect_differential_tracks([gm2n_2.g[num_g_min:]], [gm2n_1.g[num_g_min:]],
                                         sw5_l_tid.layer_id, sw5_l_tid.base_index, sw5_r_idx)

        # connect gates of gm2n and gm2p using all of the smaller number of gates (sw5 drain/gate
        # gets automatically connected in correct manner)
        for _idx3 in range(num_g_min):
            self.connect_wires([gm2p_3.g[_idx3], gm2n_1.g[_idx3]])
            self.connect_wires([gm2p_4.g[_idx3], gm2n_2.g[_idx3]])

        # 8. outputs
        outp_top_tid = self.get_track_id(ridx_p, MOSWireType.DS_MATCH, wire_idx=0, wire_name='sig')
        outm_top_idx = self.get_track_index(ridx_p, MOSWireType.DS_MATCH, wire_idx=1,
                                            wire_name='sig')

        vop_top, vom_top = self.connect_differential_tracks([gm1p_1.d, gm2p_3.d],
                                                            [gm1p_2.d, gm2p_4.d],
                                                            outp_top_tid.layer_id,
                                                            outp_top_tid.base_index, outm_top_idx)

        outp_bot_tid = self.get_track_id(ridx_ngm, MOSWireType.DS_MATCH, wire_idx=0,
                                         wire_name='sig')
        outm_bot_idx = self.get_track_index(ridx_ngm, MOSWireType.DS_MATCH, wire_idx=1,
                                            wire_name='sig')

        vop_bot, vom_bot = self.connect_differential_tracks(gm2n_1.d, gm2n_2.d,
                                                            outp_bot_tid.layer_id,
                                                            outp_bot_tid.base_index, outm_bot_idx)

        vom_vm_idx = self.grid.coord_to_track(vm_layer, sw4_s.middle, mode=RoundMode.NEAREST)
        vop_vm_idx = self.grid.coord_to_track(vm_layer, sw3_s.middle, mode=RoundMode.NEAREST)

        # Assume nch will be larger than the pch since it is mainly driving regeneration
        vop, vom = self.connect_differential_tracks([vop_bot], [vom_bot],
                                                    vm_layer, vop_vm_idx, vom_vm_idx)
        # connect vop_top and vom_top separately to avoid extending the horizontal metals
        self.connect_to_track_wires([vom_top], vom)
        self.connect_to_track_wires([vop_top], vop)

        # connect sw3/4_s separately to avoid shorting
        self.connect_to_tracks(sw4_s, vom.track_id)
        self.connect_to_tracks(sw3_s, vop.track_id)

        # --- Pins --- #
        self.add_pin('RST', clk, show=show_pins)
        self.add_pin('RSTD', clkd, show=show_pins)
        self.add_pin('VDD', vdd, label='VDD:', show=show_pins)
        self.add_pin('out_top_hm', vom_top, show=False)
        self.add_pin('RX_ENB_cross', rx_enb_cross, show=show_pins)
        self.add_pin('RX_ENB_left', rx_enb_in_l, show=show_pins)
        self.add_pin('RX_ENB_right', rx_enb_in_r, show=show_pins)
        self.add_pin('VSS', vss, label='VSS:', show=show_pins)
        self.add_pin('VSS', vss_dum, label='VSS:', show=show_pins)
        self.add_pin('VIP', vip, show=show_pins)
        self.add_pin('VIM', vim, show=show_pins)
        self.add_pin('Vb', vb, show=show_pins)
        self.add_pin('VOP', vop, show=show_pins)
        self.add_pin('VOM', vom, show=show_pins)
        self.add_pin('SW_P', sw5_l, show=show_pins)
        self.add_pin('SW_M', sw5_r, show=show_pins)

        # set properties
        self.sch_params = dict(
            seg_dict=dict(
                sw12=seg_sw12,
                sw34=seg_sw34,
                sw5=seg_sw5,
                gm1=seg_gm1,
                gm2p=seg_gm2p,
                gm2n=seg_gm2n,
                pow_cross=seg_pow_cross,
                pow_in=seg_pow_in,
            ),
            lch=pinfo.lch,
            w_dict=dict(
                sw12=w_sw5,
                sw34=w_sw,
                sw5=w_sw5,
                gm1=w_gmp,
                gm2p=w_gmp,
                gm2n=w_gm2n,
                pow_cross=w_pow,
                pow_in=w_pow,
            ),
            th_dict=dict(
                sw12=th_sw5,
                sw34=th_sw,
                sw5=th_sw5,
                gm1=th_gmp,
                gm2p=th_gmp,
                gm2n=th_gm2n,
                pow_cross=th_pow,
                pow_in=th_pow,
            ),
            dum_info=dum_info,
        )


class DynamicOffsetComp_Wrapper(MOSBaseWrapper):
    """DOC comparator with edge wrapping."""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBaseWrapper.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return DynamicOffsetComp_Cap.get_params_info()
        # return DynamicOffsetComp_Core.get_params_info()

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return DynamicOffsetComp_Core.get_default_param_values()

    def draw_layout(self) -> None:
        show_pins: bool = self.params['show_pins']
        core_params: Param = self.params

        core_params = core_params.copy(append=dict(show_pins=False))
        master = self.new_template(DynamicOffsetComp_Cap, params=core_params)
        # master = self.new_template(DynamicOffsetComp_Core, params=core_params)

        inst = self.draw_boundaries(master, master.top_layer)

        # re-export pins
        for name in inst.port_names_iter():
            self.reexport(inst.get_port(name), show=show_pins)

        # set properties
        self.sch_params = master.sch_params


class DynamicOffsetComp_Cap(MOSBase):
    """DOCCore with capacitors."""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            doc_params='Params for doc comparator',
            cap_in_params='Params for input AC coupling MOMCap',
            cap_cross_params='Params for cross MOMCap',
            show_pins='True to show pins',
        )

    def draw_layout(self) -> None:
        # DOC comparator with wrapping
        doc_params: Param = self.params['doc_params']
        show_pins: bool = self.params['show_pins']

        doc_params = doc_params.copy(append=dict(show_pins=show_pins))
        doc_master = self.new_template(DynamicOffsetComp_Core, params=doc_params)

        self.draw_base(doc_master.place_info)

        # capacitors
        cap_in_params: Param = self.params['cap_in_params']
        cap_cross_params: Param = self.params['cap_cross_params']

        cap_in_params = cap_in_params.copy(append=dict(show_pins=show_pins))
        cap_cross_params = cap_cross_params.copy(append=dict(show_pins=show_pins))

        cap_in_master = self.new_template(MOMCapCore, params=cap_in_params)
        cap_cross_master = self.new_template(MOMCapCore, params=cap_cross_params)

        doc_w, doc_h = doc_master.bound_box.w, doc_master.bound_box.h
        doc_h_g = doc_master.place_info.get_row_place_info(-1).yb
        doc_ncols = doc_master.num_cols
        sd_pitch = doc_master.sd_pitch

        cap_in_w, cap_in_h = cap_in_master.bound_box.w, cap_in_master.bound_box.h
        cap_cross_w, cap_cross_h = cap_cross_master.bound_box.w, cap_cross_master.bound_box.h
        cap_in_ncols, cap_cross_ncols = - (- cap_in_w // sd_pitch), - (- cap_cross_w // sd_pitch)

        sep_ncols = self.get_hm_sp_le_sep_col()

        cap_cols_offset = max(cap_in_ncols, cap_cross_ncols)

        cap_left_col = 0
        doc_col = cap_left_col + cap_cols_offset + sep_ncols
        cap_right_col = doc_col + doc_ncols + sep_ncols
        tot_cols = cap_right_col + cap_cols_offset

        if cap_in_h + cap_cross_h > doc_h_g:
            raise ValueError('Try decreasing capacitor height')

        self.set_mos_size(tot_cols)

        # --- Placement --- #
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1

        doc = self.add_tile(doc_master, 0, doc_col)

        # place capacitors relative to supply tracks of comparator
        # 1. Find TrackID of vss and gated vdd rails of comparator core instance
        doc_vss = doc.get_pin('VSS')
        avail_yb_idx = doc_vss.track_id.base_index
        doc_out_top_hm = doc.get_pin('out_top_hm')
        avail_yt_idx = doc_out_top_hm.track_id.base_index

        hm_sep = self.tr_manager.get_sep(hm_layer, ('sup', 'sig'))
        hm_track_pitch = self.grid.get_track_pitch(hm_layer)

        # 2. Find TrackID ports of cap masters
        port_yb_idx = avail_yt_idx
        for port_name in cap_cross_master.port_names_iter():
            port = cap_cross_master.get_port(port_name)
            for pin in port.get_pins(hm_layer):
                port_yb_idx = min(pin.track_id.base_index, port_yb_idx)

        port_yt_idx = 0
        for port_name in cap_in_master.port_names_iter():
            port = cap_in_master.get_port(port_name)
            for pin in port.get_pins(hm_layer):
                port_yt_idx = max(pin.track_id.base_index, port_yt_idx)

        off_yb = (avail_yb_idx + hm_sep - port_yb_idx) * hm_track_pitch
        off_yt = (avail_yt_idx - port_yt_idx) * hm_track_pitch

        blk_w, blk_h = self.grid.get_block_size(cap_cross_master.top_layer, half_blk_x=True,
                                                half_blk_y=True)
        off_yb = -(-off_yb // blk_h) * blk_h
        off_yt = off_yt // blk_h * blk_h

        cap_cross_right = self.add_instance(cap_cross_master, xform=Transform(cap_right_col *
                                                                              sd_pitch, off_yb))
        cap_cross_left = self.add_instance(cap_cross_master, xform=Transform((cap_left_col +
                                           cap_cols_offset) * sd_pitch, off_yb, Orientation.MY))

        cap_in_right = self.add_instance(cap_in_master, xform=Transform(cap_right_col * sd_pitch,
                                                                        off_yt))
        cap_in_left = self.add_instance(cap_in_master, xform=Transform((cap_left_col +
                                        cap_cols_offset) * sd_pitch, off_yt,
                                        Orientation.MY))

        # --- Routing --- #

        # 1. connect input cap to VIP and VIN
        vip, vim = doc.get_pin('VIP'), doc.get_pin('VIM')
        for _cap, _vi, _coord in zip([cap_in_left, cap_in_right], [vip, vim], [vip.lower,
                                                                               vim.upper]):
            # get vip/vim on vm_layer
            _vm_tidx = self.grid.coord_to_track(vm_layer, _coord, mode=RoundMode.NEAREST)
            _vi_vm = self.connect_to_tracks(_vi, TrackID(vm_layer, _vm_tidx),
                                            min_len_mode=MinLenMode.MIDDLE)

            # connect to cap_in
            _plus = _cap.get_pin('plus', layer=vm_layer)
            _xm_tidx = self.grid.coord_to_track(xm_layer, _plus.lower, mode=RoundMode.NEAREST)
            _vi_xm = self.connect_to_tracks([_plus, _vi_vm], TrackID(xm_layer, _xm_tidx))
            _name = 'VIM' if _vi == vim else 'VIP'
            self.add_pin(_name, _vi_xm)

            _name = 'VpreP:' if _cap == cap_in_left else 'VpreM:'
            self.reexport(_cap.get_port('minus'), net_name=_name)

        # 2. connect cross_cap
        # get vop/vom on vm_layer
        vop, vom = doc.get_pin('VOP'), doc.get_pin('VOM')
        self.add_pin('VOP_vm', vop, show=False)
        self.add_pin('VOM_vm', vom, show=False)

        # get sw5 terminal on vm_layer
        _sw_m, _sw_p = doc.get_pin('SW_M'), doc.get_pin('SW_P')
        left_vm_tidx = self.grid.coord_to_track(vm_layer, _sw_m.lower, mode=RoundMode.NEAREST)
        right_vm_tidx = self.grid.coord_to_track(vm_layer, _sw_p.upper, mode=RoundMode.NEAREST)
        sw_m, sw_p = self.connect_differential_tracks(_sw_m, _sw_p, vm_layer, left_vm_tidx,
                                                      right_vm_tidx)
        self.add_pin('SW_P', sw_p)
        self.add_pin('SW_M', sw_m)

        for _cap, _vo, _sw in zip([cap_cross_left, cap_cross_right], [vom, vop], [sw_m, sw_p]):
            # connect sw5 to minus port of cap
            _minus = _cap.get_pin('minus', layer=vm_layer)
            _xm_tidx = self.grid.coord_to_track(xm_layer, _minus.upper, mode=RoundMode.NEAREST)
            self.connect_to_tracks([_minus, _sw], TrackID(xm_layer, _xm_tidx))

            # connect vop/vom to plus port of cap
            _plus = _cap.get_pin('plus', layer=vm_layer)
            _xm_tidx = self.grid.coord_to_track(xm_layer, _plus.lower, mode=RoundMode.NEAREST)
            _vo_xm = self.connect_to_tracks([_plus, _vo], TrackID(xm_layer, _xm_tidx))
            _name = 'VOM' if _vo == vom else 'VOP'
            self.add_pin(_name, _vo_xm)

        # re-export pins
        for name in ['RST', 'RSTD', 'Vb', 'VDD', 'VSS', 'RX_ENB_cross', 'RX_ENB_left',
                     'RX_ENB_right']:
            self.reexport(doc.get_port(name), show=show_pins)

        # set properties
        self.sch_params = dict(
            cap_in_params=cap_in_master.sch_params,
            cap_cross_params=cap_cross_master.sch_params,
            doc_params=doc_master.sch_params,
        )
