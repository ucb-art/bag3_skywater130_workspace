"""This module contains layout generators for complex logic gates."""

from typing import Any, Dict, Type, Optional

from pybag.enum import MinLenMode

from bag.util.immutable import Param
from bag.layout.template import TemplateDB
from bag.layout.routing.base import TrackID
from bag.design.module import Module

from xbase.layout.enum import MOSWireType, MOSPortType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase

from ...schematic.reset_row import bag3_digital__reset_row
from ...schematic.inv_nor2_sym import bag3_digital__inv_nor2_sym


class ResetRow(MOSBase):
    """The complex gate for symmetric reset row

    Assumes:

    1. PMOS row above NMOS row.
    2. PMOS gate connections on bottom, NMOS gate connections on top
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag3_digital__reset_row

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='Dictionary with number of segments.',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg_dict: Dict[str, int] = self.params['seg_dict']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']

        for seg_key in ['inp', 'en_p', 'en_n', 'rstd']:
            if seg_dict[seg_key] % 2 == 1:
                raise ValueError(f'This generator does not support odd number '
                                 f'({seg_dict[seg_key]}) of segments for {seg_key}')

        seg_inp = seg_dict['inp']
        seg_en_p = seg_dict['en_p']
        seg_en_n = seg_dict['en_n']
        seg_rstd = seg_dict['rstd']
        seg_inv = seg_dict['inv']
        seg_keep = seg_dict['keep']
        seg_sep = self.min_sep_col
        if seg_inv % 2 == 0:
            seg_sep += 1

        if seg_en_n != seg_rstd:
            raise ValueError(f'Enable nch number of segments {seg_en_n} has to be equal to reset '
                             f'nch number of segments {seg_rstd}')

        seg_inner_block = max(seg_inp + seg_en_p + seg_inp, seg_en_n + seg_rstd)
        seg_tot = seg_inv + seg_sep + seg_inner_block + seg_sep + seg_keep
        self.set_mos_size(seg_tot)

        # --- Placement --- #
        left_col_p = left_col_n = 0
        right_col_p = right_col_n = seg_tot

        # output inverter
        inv_p = self.add_mos(ridx_p, left_col_p, seg_inv, w=w_p)
        left_col_p += seg_inv + seg_sep
        inv_n = self.add_mos(ridx_n, left_col_n, seg_inv, w=w_n)
        left_col_n += seg_inv + seg_sep

        # keeper inverter
        keep_p = self.add_mos(ridx_p, right_col_p, seg_keep, w=w_p, flip_lr=True)
        right_col_p -= (seg_keep + seg_sep)
        keep_n = self.add_mos(ridx_n, right_col_n, seg_keep, w=w_n, flip_lr=True)
        right_col_n -= (seg_keep + seg_sep)

        # input_minus
        seg_p_off = (seg_inner_block - seg_inp - seg_en_p - seg_inp) // 2
        left_col_p += seg_p_off
        inp_minus = self.add_mos(ridx_p, left_col_p, seg_inp, w=w_p)
        left_col_p += seg_inp

        # input_plus
        right_col_p -= seg_p_off
        inp_plus = self.add_mos(ridx_p, right_col_p, seg_inp, w=w_p, flip_lr=True)
        right_col_p -= seg_inp

        # enable pch
        en_p = self.add_mos(ridx_p, left_col_p, seg_en_p, w=w_p)
        left_col_p += seg_en_p
        assert left_col_p == right_col_p, "Check placement in pch row"

        # stacked transistors: delayed reset and enable nch
        x_inner_left = self.add_mos(ridx_n, left_col_n, seg_en_n // 2, w=w_n, stack=2, g_on_s=True,
                                    sep_g=True)
        x_inner_right = self.add_mos(ridx_n, right_col_n, seg_en_n // 2, w=w_n, stack=2,
                                     g_on_s=True, sep_g=True, flip_lr=True)

        # --- Routing --- #
        hm_layer = self.conn_layer + 1

        # 1a. Vgate- signal
        vgate_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=1)
        vgate_minus = self.connect_to_tracks([inp_minus.g], vgate_tid,
                                             min_len_mode=MinLenMode.LOWER)

        # 1b. Vgate+ signal
        vgate_plus = self.connect_to_tracks([inp_plus.g], vgate_tid, min_len_mode=MinLenMode.UPPER)

        # 2. vdd:
        vdd_tid = self.get_track_id(ridx_p, MOSWireType.DS_GATE, wire_name='sup')
        vdd = self.connect_to_tracks([inv_p.d, inp_minus.d, en_p.d, inp_plus.d, keep_p.d], vdd_tid)

        # 3. vss:
        vss_tid = self.get_track_id(ridx_n, MOSWireType.DS_GATE, wire_name='sup')
        vss = self.connect_to_tracks([inv_n.d, x_inner_left.s, x_inner_right.s, keep_n.d], vss_tid)

        # # 4. rstd
        rstd_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=0)
        rstd = self.connect_to_tracks([x_inner_left.g1, x_inner_right.g1], rstd_tid,
                                      min_len_mode=MinLenMode.MIDDLE)

        # 5. rx_en
        en_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=1)
        rx_en = self.connect_to_tracks([en_p.g, x_inner_left.g0, x_inner_right.g0], en_tid)

        # 6. output
        out_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=2)
        out = self.connect_to_tracks([keep_p.g, keep_n.g, inv_p.s, inv_n.s], out_tid)

        # 7. intermediate output
        int_out_top_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=0)
        int_out_top = self.connect_to_tracks([inp_minus.s, en_p.s, inp_plus.s, keep_p.s,
                                              keep_n.s, inv_p.g, inv_n.g], int_out_top_tid)

        int_out_bot_tid = self.get_track_id(ridx_n, MOSWireType.DS_GATE, wire_name='sig')
        int_out_bot = self.connect_to_tracks([x_inner_left.d, x_inner_right.d, keep_n.s],
                                             int_out_bot_tid)

        # extend horizontal wires for symmetry
        lower_coord = min(out.lower, int_out_bot.lower, int_out_top.lower)
        upper_coord = max(out.upper, int_out_bot.upper, int_out_top.upper)
        for _tid in [out_tid, int_out_top_tid, int_out_bot_tid]:
            self.add_wires(hm_layer, _tid.base_index, lower=lower_coord, upper=upper_coord,
                           width=_tid.width)

        self.add_pin('VDD', vdd)
        self.add_pin('VSS', vss)
        self.add_pin('out', out)
        self.add_pin('vgate_minus', vgate_minus)
        self.add_pin('vgate_plus', vgate_plus)
        self.add_pin('rx_en', rx_en)
        self.add_pin('rstd', rstd)

        sch_params = dict(
            lch=pinfo.lch,
            w_p=pinfo.get_row_place_info(ridx_p).row_info.width,
            w_n=pinfo.get_row_place_info(ridx_n).row_info.width,
            th_p=pinfo.get_row_place_info(ridx_p).row_info.threshold,
            th_n=pinfo.get_row_place_info(ridx_n).row_info.threshold,
        )
        # set properties
        self.sch_params = dict(
            core_params=dict(
                seg_dict=dict(
                    en_p=seg_en_p,
                    inner_n=seg_en_n,
                    inp=seg_inp,
                ),
                **sch_params,
            ),
            inv_keep_params=dict(
                seg=seg_keep,
                **sch_params,
            ),
            inv_out_params=dict(
                seg=seg_inv,
                **sch_params,
            ),
        )


class InvNOR2Sym(MOSBase):
    """The complex gate for symmetric inverter --> NOR2

    Assumes:

    1. PMOS row above NMOS row.
    2. PMOS gate connections on bottom, NMOS gate connections on top
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag3_digital__inv_nor2_sym

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='Dictionary with number of segments.',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg_dict: Dict[str, int] = self.params['seg_dict']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']

        for seg_key, seg_val in seg_dict.items():
            if seg_val % 2 == 1:
                raise ValueError(f'This generator does not support odd number {seg_val} of '
                                 f'segments for {seg_key}')

        seg_inv_p = seg_dict['inv_p']
        seg_inv_n = seg_dict['inv_n']
        seg_nor_p = seg_dict['nor_p']
        seg_nor_n = seg_dict['nor_n']
        seg_sep = self.min_sep_col + (seg_inv_p // 2) % 2

        if seg_nor_p < seg_nor_n:
            raise ValueError(f'This generator works if segments of NOR pch ({seg_nor_p}) is greater'
                             f' than or equal to segments of NOR nch ({seg_nor_n})')
        seg_inv = max(seg_inv_p, seg_inv_n)
        seg_tot = seg_nor_p + seg_sep + seg_inv + seg_sep + seg_nor_p
        self.set_mos_size(seg_tot)

        # --- Placement --- #
        left_col = 0

        offset_nor_n = (seg_nor_p - seg_nor_n) // 2
        offset_inv_n = (seg_inv - seg_inv_n) // 2
        offset_inv_p = (seg_inv - seg_inv_p) // 2
        g_on_s = offset_nor_n % 2 == 0
        if (seg_inv // 2 + seg_sep + offset_nor_n) % 2 == 1:
            sup_port, sig_port = MOSPortType.D, MOSPortType.S
        else:
            sup_port, sig_port = MOSPortType.S, MOSPortType.D

        nor_left_p = self.add_mos(ridx_p, left_col, seg_nor_p // 2, w=w_p, stack=2, g_on_s=True,
                                  sep_g=True)
        nor_slow_n = self.add_mos(ridx_n, left_col + offset_nor_n, seg_nor_n, w=w_n, g_on_s=g_on_s)
        left_col += seg_nor_p + seg_sep

        inv_p = self.add_mos(ridx_p, left_col + offset_inv_p, seg_inv_p, w=w_p)
        inv_n = self.add_mos(ridx_n, left_col + offset_inv_n, seg_inv_n, w=w_n)
        left_col += seg_inv + seg_sep

        nor_right_p = self.add_mos(ridx_p, left_col, seg_nor_p // 2, w=w_p, stack=2, g_on_s=True,
                                   sep_g=True)
        nor_fast_n = self.add_mos(ridx_n, left_col + offset_nor_n, seg_nor_n, w=w_n, g_on_s=g_on_s)

        assert left_col + seg_nor_p == seg_tot, "Check row placement"

        # --- Routing --- #
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1

        # 1. Inverter input
        in_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=0)
        in_hm = self.connect_to_tracks([inv_p.g, inv_n.g], in_tid, min_len_mode=MinLenMode.MIDDLE)

        # 2. Reset input to NOR2
        rst_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=-1)
        rstd = self.connect_to_tracks([nor_left_p.g1, nor_right_p.g1, nor_fast_n.g], rst_tid)

        # 3. Inverter output to NOR input
        mid_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=1)
        self.connect_to_tracks([nor_left_p.g0, nor_right_p.g0, nor_slow_n.g, inv_p.s, inv_n.s],
                               mid_tid)

        # 4. vdd
        vdd_tid = self.get_track_id(ridx_p, MOSWireType.DS_GATE, wire_name='sup')
        vdd = self.connect_to_tracks([nor_left_p.s, nor_right_p.s, inv_p.d], vdd_tid)

        # 5. vss
        vss_tid = self.get_track_id(ridx_n, MOSWireType.DS_GATE, wire_name='sup')
        vss = self.connect_to_tracks([nor_slow_n[sup_port], inv_n.d, nor_fast_n[sup_port]], vss_tid)

        # 6. output node
        nd_tid = self.get_track_id(ridx_n, MOSWireType.DS_GATE, wire_name='sig')
        out_n = self.connect_to_tracks([nor_slow_n[sig_port], nor_fast_n[sig_port]], nd_tid)

        pd_tid = self.get_track_id(ridx_p, MOSWireType.DS_GATE, wire_name='sig')
        out_p = self.connect_to_tracks([nor_left_p.d, nor_right_p.d], pd_tid)

        out_left_idx = self.tr_manager.get_next_track(self.conn_layer,
                                                      inv_p.s[0].track_id.base_index,
                                                      'sig', 'sig', up=False)
        out_right_idx = self.tr_manager.get_next_track(self.conn_layer,
                                                       inv_p.s[1].track_id.base_index,
                                                       'sig', 'sig', up=True)
        out_tid = TrackID(self.conn_layer, out_left_idx, num=2, pitch=out_right_idx-out_left_idx)
        out = self.connect_to_tracks([out_p, out_n], out_tid)

        self.add_pin('VDD', vdd)
        self.add_pin('VSS', vss)
        self.add_pin('out', out)
        self.add_pin('rstd', rstd)
        self.add_pin('in', in_hm)

        self.add_pin('out_p_hm', out_p, hide=True)
        self.add_pin('out_n_hm', out_n, hide=True)

        sch_params = dict(
            lch=pinfo.lch,
            w_p=pinfo.get_row_place_info(ridx_p).row_info.width,
            w_n=pinfo.get_row_place_info(ridx_n).row_info.width,
            th_p=pinfo.get_row_place_info(ridx_p).row_info.threshold,
            th_n=pinfo.get_row_place_info(ridx_n).row_info.threshold,
        )
        # set properties
        self.sch_params = dict(
            inv_params=dict(
                seg_p=seg_inv_p,
                seg_n=seg_inv_n,
                **sch_params,
            ),
            nor_params=dict(
                seg_p=seg_nor_p,
                seg_n=seg_nor_n,
                **sch_params,
            ),
        )
