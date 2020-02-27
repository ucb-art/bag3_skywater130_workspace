"""This module contains layout generators for front-end of the comparators."""

from typing import Any, Dict, Type, Optional

from pybag.enum import MinLenMode, RoundMode

from bag.util.immutable import Param
from bag.layout.template import TemplateDB
from bag.design.module import Module

from xbase.layout.enum import MOSWireType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase
from xbase.layout.mos.guardring import GuardRing

from ..stdcells.levelshifter import LevelShifterCoreOutBuffer

from ...schematic.break_before_make_buf import bag3_digital__break_before_make_buf
from ...schematic.front_end_switch import bag3_digital__front_end_switch


class FrontEndSwitch(MOSBase):
    """The front-end switches for the comparator
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag3_digital__front_end_switch

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='Dictionary with number of segments.',
            w_n='nmos width.',
            ridx_n_ref='nmos row index for reference signal.',
            ridx_n_sig='nmos row index for input signal.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_n=0,
            ridx_n_ref=0,
            ridx_n_sig=1,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg_dict: Dict[str, int] = self.params['seg_dict']
        w_n: int = self.params['w_n']
        ridx_n_ref: int = self.params['ridx_n_ref']
        ridx_n_sig: int = self.params['ridx_n_sig']

        seg_sig = seg_dict['seg_in']
        seg_ref = seg_dict['seg_ref']

        tap_n_cols = self.get_tap_ncol()
        sub_sep = self.sub_sep_col

        seg_tot = tap_n_cols + sub_sep + max(seg_sig, seg_ref) + sub_sep + tap_n_cols
        self.set_mos_size(seg_tot)

        # --- Placement --- #
        sig_mos = self.add_mos(ridx_n_sig, seg_tot // 2 - seg_sig // 2, seg_sig, w=w_n)
        ref_mos = self.add_mos(ridx_n_ref, seg_tot // 2 - seg_ref // 2, seg_ref, w=w_n)

        tap_vss_list = []
        self.add_tap(0, [], tap_vss_list)
        self.add_tap(seg_tot, [], tap_vss_list, flip_lr=True)
        self.add_pin('VSS', tap_vss_list, label='VSS:')

        # --- Routing --- #

        # 1. reset and reset_bar
        rst_tid = self.get_track_id(ridx_n_ref, MOSWireType.G, wire_name='clk')
        rstb_tid = self.get_track_id(ridx_n_sig, MOSWireType.G, wire_name='clk')

        rst = self.connect_to_tracks(ref_mos.g, rst_tid, min_len_mode=MinLenMode.MIDDLE)
        rstb = self.connect_to_tracks(sig_mos.g, rstb_tid, min_len_mode=MinLenMode.MIDDLE)

        # 2. signal and reference
        sig_in_tid = self.get_track_id(ridx_n_sig, MOSWireType.DS, wire_name='sig')
        ref_in_tid = self.get_track_id(ridx_n_ref, MOSWireType.DS, wire_name='sig', wire_idx=0)

        sig_in = self.connect_to_tracks(sig_mos.d, sig_in_tid, min_len_mode=MinLenMode.MIDDLE)
        ref_in = self.connect_to_tracks(ref_mos.d, ref_in_tid, min_len_mode=MinLenMode.MIDDLE)

        # 3. output
        out_tid = self.get_track_id(ridx_n_ref, MOSWireType.DS, wire_name='sig', wire_idx=1)

        out = self.connect_to_tracks([sig_mos.s, ref_mos.s], out_tid)

        # pins
        self.add_pin('RST', rst)
        self.add_pin('RSTB', rstb)
        self.add_pin('v_sig_in', sig_in)
        self.add_pin('v_ref_in', ref_in)
        self.add_pin('v_out', out)

        # set properties
        if isinstance(pinfo, MOSBasePlaceInfo):
            _pinfo = pinfo
        else:
            _pinfo = pinfo[1]['switch_tile']
        self.sch_params = dict(
            seg_dict=dict(
                sig=seg_sig,
                ref=seg_ref,
            ),
            lch=_pinfo.lch,
            w_dict=dict(
                sig=_pinfo.get_row_place_info(ridx_n_sig).row_info.width,
                ref=_pinfo.get_row_place_info(ridx_n_ref).row_info.width,
            ),
            th_dict=dict(
                sig=_pinfo.get_row_place_info(ridx_n_sig).row_info.threshold,
                ref=_pinfo.get_row_place_info(ridx_n_ref).row_info.threshold,
            ),
        )


class FrontEndSwitchDual(MOSBase):
    """Dual front-end switches for the comparator
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag3_digital__front_end_switch

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='Dictionary with number of segments.',
            w_n='nmos width.',
            ridx_n_ref='nmos row index for reference signal.',
            ridx_n_sig='nmos row index for input signal.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_n=0,
            ridx_n_ref=0,
            ridx_n_sig=1,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg_dict: Dict[str, int] = self.params['seg_dict']
        w_n: int = self.params['w_n']
        ridx_n_ref: int = self.params['ridx_n_ref']
        ridx_n_sig: int = self.params['ridx_n_sig']

        seg_sig = seg_dict['seg_in']
        seg_ref = seg_dict['seg_ref']

        tap_n_cols = self.get_tap_ncol()
        sub_sep = self.sub_sep_col

        seg_tot = 2 * max(seg_sig, seg_ref) + 2 * sub_sep + tap_n_cols
        self.set_mos_size(seg_tot)

        # --- Placement --- #
        sig_left = self.add_mos(ridx_n_sig, 0, seg_sig, w=w_n)
        ref_left = self.add_mos(ridx_n_ref, 0, seg_ref, w=w_n)

        sig_right = self.add_mos(ridx_n_sig, seg_tot, seg_sig, w=w_n, flip_lr=True)
        ref_right = self.add_mos(ridx_n_ref, seg_tot, seg_ref, w=w_n, flip_lr=True)

        tap_vss_list = []
        self.add_tap(max(seg_sig, seg_ref) + sub_sep, [], tap_vss_list)
        self.add_pin('VSS', tap_vss_list, label='VSS:')

        # --- Routing --- #

        # 1. reset and reset_bar
        rst_tid = self.get_track_id(ridx_n_ref, MOSWireType.G, wire_name='clk')
        rstb_tid = self.get_track_id(ridx_n_sig, MOSWireType.G, wire_name='clk')

        rst_left = self.connect_to_tracks(ref_left.g, rst_tid, min_len_mode=MinLenMode.MIDDLE)
        rstb_left = self.connect_to_tracks(sig_left.g, rstb_tid, min_len_mode=MinLenMode.MIDDLE)

        rst_right = self.connect_to_tracks(ref_right.g, rst_tid, min_len_mode=MinLenMode.MIDDLE)
        rstb_right = self.connect_to_tracks(sig_right.g, rstb_tid, min_len_mode=MinLenMode.MIDDLE)

        # 2. signal and reference
        sig_in_tid = self.get_track_id(ridx_n_sig, MOSWireType.DS, wire_name='sig')
        ref_in_tid = self.get_track_id(ridx_n_ref, MOSWireType.DS, wire_name='sig', wire_idx=0)

        sig_in_left = self.connect_to_tracks(sig_left.d, sig_in_tid, min_len_mode=MinLenMode.MIDDLE)
        ref_in_left = self.connect_to_tracks(ref_left.d, ref_in_tid, min_len_mode=MinLenMode.MIDDLE)

        sig_in_right = self.connect_to_tracks(sig_right.d, sig_in_tid,
                                              min_len_mode=MinLenMode.MIDDLE)
        ref_in_right = self.connect_to_tracks(ref_right.d, ref_in_tid,
                                              min_len_mode=MinLenMode.MIDDLE)

        # 3. output
        out_tid = self.get_track_id(ridx_n_ref, MOSWireType.DS, wire_name='sig', wire_idx=1)

        out_left = self.connect_to_tracks([sig_left.s, ref_left.s], out_tid)
        out_right = self.connect_to_tracks([sig_right.s, ref_right.s], out_tid)

        # pins
        self.add_pin('RST_left', rst_left)
        self.add_pin('RSTB_left', rstb_left)
        self.add_pin('v_sig_in_left', sig_in_left)
        self.add_pin('v_ref_in_left', ref_in_left)
        self.add_pin('v_out_left', out_left)

        self.add_pin('RST_right', rst_right)
        self.add_pin('RSTB_right', rstb_right)
        self.add_pin('v_sig_in_right', sig_in_right)
        self.add_pin('v_ref_in_right', ref_in_right)
        self.add_pin('v_out_right', out_right)

        # set properties
        if isinstance(pinfo, MOSBasePlaceInfo):
            _pinfo = pinfo
        else:
            _pinfo = pinfo[1]['switch_tile']
        self.sch_params = dict(
            seg_dict=dict(
                sig=seg_sig,
                ref=seg_ref,
            ),
            lch=_pinfo.lch,
            w_dict=dict(
                sig=_pinfo.get_row_place_info(ridx_n_sig).row_info.width,
                ref=_pinfo.get_row_place_info(ridx_n_ref).row_info.width,
            ),
            th_dict=dict(
                sig=_pinfo.get_row_place_info(ridx_n_sig).row_info.threshold,
                ref=_pinfo.get_row_place_info(ridx_n_ref).row_info.threshold,
            ),
        )


class FrontEndSwitchGuardRing(GuardRing):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        GuardRing.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        ans = FrontEndSwitch.get_params_info()
        ans.update(
            dual='True to use FrontEndSwitchDual instead of FrontEndSwitch',
            nmos_gr='nmos guard ring tile name.',
            pmos_gr='pmos guard ring tile name.',
            edge_ncol='Number of columns on guard ring edge.  Use 0 for default.',
        )
        return ans

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        ans = FrontEndSwitch.get_default_param_values()
        ans.update(
            nmos_gr='ngr',
            pmos_gr='pgr',
            edge_ncol=0,
        )
        return ans

    def get_layout_basename(self) -> str:
        return self.__class__.__name__

    def draw_layout(self) -> None:
        params = self.params
        dual: bool = params['dual']
        nmos_gr: str = params['nmos_gr']
        pmos_gr: str = params['pmos_gr']
        edge_ncol: int = params['edge_ncol']

        core_params = params.copy(remove=['nmos_gr', 'pmos_gr', 'edge_ncol'])
        fes_temp = FrontEndSwitchDual if dual else FrontEndSwitch
        master = self.new_template(fes_temp, params=core_params)

        sub_sep = master.sub_sep_col
        sep_ncol_left = sep_ncol_right = sub_sep
        sep_ncol = (sep_ncol_left, sep_ncol_right)

        inst, sup_list = self.draw_guard_ring(master, pmos_gr, nmos_gr, sep_ncol, edge_ncol)
        vdd_hm_list, vss_hm_list = [], []
        for (vss_list, vdd_list) in sup_list:
            vss_hm_list.extend(vss_list)
            vdd_hm_list.extend(vdd_list)

        self.connect_to_track_wires(vss_hm_list, inst.get_all_port_pins('VSS_vm'))
        self.connect_to_track_wires(vdd_hm_list, inst.get_all_port_pins('VDD_vm'))


class BreakBeforeMakeBuffer(MOSBase):
    """A level above Level Shifter to convert it into a break before make buffer."""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag3_digital__break_before_make_buf

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            levelshifter_params='Params for level shifter',
        )

    def draw_layout(self) -> None:
        ls_params: Param = self.params['levelshifter_params']

        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, ls_params['pinfo'])
        self.draw_base(pinfo)

        tap_n_cols = self.get_tap_ncol()
        sub_sep = self.sub_sep_col

        ls_master = self.new_template(LevelShifterCoreOutBuffer, params=ls_params)

        # --- Placement --- #
        tap_vdd_list, tap_vss_list = [], []
        self.add_tap(0, tap_vdd_list, tap_vss_list)

        ls_inst = self.add_tile(ls_master, 0, tap_n_cols + sub_sep + ls_master.num_cols,
                                flip_lr=True)

        ls_pout = ls_inst.get_pin('pout')
        seg_mid = pinfo.get_source_track_col(self.grid.coord_to_track(self.conn_layer,
                                                                      ls_pout.middle,
                                                                      mode=RoundMode.NEAREST))
        seg_tot = 2 * seg_mid
        self.add_tap(seg_tot, tap_vdd_list, tap_vss_list, flip_lr=True)

        self.set_mos_size(seg_tot)

        # --- Routing --- #
        # 1. Connect rst_h and rst_l
        rst_h = ls_inst.get_pin('rst_h')
        rst_l = ls_inst.get_pin('rst_l')
        rx_enb = self.connect_wires([rst_h, rst_l])

        # 2. Connect rst_hb and rst_lb
        rst_hb = ls_inst.get_pin('rst_hb')
        rst_lb = ls_inst.get_pin('rst_lb')
        rx_en = self.connect_wires([rst_hb, rst_lb])

        # 3. supplies
        vdd = ls_inst.get_pin('VDD')
        self.connect_to_track_wires(tap_vdd_list, vdd)

        vss = ls_inst.get_pin('VSS')
        self.connect_to_track_wires(tap_vss_list, vss)

        # pins
        self.add_pin('RX_EN', rx_en)
        self.add_pin('RX_ENB', rx_enb)
        self.add_pin('VDD', vdd, label='VDD:')
        self.add_pin('VSS', vss, label='VSS:')

        self.reexport(ls_inst.get_port('in'), net_name='CLK')
        self.reexport(ls_inst.get_port('inb'), net_name='CLKB')
        self.reexport(ls_inst.get_port('midp'), net_name='RST', hide=False)
        self.reexport(ls_inst.get_port('midn'), net_name='RSTB', hide=False)
        self.reexport(ls_inst.get_port('pout'), label='RSTD:')
        self.reexport(ls_inst.get_port('nout'), label='RSTD:')

        # set properties
        self.sch_params = dict(
            levelshifter_params=ls_master.sch_params,
        )
