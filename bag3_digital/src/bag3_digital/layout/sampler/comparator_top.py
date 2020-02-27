"""This module contains layout generators for top level of comparator."""

from typing import Any, Dict, Type, Optional

from pybag.enum import MinLenMode, RoundMode

from bag.util.immutable import Param
from bag.layout.template import TemplateDB
from bag.layout.routing.base import TrackID
from bag.design.module import Module

from xbase.layout.enum import MOSWireType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase
from xbase.layout.mos.placement.data import TilePatternElement

from .complex_logic_gates import InvNOR2Sym, ResetRow
from .sr_latch import SRLatchSymmetric
from .DynOffComp_v2 import DynOffCompCap
from ..stdcells.gates import NAND2Core

from ...schematic.output_stage import bag3_digital__output_stage
from ...schematic.dyn_off_comp_w_out import bag3_digital__dyn_off_comp_w_out


class OutputStage(MOSBase):
    """The top level of gated SR latch output stage

    Tile order (top to bottom):
    2. NAND2Core, InvNOR2Sym, NAND2Core
    1. ResetRow
    0. SRLatch
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag3_digital__output_stage

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            latch_seg_dict='Segments for SR latch',
            rstrow_seg_dict='Segments for Reset Row',
            invnor2_seg_dict='Segments for Inv->NOR2',
            nand2_seg='Segments for NAND2',
            draw_taps='True to draw substrate taps.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(draw_taps=True)

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        # create masters
        # 1. SR latch
        latch_params = dict(
            pinfo=pinfo[1]['latch_tile'],
            seg_dict=self.params['latch_seg_dict'],
        )
        latch_master = self.new_template(SRLatchSymmetric, params=latch_params)
        latch_n_cols = latch_master.num_cols

        # 2. Reset Row
        rstrow_params = dict(
            pinfo=pinfo[1]['reset_tile'],
            seg_dict=self.params['rstrow_seg_dict'],
        )
        rstrow_master = self.new_template(ResetRow, params=rstrow_params)
        rstrow_n_cols = rstrow_master.num_cols

        # 3. Inv -> NOR2
        invnor2_params = dict(
            pinfo=pinfo[1]['logic_tile'],
            seg_dict=self.params['invnor2_seg_dict'],
        )
        invnor2_master = self.new_template(InvNOR2Sym, params=invnor2_params)
        invnor2_n_cols = invnor2_master.num_cols

        # 4. NAND2
        logic_pinfo = pinfo[1]['logic_tile']
        logic_tp = TilePatternElement(logic_pinfo)
        nand2_params = dict(
            pinfo=logic_pinfo,
            seg=self.params['nand2_seg'],
            sig_locs=dict(
              pin0=logic_tp.get_track_index(1, MOSWireType.G, wire_name='sig', wire_idx=-1),
              pin1=logic_tp.get_track_index(1, MOSWireType.G, wire_name='sig', wire_idx=-2),
            ),
        )
        nand2_master = self.new_template(NAND2Core, params=nand2_params)
        nand2_n_cols = nand2_master.num_cols

        seg_sep = self.min_sep_col

        # set total number of columns
        seg_tot = max(latch_n_cols, rstrow_n_cols, invnor2_n_cols + 2 * nand2_n_cols + 2 *
                      seg_sep)

        # taps on left and right
        draw_taps: bool = self.params['draw_taps']
        if draw_taps:
            tap_n_cols = self.get_tap_ncol()
            tap_sep_col = self.sub_sep_col
            seg_tot += 2 * (tap_sep_col + tap_n_cols)

        # --- Placement --- #
        seg_tot2 = seg_tot // 2

        latch_inst = self.add_tile(latch_master, 0, seg_tot2 - latch_n_cols // 2)
        rstrow_inst = self.add_tile(rstrow_master, 1, seg_tot2 - rstrow_n_cols // 2)
        invnor2_inst = self.add_tile(invnor2_master, 2, seg_tot2 - invnor2_n_cols // 2)
        nand2_left_inst = self.add_tile(nand2_master, 2, seg_tot2 - invnor2_n_cols // 2 - seg_sep,
                                        flip_lr=True)
        nand2_right_inst = self.add_tile(nand2_master, 2, seg_tot2 + invnor2_n_cols // 2 + seg_sep)

        # add taps
        tap_vdd_list, tap_vss_list = [], []
        if draw_taps:
            for tile_idx in range(self.num_tile_rows):
                self.add_tap(0, tap_vdd_list, tap_vss_list, tile_idx=tile_idx)
                self.add_tap(seg_tot, tap_vdd_list, tap_vss_list, tile_idx=tile_idx, flip_lr=True)

        self.set_mos_size(seg_tot)

        # --- Routing --- #
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1

        # 1. Connect vgate+ and vgate-
        setb_hm = rstrow_inst.get_pin('vgate_minus')
        self.connect_to_track_wires(setb_hm, nand2_left_inst.get_pin('out'))
        latch_sb = latch_inst.get_pin('sb')
        self.connect_to_track_wires(setb_hm, latch_sb)

        rstb_hm = rstrow_inst.get_pin('vgate_plus')
        self.connect_to_track_wires(rstb_hm, nand2_right_inst.get_pin('out'))
        latch_rb = latch_inst.get_pin('rb')
        self.connect_to_track_wires(rstb_hm, latch_rb)

        # 2. Connect output of InvNOR2 to fast inputs of NAND2
        nand_left_in_fast = nand2_left_inst.get_pin('pin<1>')
        nand_right_in_fast = nand2_right_inst.get_pin('pin<1>')
        invnor2_out = invnor2_inst.get_pin('out')
        self.connect_to_track_wires([nand_left_in_fast, nand_right_in_fast], invnor2_out)

        # 3. Connect rstd inputs of ResetRow and InvNOR2
        invnor2_rstd = invnor2_inst.get_pin('rstd')
        rstrow_rstd = rstrow_inst.get_pin('rstd')
        self.add_pin('RSTD', [invnor2_rstd, rstrow_rstd], label='RSTD:')

        # 4. Re-use latch_rb and latch_sb vm_layer tracks for intermediate wire connection
        int_warrs = [invnor2_inst.get_pin('in'), rstrow_inst.get_pin('out')]
        self.connect_to_tracks(int_warrs, latch_sb.track_id)
        self.connect_to_tracks(int_warrs, latch_rb.track_id)

        # 5. Rx_en connection
        rx_en_hm = rstrow_inst.get_pin('rx_en')
        self.add_pin('RX_EN', rx_en_hm)

        # 6. vdd, vss
        vdd_list, vss_list = [], []
        for inst in (latch_inst, rstrow_inst, invnor2_inst, nand2_left_inst, nand2_right_inst):
            vdd_list += inst.get_all_port_pins('VDD')
            vss_list += inst.get_all_port_pins('VSS')
        vdd = self.connect_wires(vdd_list)
        vss = self.connect_wires(vss_list)

        self.connect_to_track_wires(tap_vdd_list, vdd)
        self.connect_to_track_wires(tap_vss_list, vss)

        # add and export pins
        self.add_pin('VDD', vdd, label='VDD:')
        self.add_pin('VSS', vss, label='VSS:')

        self.reexport(nand2_left_inst.get_port('nin<0>'), net_name='VcM', hide=False)
        self.reexport(nand2_right_inst.get_port('nin<0>'), net_name='VcP', hide=False)

        self.reexport(latch_inst.get_port('qb'), net_name='OUTP')
        self.reexport(latch_inst.get_port('q'), net_name='OUTM')

        # set properties
        self.sch_params = dict(
            latch_params=latch_master.sch_params,
            rstrow_params=rstrow_master.sch_params,
            invnor2_params=invnor2_master.sch_params,
            nand2_params=nand2_master.sch_params,
        )


class DynOffCompWOut(MOSBase):
    """The dynamic offset cancelled comparator with output stage"""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag3_digital__dyn_off_comp_w_out

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            comp_params='Parameters for Comparator',
            out_params='Parameters for Output stage',
            draw_taps='True to draw substrate taps.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(draw_taps=True)

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        comp_params: Dict[str, Any] = self.params['comp_params']
        out_params: Dict[str, Any] = self.params['out_params'].copy(append=dict(draw_taps=False))
        draw_taps: bool = self.params['draw_taps']

        # create masters
        # 1. Comparator
        comp_params = dict(
            mode=comp_params['mode'],
            cap_in_params=comp_params['cap_in_params'],
            cap_cross_params=comp_params['cap_cross_params'],
            doc_params=dict(
                pinfo=pinfo[1]['comp_tile'],
                seg_dict=comp_params['doc_params']['seg_dict'],
            ),
            draw_taps=False,
        )
        comp_master = self.new_template(DynOffCompCap, params=comp_params)
        comp_n_cols = comp_master.num_cols

        # 2. Output stage
        out_params = dict(
            pinfo=pinfo,
            **out_params
        )
        out_master = self.new_template(OutputStage, params=out_params)
        out_n_cols = out_master.num_cols

        seg_tot = max(comp_n_cols, out_n_cols)

        # taps on left and right
        if draw_taps:
            tap_n_cols = self.get_tap_ncol()
            tap_sep_col = self.sub_sep_col
            seg_tot += 2 * (tap_sep_col + tap_n_cols)

        # --- Placement --- #
        seg_tot2 = seg_tot // 2

        out_inst = self.add_tile(out_master, 0, seg_tot2 - out_n_cols // 2)
        comp_inst = self.add_tile(comp_master, 3, seg_tot2 - comp_n_cols // 2)

        # add taps
        tap_vdd_list, tap_vss_list = [], []
        if draw_taps:
            for tile_idx in range(self.num_tile_rows):
                self.add_tap(0, tap_vdd_list, tap_vss_list, tile_idx=tile_idx)
                self.add_tap(seg_tot, tap_vdd_list, tap_vss_list, tile_idx=tile_idx, flip_lr=True)

        self.set_mos_size(seg_tot)

        # --- Routing --- #
        tr_manager = self.tr_manager
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1

        # 1. Connect RSTD signals of comparator and output stage
        rstd_out = out_inst.get_all_port_pins('RSTD')
        rstd_comp = comp_inst.get_pin('RSTD')
        rstd_idx = self.grid.coord_to_track(vm_layer, rstd_comp.middle, mode=RoundMode.NEAREST)
        rstd_warrs = [rstd_comp] + rstd_out
        clk_vm_w = tr_manager.get_width(vm_layer, 'clk')
        rstd = self.connect_to_tracks(rstd_warrs, TrackID(vm_layer, rstd_idx, width=clk_vm_w))

        # 2. connect RST signals of comparator and clkbuf
        rst_comp = comp_inst.get_pin('RST')
        rst0_idx = tr_manager.get_next_track(vm_layer, rstd_idx, 'clk', 'clk', up=False)
        rst1_idx = tr_manager.get_next_track(vm_layer, rstd_idx, 'clk', 'clk', up=True)
        rst_tid = TrackID(vm_layer, rst0_idx, num=2, pitch=rst1_idx - rst0_idx,
                          width=tr_manager.get_width(vm_layer, 'clk'))
        rst = self.connect_to_tracks([rst_comp], rst_tid, min_len_mode=MinLenMode.MIDDLE)

        # 3. Connect RX_EN signals of comparator and output stage
        rx_en0_idx = tr_manager.get_next_track(vm_layer, rst0_idx, 'clk', 'sig', up=False)
        rx_en1_idx = tr_manager.get_next_track(vm_layer, rst1_idx, 'clk', 'sig', up=True)
        sig_vm_w = tr_manager.get_width(vm_layer, 'sig')
        rx_en_tid = TrackID(vm_layer, rx_en0_idx, num=2, pitch=rx_en1_idx - rx_en0_idx,
                            width=sig_vm_w)
        rx_en_out = out_inst.get_all_port_pins('RX_EN')
        rx_en_comp = comp_inst.get_pin('RX_EN')
        rx_en = self.connect_to_tracks([rx_en_comp] + rx_en_out, rx_en_tid)

        # 4. Connect all outputs of comparator to input of output stage
        vc_minus, vc_plus = out_inst.get_pin('VcM'), out_inst.get_pin('VcP')

        # a) Find the vertical metal positions based on VOP_sw, VOM_sw
        vom_sw, vop_sw = comp_inst.get_pin('VOM_sw'), comp_inst.get_pin('VOP_sw')
        vom_vm_idx = self.grid.coord_to_track(vm_layer, vom_sw.middle, mode=RoundMode.NEAREST)
        vop_vm_idx = self.grid.coord_to_track(vm_layer, vop_sw.middle, mode=RoundMode.NEAREST)

        # b) Connect VOP_bot and VOM_bot
        vom_bot, vop_bot = comp_inst.get_pin('VOM_bot'), comp_inst.get_pin('VOP_bot')
        comp_m, comp_p = self.connect_differential_tracks(vom_bot, vop_bot, vm_layer,
                                                          vom_vm_idx, vop_vm_idx, width=sig_vm_w)

        comp_m = self.connect_to_track_wires([vc_minus, vom_sw], comp_m)
        comp_p = self.connect_to_track_wires([vc_plus, vop_sw], comp_p)

        # c) Connect VOP_top, VOM_top
        vom_top, vop_top = comp_inst.get_pin('VOM_top'), comp_inst.get_pin('VOP_top')
        self.connect_differential_tracks(vom_top, vop_top, vm_layer, vom_vm_idx, vop_vm_idx,
                                         width=sig_vm_w)

        # d) Connect VOP_cap, VOM_cap
        vom_cap, vop_cap = comp_inst.get_pin('VOM_cap'), comp_inst.get_pin('VOP_cap')
        self.connect_to_track_wires(vom_cap, comp_m)
        self.connect_to_track_wires(vop_cap, comp_p)

        # e) Make pseudo-differential connection, if necessary
        if comp_params['mode'] == 1:
            vpreP = comp_inst.get_pin('VpreP', layer=xm_layer)
            coord = vpreP.upper + self.grid.get_line_end_space(xm_layer, vpreP.track_id.num)
            vom_cap1 = comp_inst.get_pin('VOM_cap1')
            self.connect_to_tracks(comp_m, vom_cap1.track_id, track_lower=coord,
                                   track_upper=vom_cap1.upper)

        self.add_pin('COMP_P', comp_p)
        self.add_pin('COMP_M', comp_m)

        # add and export pins
        self.add_pin('RSTD', rstd)
        self.add_pin('RST', rst, label='RST:')
        self.add_pin('RX_EN', rx_en, label='RX_EN:')

        # supplies
        vdd_comp = comp_inst.get_all_port_pins('VDD')
        vss_comp = comp_inst.get_all_port_pins('VSS')
        vdd_out = out_inst.get_all_port_pins('VDD')
        vss_out = out_inst.get_all_port_pins('VSS')

        vdd = self.connect_to_track_wires(tap_vdd_list, vdd_comp + vdd_out)
        vss = self.connect_to_track_wires(tap_vss_list, vss_comp + vss_out)

        self.add_pin('VDD', vdd, connect=True)
        self.add_pin('VSS', vss, connect=True)

        for name in comp_inst.port_names_iter():
            if name in ['Vb', 'VpreP', 'SW_P', 'SW_M', 'VIP', 'VIM']:
                self.reexport(comp_inst.get_port(name))

        for name in out_inst.port_names_iter():
            if name in ['OUTP', 'OUTM']:
                self.reexport(out_inst.get_port(name))

        # set properties
        remove_pins_list = []
        if comp_params['mode'] == 1:
            remove_pins_list.append('VpreM')
        self.sch_params = dict(
            comp_params=comp_master.sch_params,
            out_params=out_master.sch_params,
            remove_pins_list=remove_pins_list,
        )
