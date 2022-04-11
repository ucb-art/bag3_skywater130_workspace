from typing import Any, Dict, List, Union

from bag.layout.core import PyLayInstance
from bag.layout.routing.base import WireArray

from bag.util.immutable import Param
from bag.layout.template import TemplateDB
from bag.layout.routing.base import TrackID

from xbase.layout.enum import MOSWireType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase

from ..stdcells.gates import NAND2Core, NOR2Core, InvChainCore


class NColDecoder(MOSBase):
    """
    N bit Column Decoder

    Assumes:
        nbit should be odd (TODO for later)
        Inputs come in from right side on metal conn_layer + 1
        From bottom to top inputs are a,a',b,b',...,z,z'
        Outputs are at top on metal conn_layer + 2, en is on right enb is on left
        The left most output is abc...z and the right most is a'b'c'...z'
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='seg of sub cell (nand, nor, passgates)',
            nbits='number of bits',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            tap_rate='the column rate at which taps are inserted',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            tap_rate=20,
        )

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        nbits: int = self.params['nbits']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        tap_rate: int = self.params['tap_rate']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        show_pins: bool = self.show_pins

        assert self.params['seg'] % 2 == 0, "seg should be even"
        assert self.params['nbits'] % 2 != 0, "nbits should be odd"

        nouts = 2 ** nbits
        ntiles = nbits  # nbits-1 is the number of nand/nor stages and 1 is the number of last inv

        inv_params = dict(
            pinfo=self.params['pinfo'],
            seg_list=[self.params['seg']],
            show_pins=False,
            vertical_out=False,
        )

        nand_params = dict(
            pinfo=self.params['pinfo'],
            seg=self.params['seg']//2,
            show_pins=False,
            vertical_in=True,
            vertical_out=False,
            connect_inputs=False,
        )
        nor_params = dict(
            pinfo=self.params['pinfo'],
            seg=self.params['seg'] // 2,
            show_pins=False,
            vertical_in=True,
            vertical_out=False,
            connect_inputs=False,
        )

        inv_master = self.new_template(InvChainCore, params=inv_params)
        nand_master = self.new_template(NAND2Core, params=nand_params)
        nor_master = self.new_template(NOR2Core, params=nor_params)

        assert inv_master.num_cols == nand_master.num_cols, "inv and nand have different n_cols"
        assert nand_master.num_cols == nor_master.num_cols, "nand and nor have different n_cols"

        master_n_cols = inv_master.num_cols
        sep_min = self.min_sep_col
        sub_sep = self.sub_sep_col
        col_delta = master_n_cols + sep_min
        vss_list, vdd_list = [], []

        inst_list: List[List[PyLayInstance]] = [[] for _ in range(ntiles)]
        en_list, enb_list = [], []
        inv_input_m2_list = []
        row_masters = []
        for r in reversed(range(ntiles)):
            cur_col = 0

            # figure out the row master
            if r == ntiles - 1:
                master = inv_master
                row_masters.append('inv')
            elif r % 2 == 0:
                master = nand_master
                row_masters.append('nand')
            else:
                master = nor_master
                row_masters.append('nor')

            for c in range(nouts):
                if c % tap_rate == 0:
                    if cur_col != 0:
                        cur_col += sub_sep - sep_min
                    tap_col = self.add_tap(cur_col, vdd_list, vss_list, tile_idx=r)
                    cur_col += tap_col + sub_sep

                inst = self.add_tile(master, r, cur_col)
                cur_col += col_delta
                inst_list[r].append(inst)

                # at stage i we do the connections to stage i+1
                if r == ntiles - 1:
                    # for inverter get the left and right most track ids on layer 3
                    # and connect en and enb
                    out_m2_warrs = inst.get_all_port_pins('out', layer=self.conn_layer+1)
                    out_m2 = max(out_m2_warrs, key=lambda x: x.base_htr)
                    in_tid = self.get_track_id(ridx_n, MOSWireType.G, tile_idx=r)
                    in_m2 = self.connect_to_tracks(inst.get_pin('in_vm'), in_tid)
                    coord_end = inst.bound_box.xh
                    coord_start = inst.bound_box.xl
                    t_end_index = self.grid.coord_to_track(self.conn_layer+2, coord_end)
                    t_start_index = self.grid.coord_to_track(self.conn_layer+2, coord_start)
                    enb_tid = TrackID(self.conn_layer+2, t_end_index)
                    en_tid = TrackID(self.conn_layer+2, t_start_index)
                    enb_list.append(self.connect_to_tracks(out_m2, enb_tid))
                    en_list.append(self.connect_to_tracks(in_m2, en_tid))
                    inv_input_m2_list.append(in_m2)

                elif r == ntiles - 2:
                    # connection is happening between a nand/nor from stage ntiles-1 to the
                    # inverter in stage ntiles, for track ids we can use out_list/outb_list tids
                    inst_out_m2 = inst.get_pin('out')
                    out_in_tid = enb_list[c].track_id
                    self.connect_to_tracks([inst_out_m2, inv_input_m2_list[c]], out_in_tid)

                else:
                    if r % 2 != 0:
                        # connect output of i to right input of stage i+1 tru right most track id
                        # on layer 3, and also stage i is a nor
                        in1_m1 = inst_list[r + 1][c].get_all_port_pins('in1', layer=self.conn_layer)
                        in1_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sig',
                                                    tile_idx=r+1)
                        in1_m2 = self.connect_to_tracks(in1_m1, in1_tid)
                        out_m2 = inst.get_pin('out')

                        coord_right = inst.bound_box.xh
                        t_index = self.grid.coord_to_track(self.conn_layer + 2, coord_right)
                        self.connect_to_tracks([in1_m2, out_m2], TrackID(self.conn_layer + 2,
                                                                         t_index))

                    else:
                        # connect output of i to left input of stage i+1 tru middle track id
                        # on layer 3, and also state i is a nand
                        in0_m1 = inst_list[r+1][c].get_pin('in0', layer=self.conn_layer)
                        in0_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sig',
                                                    tile_idx=r + 1)
                        in0_m2 = self.connect_to_tracks(in0_m1, in0_tid)
                        out_m2 = inst.get_pin('out')

                        coord_middle = inst.bound_box.xm
                        t_index = self.grid.coord_to_track(self.conn_layer+2, coord_middle)
                        self.connect_to_tracks([in0_m2, out_m2], TrackID(self.conn_layer+2,
                                                                         t_index))

            self.add_tap(cur_col - sep_min + sub_sep, vdd_list, vss_list,
                         flip_lr=True, tile_idx=r)

        # connect first row's inputs
        first_row_inputs = []
        for c in range(nouts):
            # for first row four input come in horizentally
            # doesnt' check if it can fit four tracks, pinfo should provide the space

            in0_m1 = inst_list[0][c].get_all_port_pins('in0', layer=self.conn_layer)
            in1_m1 = inst_list[0][c].get_all_port_pins('in1', layer=self.conn_layer)

            repeat = c // (2 ** (nbits-2))
            a_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sig',
                                      wire_idx=(repeat // 2))
            b_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sig',
                                      wire_idx=(repeat % 2))

            first_row_inputs.append(self.connect_to_tracks(in0_m1, a_tid))
            first_row_inputs.append(self.connect_to_tracks(in1_m1, b_tid))

        list_of_inputs = self.connect_wires(first_row_inputs)

        # connect other rows
        for r in range(1, ntiles-1):
            row_r_inputs = []
            for c in range(nouts):
                repeat = c // (2 ** (nbits - 2 - r))
                if r % 2 != 0:
                    # it is a nor and the input is on the right: in1
                    in1_m1 = inst_list[r][c].get_all_port_pins('in1', layer=self.conn_layer)
                    in1_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sig', tile_idx=r,
                                                wire_idx=repeat % 2)
                    row_r_inputs.append(self.connect_to_tracks(in1_m1, in1_tid))
                else:
                    # it is a nand and the input is on the left: in0
                    in0_m1 = inst_list[r][c].get_all_port_pins('in0', layer=self.conn_layer)
                    in0_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sig', tile_idx=r,
                                                wire_idx=repeat % 2)
                    row_r_inputs.append(self.connect_to_tracks(in0_m1, in0_tid))
            list_of_inputs += self.connect_wires(row_r_inputs)

        for i, in_row in enumerate(list_of_inputs):
            bit_number = nbits-1-i//2
            if i % 2 == 0:
                self.add_pin('in<{}>'.format(bit_number), in_row, show=show_pins)
            else:
                self.add_pin('inb<{}>'.format(bit_number), in_row, show=show_pins)

        for i, (en, enb) in enumerate(zip(en_list, enb_list)):
            self.add_pin("en<{}>".format(i), en, show=show_pins)
            self.add_pin("enb<{}>".format(i), enb, show=show_pins)

        vdd_m2_list, vss_m2_list = [], []
        for r in range(ntiles):
            vdd_row, vss_row = None, None
            for c in range(nouts):
                vdd_warr = inst_list[r][c].get_pin('VDD')
                vss_warr = inst_list[r][c].get_pin('VSS')
                if vdd_row:
                    vdd_row = self.connect_wires([vdd_row, vdd_warr])[0]
                    vss_row = self.connect_wires([vss_row, vss_warr])[0]
                else:
                    vdd_row = vdd_warr
                    vss_row = vss_warr
            vdd_m2_list.append(vdd_row)
            vss_m2_list.append(vss_row)

        self.connect_to_track_wires(vdd_list, vdd_m2_list)
        self.connect_to_track_wires(vss_list, vss_m2_list)
        self.add_pin('VDD', vdd_m2_list, show=show_pins)
        self.add_pin('VSS', vss_m2_list, show=show_pins)

        self.set_mos_size()

        self.sch_params = dict(
            nbits=nbits,
            master_params=dict(
                seg=self.params['seg'],
                lch=self.place_info.lch,
                w_p=self.place_info.get_row_place_info(ridx_n).row_info.width if w_p == 0 else w_p,
                w_n=self.place_info.get_row_place_info(ridx_p).row_info.width if w_n == 0 else w_n,
                th_n=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
                th_p=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
            ),
            row_masters=list(reversed(row_masters)),
        )


class NRowDecoder(MOSBase):
    """
    N bit Row Decoder

    Assumes:
        Inputs come in from bottom side on metal conn_layer + 2
        From right to left inputs are a,a',b,b',...,z,z' (MSB on right)
        Outputs are at left on metal conn_layer + 2, en is the lower enb is the upper one
        From bottom to top outputs are abc...z to a'b'c'...z'

        For 1 bit it's gonna be just wires connecting input to en, the inversion happens in the
        inverter buffer
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='seg of sub cell (nand, nor, passgates)',
            nbits='number of bits',
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

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        nbits: int = self.params['nbits']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        show_pins: bool = self.show_pins

        assert self.params['seg'] % 2 == 0, "seg should be even"

        nouts = 2 ** nbits
        ncol = nbits  # nbits-1 is the number of nand/nor stages and 1 is the number of last inv

        inv_params = dict(
            pinfo=self.params['pinfo'],
            seg_list=[self.params['seg']],
            show_pins=False,
            vertical_out=False,
        )

        nand_params = dict(
            pinfo=self.params['pinfo'],
            seg=self.params['seg']//2,
            show_pins=False,
            vertical_in=True,
            vertical_out=True,
            connect_inputs=False,
        )
        nor_params = dict(
            pinfo=self.params['pinfo'],
            seg=self.params['seg'] // 2,
            show_pins=False,
            vertical_in=True,
            vertical_out=True,
            connect_inputs=False,
        )

        inv_master = self.new_template(InvChainCore, params=inv_params)
        nand_master = self.new_template(NAND2Core, params=nand_params)
        nor_master = self.new_template(NOR2Core, params=nor_params)

        assert inv_master.num_cols == nand_master.num_cols, "inv and nand have different n_cols"
        assert nand_master.num_cols == nor_master.num_cols, "nand and nor have different n_cols"

        master_n_cols = inv_master.num_cols
        sep_min = self.min_sep_col

        # special case for one bit, it is just shorts from in_0 and inb_0 to to (en_0, enb_1) and
        # (enb_0, en_1) respectively
        if nbits == 1:
            self.set_mos_size(master_n_cols)
            en_0_tid = self.get_track_id(ridx_n, MOSWireType.G, tile_idx=0)
            enb_0_tid = self.get_track_id(ridx_p, MOSWireType.G, tile_idx=0)
            en_1_tid = self.get_track_id(ridx_n, MOSWireType.G, tile_idx=1)
            enb_1_tid = self.get_track_id(ridx_p, MOSWireType.G, tile_idx=1)
            upper = self.grid.track_to_coord(self.conn_layer, master_n_cols-1)
            en_0 = self.add_wires(self.conn_layer+1, en_0_tid.base_index, lower=0, upper=upper)
            enb_0 = self.add_wires(self.conn_layer+1, en_1_tid.base_index, lower=0, upper=upper)
            en_1 = self.add_wires(self.conn_layer+1, enb_0_tid.base_index, lower=0, upper=upper)
            enb_1 = self.add_wires(self.conn_layer+1, enb_1_tid.base_index, lower=0, upper=upper)
            self.add_pin('en<0>', en_0, show=show_pins)
            self.add_pin('en<1>', enb_0, show=show_pins)
            self.add_pin('enb<0>', en_1, show=show_pins)
            self.add_pin('enb<1>', enb_1, show=show_pins)
            return

        inst_list: List[List[PyLayInstance]] = [[] for _ in range(nouts)]
        en_list, enb_list = [], []
        next_stage_input: List[List[Union[WireArray, None]]] = [[None]*ncol for _ in range(nouts)]

        # if output nbits is even inverter's output should be enable otherwise input of inverter is
        # enable
        if nbits % 2 == 0:
            out_row = ridx_n
            in_row = ridx_p
        else:
            out_row = ridx_p
            in_row = ridx_n

        row_masters = None
        for r in range(nouts):
            row_masters = []
            for c in range(ncol):
                # figure out the col master
                if c == 0:
                    master = inv_master
                    row_masters.append('inv')
                elif self._is_nand(c, nbits):
                    master = nand_master
                    row_masters.append('nand')
                else:
                    master = nor_master
                    row_masters.append('nor')

                inst = self.add_tile(master, r, c * (master_n_cols + sep_min))
                inst_list[r].append(inst)

                # at stage i we do the connections to stage i+1
                if c == 0:
                    # for inverter input is going to be en (lower m2) the output is enb (upper one)
                    inv_in_vm = inst.get_pin('in_vm')
                    inv_out_vm = inst.get_pin('out_vm')
                    out_tid = self.get_track_id(out_row, MOSWireType.G, wire_name='sig', tile_idx=r)
                    in_tid = self.get_track_id(in_row, MOSWireType.G, wire_name='sig', tile_idx=r)
                    outb = self.connect_to_tracks(inv_out_vm, out_tid)
                    out = self.connect_to_tracks(inv_in_vm, in_tid)

                    if nbits % 2 == 0:
                        en_list.append(outb)
                        enb_list.append(out)
                    else:
                        en_list.append(out)
                        enb_list.append(outb)

                    if nbits > 1:
                        next_stage_input[r][c+1] = out

                elif c == ncol - 1:
                    # last stage (input nand)
                    # connect inputs
                    in0_m1 = inst.get_pin('in0')
                    in1_m1 = inst.get_pin('in1')
                    in0_tid = self.get_track_id(out_row, MOSWireType.G, wire_name='sig',
                                                tile_idx=r, wire_idx=0)
                    in1_tid = self.get_track_id(out_row, MOSWireType.G, wire_name='sig',
                                                tile_idx=r, wire_idx=1)
                    in0_m2 = self.connect_to_tracks(in0_m1, in0_tid)
                    in1_m2 = self.connect_to_tracks(in1_m1, in1_tid)

                    # connect output of this stage to input of next
                    out_vm = inst.get_pin('out_vm')
                    next_in = next_stage_input[r][c]
                    if next_in:
                        self.connect_to_track_wires(out_vm, next_in)
                    else:
                        raise ValueError("next_stage_input[{}][{}] is None".format(r, c))

                    # add pins
                    toggle_rate = 2 ** (c - 1)
                    repeat = r // toggle_rate

                    if repeat // 2 == 0:
                        in0_pin_str = "in<{}>".format(c)
                    else:
                        in0_pin_str = "inb<{}>".format(c)

                    if repeat % 2 == 0:
                        in1_pin_str = "in<{}>".format(c-1)
                    else:
                        in1_pin_str = "inb<{}>".format(c-1)

                    self.add_pin(in0_pin_str, in0_m2, label=in0_pin_str + ':', show=show_pins)
                    self.add_pin(in1_pin_str, in1_m2, label=in1_pin_str + ':', show=show_pins)

                else:
                    # generic stage i
                    in0_m1 = inst.get_pin('in0')
                    in1_m1 = inst.get_pin('in1')
                    in0_tid = self.get_track_id(out_row, MOSWireType.G, wire_name='sig',
                                                tile_idx=r, wire_idx=0)
                    in1_tid = self.get_track_id(out_row, MOSWireType.G, wire_name='sig',
                                                tile_idx=r, wire_idx=1)
                    tid_inputs = {in0_m1: in0_tid, in1_m1: in1_tid}
                    # As we move from row 0 to nout-1 every toggle rate, we toggle the states
                    # of inputs a -> a', determine which pin is extra input
                    toggle_rate = 2 ** (c - 1)
                    repeat = r // toggle_rate
                    if repeat % 2 == 0:
                        extra_in = in0_m1 if self._is_nand(c, nbits) else in1_m1
                        extra_in_pin_str = "in<{}>".format(c-1) if self._is_nand(c, nbits) else \
                            "inb<{}>".format(c-1)
                    else:
                        extra_in = in1_m1 if self._is_nand(c, nbits) else in0_m1
                        extra_in_pin_str = "inb<{}>".format(c-1) if self._is_nand(c, nbits) else \
                            "in<{}>".format(c-1)

                    next_stage_input[r][c+1] = in0_m1 if extra_in == in1_m1 else in1_m1

                    extra_in = self.connect_to_tracks(extra_in, tid_inputs[extra_in])

                    # get the output of the next stage right
                    out0_tid = self.get_track_id(in_row, MOSWireType.G, wire_name='sig',
                                                 tile_idx=r, wire_idx=0)
                    out1_tid = self.get_track_id(in_row, MOSWireType.G, wire_name='sig',
                                                 tile_idx=r, wire_idx=1)
                    if next_stage_input[r][c].base_htr == out0_tid.base_htr:
                        out_tid = out1_tid
                    else:
                        out_tid = out0_tid
                    next_stage_input[r][c+1] = self.connect_to_tracks(next_stage_input[r][c + 1],
                                                                      out_tid)

                    # connect output of this stage to input of next stage
                    out_vm = inst.get_pin('out_vm')
                    next_in = next_stage_input[r][c]
                    if next_in:
                        self.connect_to_track_wires(out_vm, next_in)
                    else:
                        raise ValueError("next_stage_input[{}][{}] is None".format(r, c))

                    # add pin
                    self.add_pin(extra_in_pin_str, extra_in,
                                 label=extra_in_pin_str + ':', show=show_pins)

        for i, (en, enb) in enumerate(zip(en_list, enb_list)):
            self.add_pin(f'en<{i}>', en, show=show_pins)
            self.add_pin(f'enb<{i}>', enb, show=show_pins)

        # add p/n taps for body connection
        tap_vdd_list, tap_vss_list = [], []
        tap_col_idx = self.sub_sep_col + ncol * (master_n_cols + sep_min)
        for r in range(nouts):
            self.add_tap(col_idx=tap_col_idx,
                         vdd_list=tap_vdd_list,
                         vss_list=tap_vss_list,
                         tile_idx=r, flip_lr=True)
        tap_vdd_list = self.connect_wires(tap_vdd_list)
        tap_vss_list = self.connect_wires(tap_vss_list)

        # connect vdd, vss
        vdd_m2_list, vss_m2_list = [], []
        for r in range(nouts):
            vdd_row, vss_row = None, None
            for c in range(ncol):
                vdd_warr = inst_list[r][c].get_pin('VDD')
                vss_warr = inst_list[r][c].get_pin('VSS')
                if vdd_row:
                    vdd_row = self.connect_wires([vdd_row, vdd_warr])[0]
                    vss_row = self.connect_wires([vss_row, vss_warr])[0]
                else:
                    vdd_row = vdd_warr
                    vss_row = vss_warr
            vdd_m2_list.append(vdd_row)
            vss_m2_list.append(vss_row)

        self.connect_to_track_wires(vdd_m2_list, tap_vdd_list)
        self.connect_to_track_wires(vss_m2_list, tap_vss_list)

        self.add_pin('VDD', vdd_m2_list, show=show_pins)
        self.add_pin('VSS', vss_m2_list, show=show_pins)

        self.set_mos_size()

        self.sch_params = dict(
            nbits=nbits,
            master_params=dict(
                seg=self.params['seg'],
                lch=self.place_info.lch,
                w_p=self.place_info.get_row_place_info(ridx_n).row_info.width if w_p == 0 else w_p,
                w_n=self.place_info.get_row_place_info(ridx_p).row_info.width if w_n == 0 else w_n,
                th_n=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
                th_p=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
            ),
            row_masters=list(reversed(row_masters)),
            is_row_decoder=True,
        )

    @staticmethod
    def _is_nand(stage_idx: int, nbits: int) -> bool:
        """
        Determines if stage i is a nand or a nor given number of bits
        :param stage_idx: stage index, for last inverter it is zero
        :param nbits: number of bits
        :return: True if stage i is a nand
        """
        assert stage_idx >= 0, "stage_index should a non_negative number "
        if stage_idx == 0:
            return False
        if nbits % 2 == 0:
            if stage_idx % 2 == 0:
                return False
            return True
        else:
            if stage_idx % 2 == 0:
                return True
            return False


class InvBuffer(MOSBase):
    """
    Inverter buffer

    Assumes:
        For column decoder outputs should be on left from bottom to top in MSB to LSB order
        For row decoder outputs should be on top from left to right in LSB to MSB order
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='seg of sub cell (nand, nor, passgates)',
            n_col_bits='number of bits for column decoder',
            n_row_bits='number of bits for row decoder',
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

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        n_col_bits = self.params['n_col_bits']
        n_row_bits = self.params['n_row_bits']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        show_pins = self.params['show_pins']

        inv_params = dict(
            pinfo=self.params['pinfo'],
            seg_list=[self.params['seg']],
            show_pins=False,
            vertical_out=False,
        )

        inv_master = self.new_template(InvChainCore, params=inv_params)
        inv_ncols = inv_master.num_cols
        sep = self.min_sep_col
        inst_list = [[] for _ in range(n_col_bits)]
        col_bit_counter = n_col_bits - 1

        for r in range(n_col_bits-1):
            inst = self.add_tile(inv_master, r, 0)
            inst_list[r].append(inst)

            self.reexport(inst.get_port('in_vm'), net_name=f'sel<{col_bit_counter}>',
                          show=show_pins)
            self.reexport(inst.get_port('out_vm'), net_name=f'selb<{col_bit_counter}>',
                          show=show_pins)
            col_bit_counter -= 1
            if r == 0:
                inst = self.add_tile(inv_master, r, sep + inv_ncols)
                inst_list[r].append(inst)
                self.reexport(inst.get_port('in_vm'), net_name=f'sel<{col_bit_counter}>',
                              show=show_pins)
                self.reexport(inst.get_port('out_vm'), net_name=f'selb<{col_bit_counter}>',
                              show=show_pins)
                col_bit_counter -= 1

        row_bit_counter = n_col_bits
        for c in range(n_row_bits):
            inst = self.add_tile(inv_master, n_col_bits-1, (2 + c) * (inv_ncols + sep))
            inst_list[-1].append(inst)
            sel_m1 = inst.get_pin('in_vm')
            selb_m1 = inst.get_pin('out_vm')

            # connect sel and selb to left and right most side of the cell
            r_coor = inst.bound_box.xl
            l_coor = inst.bound_box.xh

            sel_htid = self.get_track_id(ridx_n, MOSWireType.G, tile_idx=n_col_bits-1,
                                         wire_name='sig', wire_idx=0)
            selb_htid = self.get_track_id(ridx_n, MOSWireType.G, tile_idx=n_col_bits - 1,
                                          wire_name='sig', wire_idx=1)

            sel_m2 = self.connect_to_tracks(sel_m1, sel_htid)
            selb_m2 = self.connect_to_tracks(selb_m1, selb_htid)
            sel_vtid = TrackID(self.conn_layer + 2,
                               self.grid.coord_to_track(self.conn_layer + 2, r_coor))
            selb_vtid = TrackID(self.conn_layer + 2,
                                self.grid.coord_to_track(self.conn_layer + 2, l_coor))
            sel_m3 = self.connect_to_tracks(sel_m2, sel_vtid)
            selb_m3 = self.connect_to_tracks(selb_m2, selb_vtid)

            self.add_pin(f'sel<{row_bit_counter}>', sel_m3, show=show_pins)
            self.add_pin(f'selb<{row_bit_counter}>', selb_m3, show=show_pins)

            row_bit_counter += 1

        # add taps
        vdd_list, vss_list = [], []
        tap_col_idx = self.sub_sep_col + (2 + n_row_bits) * (inv_ncols + sep)
        for r in range(n_col_bits):
            self.add_tap(tap_col_idx, vdd_list, vss_list, tile_idx=r, flip_lr=True)
        # connect vdd and vss
        vdd_m2_list, vss_m2_list = [], []
        for inst_row in inst_list:
            vdd_row, vss_row = None, None
            for inst in inst_row:
                vdd_warr = inst.get_pin('VDD')
                vss_warr = inst.get_pin('VSS')
                if vdd_row:
                    vdd_row = self.connect_wires([vdd_row, vdd_warr])[0]
                    vss_row = self.connect_wires([vss_row, vss_warr])[0]
                else:
                    vdd_row = vdd_warr
                    vss_row = vss_warr
            vdd_m2_list.append(vdd_row)
            vss_m2_list.append(vss_row)
        # just make sure we don't have redundant warrs
        vdd_m2_list = self.connect_wires(vdd_m2_list)
        vss_m2_list = self.connect_wires(vss_m2_list)
        self.connect_to_track_wires(vdd_list, vdd_m2_list)
        self.connect_to_track_wires(vss_list, vss_m2_list)

        self.add_pin('VDD', vdd_m2_list, show=show_pins)
        self.add_pin('VSS', vss_m2_list, show=show_pins)

        self.set_mos_size()

        self.sch_params = dict(
            num_inputs=n_col_bits + n_row_bits,
            inv_params=dict(
                seg=self.params['seg'],
                lch=self.place_info.lch,
                w_p=self.place_info.get_row_place_info(ridx_n).row_info.width if w_p == 0 else w_p,
                w_n=self.place_info.get_row_place_info(ridx_p).row_info.width if w_n == 0 else w_n,
                th_n=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
                th_p=self.place_info.get_row_place_info(ridx_n).row_info.threshold
            ),
        )
