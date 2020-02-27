from typing import Any, Dict, Union

import fnmatch
import math
import re

from bag.util.immutable import Param
from bag.layout.template import TemplateDB
from bag.layout.routing.base import TrackID, WireArray

from pybag.enum import MinLenMode
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase
from bag3_digital.layout.resdac.switch_arr import SwitchArray
from bag3_digital.layout.resdac.decoder import NColDecoder, NRowDecoder, InvBuffer


class Controller(MOSBase):
    """The entire controller for the RES_DAC array

    Assumes:

    1. All transistors have the same seg number

    2. number of col decoder bits should be more than 3 bit and also odd
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='List of segments per stage.',
            n_cols='number of columns in res dac unit',
            n_rows='number of rows in res dac unit',
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

        n_cols: int = self.params['n_cols']
        n_rows: int = self.params['n_rows']
        show_pins: bool = self.show_pins

        assert math.ceil(math.log2(n_cols)) == math.floor(math.log2(n_cols)), "n_cols should be " \
                                                                              "a power of two"
        assert math.ceil(math.log2(n_rows)) == math.floor(math.log2(n_rows)), "n_rows should be " \
                                                                              "a power of two"

        col_decoder_nbits = int(math.log2(n_cols))
        row_decoder_nbits = int(math.log2(n_rows))

        vm_layer = self.conn_layer + 2

        col_decoder_params = dict(
            pinfo=self.params['pinfo'],
            seg=self.params['seg'],
            nbits=col_decoder_nbits,
            show_pins=False
        )

        row_decoder_params = dict(
            pinfo=self.params['pinfo'],
            seg=self.params['seg'],
            nbits=row_decoder_nbits,
            show_pins=False
        )

        switch_array_params = dict(
            pinfo=self.params['pinfo'],
            seg=self.params['seg'],
            n_cols=n_cols,
            n_rows=n_rows,
            show_pins=False
        )

        inv_buffer_params = dict(
            pinfo=self.params['pinfo'],
            seg=self.params['seg'],
            n_col_bits=int(math.log2(n_cols)),
            n_row_bits=int(math.log2(n_rows)),
            show_pins=False
        )

        col_dec_master = self.new_template(NColDecoder, params=col_decoder_params)
        row_dec_master = self.new_template(NRowDecoder, params=row_decoder_params)
        switch_array_master = self.new_template(SwitchArray, params=switch_array_params)
        inv_buffer_master = self.new_template(InvBuffer, params=inv_buffer_params)

        col_dec_ncols = col_dec_master.num_cols
        col_dec_nrows = col_dec_master.num_tile_rows
        switch_arr_ncols = switch_array_master.num_cols
        inv_buffer_ncols = inv_buffer_master.num_cols
        min_sep = self.min_sep_col
        min_sub_sep = self.sub_sep_col

        x_row_dec = switch_arr_ncols + min_sub_sep
        x_inv_buffer = col_dec_ncols + min_sub_sep
        inst_col_dec = self.add_tile(col_dec_master, 0, 0)
        inst_switch_arr = self.add_tile(switch_array_master, col_dec_nrows + 1, 0)
        inst_row_dec = self.add_tile(row_dec_master, col_dec_nrows + 1, x_row_dec)
        inst_inv_buffer = self.add_tile(inv_buffer_master, 0, x_inv_buffer)
        inst_list = [inst_col_dec, inst_row_dec, inst_switch_arr, inst_inv_buffer]

        col_dec_en_list, col_dec_enb_list = [], []
        col_dec_input_dict = dict()
        for port_name in inst_col_dec.port_names_iter():
            if fnmatch.fnmatch(port_name, 'en*'):
                col_dec_en_list.append(inst_col_dec.get_pin(port_name))
            elif fnmatch.fnmatch(port_name, 'enb*'):
                col_dec_enb_list.append(inst_col_dec.get_pin(port_name))
            elif fnmatch.fnmatch(port_name, 'in*'):
                col_dec_input_dict[port_name] = inst_col_dec.get_pin(port_name)

        row_dec_en_list, row_dec_enb_list = [], []
        row_dec_input_dict = dict()
        for port_name in inst_row_dec.port_names_iter():
            if fnmatch.fnmatch(port_name, 'en<*'):
                row_dec_en_list.append(inst_row_dec.get_pin(port_name))
            elif fnmatch.fnmatch(port_name, 'enb<*'):
                row_dec_enb_list.append(inst_row_dec.get_pin(port_name))
            elif fnmatch.fnmatch(port_name, 'in<*'):
                if port_name in row_dec_input_dict:
                    row_dec_input_dict[port_name] += inst_row_dec.get_all_port_pins(port_name)
                else:
                    row_dec_input_dict[port_name] = inst_row_dec.get_all_port_pins(port_name)

        # special case to deal with one bit of row
        if row_decoder_nbits == 1:
            row_dec_input_dict['in<0>'] = inst_row_dec.get_all_port_pins('en<0>') + \
                                          inst_row_dec.get_all_port_pins('enb<1>')

            row_dec_input_dict['inb<0>'] = inst_row_dec.get_all_port_pins('enb<0>') + \
                                           inst_row_dec.get_all_port_pins('en<1>')

        switch_arr_c_en_list, switch_arr_c_enb_list = [], []
        switch_arr_r_en_list, switch_arr_r_enb_list = [], []
        for port_name in inst_switch_arr.port_names_iter():
            if fnmatch.fnmatch(port_name, 'c_en<*'):
                switch_arr_c_en_list.append(inst_switch_arr.get_pin(port_name))
            elif fnmatch.fnmatch(port_name, 'c_enb<*'):
                switch_arr_c_enb_list.append(inst_switch_arr.get_pin(port_name))
            elif fnmatch.fnmatch(port_name, 'r_en<*'):
                switch_arr_r_en_list.append(inst_switch_arr.get_pin(port_name))
            elif fnmatch.fnmatch(port_name, 'r_enb<*'):
                switch_arr_r_enb_list.append(inst_switch_arr.get_pin(port_name))
            elif fnmatch.fnmatch(port_name, 'in<*') or port_name == 'out':
                self.reexport(inst_switch_arr.get_port(port_name), show=show_pins)

        inv_buffer_sel_dict = dict()
        for port_name in inst_inv_buffer.port_names_iter():
            if fnmatch.fnmatch(port_name, 'sel*'):
                inv_buffer_sel_dict[port_name] = inst_inv_buffer.get_pin(port_name)

        self.connect_wires(switch_arr_c_en_list+col_dec_en_list)
        self.connect_wires(switch_arr_c_enb_list+col_dec_enb_list)
        self.connect_wires(switch_arr_r_en_list + row_dec_en_list)
        self.connect_wires(switch_arr_r_enb_list + row_dec_enb_list)

        inv_buffer_inputs = dict()
        for kwrd in col_dec_input_dict.keys():
            input_number = int(re.search('<(.+)>', kwrd).group(1))
            sig_type = kwrd.split('<')[-2]

            if sig_type == 'in':
                net_name = 'sel<{}>'.format(input_number)
                warr = inv_buffer_sel_dict[net_name]
            else:
                net_name = 'selb<{}>'.format(input_number)
                warr = inv_buffer_sel_dict[net_name]

            self.connect_to_track_wires(col_dec_input_dict[kwrd], warr)
            if sig_type == 'in':
                inv_buffer_inputs[net_name] = col_dec_input_dict[kwrd]

        for kwrd in row_dec_input_dict.keys():
            r_input_number = int(re.search('<(.+)>', kwrd).group(1))
            sig_type = kwrd.split('<')[-2]

            if sig_type == 'in':
                net_name = 'sel<{}>'.format(int(r_input_number) + col_decoder_nbits)
                warr = inv_buffer_sel_dict[net_name]
            else:
                net_name = 'selb<{}>'.format(int(r_input_number) + col_decoder_nbits)
                warr = inv_buffer_sel_dict[net_name]

            self.connect_to_track_wires(row_dec_input_dict[kwrd], warr)
            if sig_type == 'in':
                inv_buffer_inputs[net_name] = warr

        # route the inverter buffer's inputs to the bottom of the overall block
        tot_n_inputs = col_decoder_nbits + row_decoder_nbits
        # sort them by their number and put them in a list
        inv_input_items = sorted(inv_buffer_inputs.items(), key=lambda x: x[0])

        # compute the starting index of the inverter buffer inputs
        start_idx = inst_inv_buffer.get_pin('selb<{}>'.format(tot_n_inputs-1)).track_id.base_index
        # for 1 bit for row space is not gonna be enough, so we shift it two column to to the right
        if row_decoder_nbits == 1:
            start_idx += 2

        input_warr_dict = dict()
        _, warr_locs = self.tr_manager.place_wires(vm_layer, ['sig']*tot_n_inputs,
                                                   start_idx=start_idx)
        for i, index in enumerate(warr_locs):
            index -= tot_n_inputs - 1
            net_name = 'in<{}>'.format(i)
            input_warr_dict[net_name] = self.add_wires(vm_layer, index, lower=0, upper=100)
            self.add_pin(net_name, input_warr_dict[net_name], show=show_pins)

        # connect the wires to the edge
        track_ids_used = []
        # do connections for 0 to col_dec_nbits
        input_col_items = sorted(inv_input_items[:col_decoder_nbits], key=lambda x: x[0],
                                 reverse=True)
        for kwrd, warr in input_col_items:
            input_number = int(re.search('<(.+)>', kwrd).group(1))

            # connect them to corresponding edge routes directly
            if input_number == 0:
                hm_tid = self._get_next_tid(track_ids_used[0], 'sig', 'sig', up_idx=-2)
                warr_m2 = self.connect_to_tracks(input_warr_dict['in<0>'], hm_tid,
                                                 min_len_mode=MinLenMode.LOWER)

                vm_tid = inst_inv_buffer.get_pin('sel<{}>'.format(col_decoder_nbits)).track_id
                # vm_tid = self._get_next_tid(vm_tid, 'sig', 'sig', up_idx=-2)
                vm_corr = self.grid.track_to_coord(vm_layer, vm_tid.base_index)

                conn_tid = TrackID(self.conn_layer,
                                   self.grid.coord_to_track(self.conn_layer, vm_corr))
                warr_m1 = self.connect_to_tracks(warr_m2, conn_tid)
                self.connect_to_track_wires(warr, warr_m1)
            else:
                self.connect_to_track_wires(warr, input_warr_dict['in<{}>'.format(input_number)])
                track_ids_used.append(warr.track_id)

        # do the connections for col_dec_nbits  to col_dec_nbits+ row_dec_nbits
        input_col_items = sorted(inv_input_items[-row_decoder_nbits:], key=lambda x: x[0])
        ref_tid = track_ids_used[-1]
        for kwrd, warr in input_col_items:
            input_number = int(re.search('<(.+)>', kwrd).group(1))
            # do a zig zag
            ref_tid = self._get_next_tid(ref_tid, 'sig', 'sig', up_idx=2)
            warr2 = input_warr_dict['in<{}>'.format(input_number)]
            self.connect_to_tracks([warr, warr2], ref_tid)

        # VDD, VSS
        vdd_list, vss_list = [], []
        for inst in inst_list:
            vdd_list += inst.get_all_port_pins('VDD')
            vss_list += inst.get_all_port_pins('VSS')
        # get rid of all redundant wires
        vdd_list = self.connect_wires(vdd_list)
        vss_list = self.connect_wires(vss_list)

        self.add_pin('VDD', vdd_list, label='VDD:', show=show_pins)
        self.add_pin('VSS', vss_list, label='VSS:', show=show_pins)

        self.set_mos_size()

        self.sch_params = dict(
            n_rows=n_rows,
            n_cols=n_cols,
            inv_buffer_params=inv_buffer_master.sch_params,
            col_decoder_params=col_dec_master.sch_params,
            row_decoder_params=row_dec_master.sch_params,
            passgate_arr_params=switch_array_master.sch_params,
        )

    def _get_num_tracks(self, t1: TrackID, t2: TrackID, sig_type: Union[str, int], *,
                        skip: int = 0) -> int:

        assert t1.layer_id == t2.layer_id, "Two track ids have to have the same layer"
        sep = self.tr_manager.get_sep(t1.layer_id, (sig_type, sig_type))
        d = sep * (skip + 1)
        n = abs((t2.base_htr - t1.base_htr)) // d - 1
        return n

    def _get_next_tid(self, wire_track_object: Union[TrackID, WireArray],
                      cur_type: Union[str, int],
                      next_type: Union[str, int], up_idx: int = 1) -> TrackID:
        """
        computes next track id given the WireArray or TrackID object and given track types
        up_idx: int
            Determines the number of skipped track ids, +1 means the immediate next track id
            -1 means immediate previous track id, +2 means the one after the next track id, etc.
        """

        layer_id = wire_track_object.layer_id
        cur_idx = wire_track_object.base_index if isinstance(wire_track_object, TrackID) else \
            wire_track_object.track_id.base_index

        thtr = cur_idx
        for _ in range(abs(up_idx)):
            thtr = self.tr_manager.get_next_track(layer_id, thtr, cur_type, next_type, up_idx > 0)

        return TrackID(layer_id, thtr, width=self.tr_manager.get_width(layer_id, next_type),
                       pitch=self.tr_manager.get_sep(layer_id, (cur_type, next_type)))
