from typing import Any, Dict, Union

from bag.layout.routing.base import WireArray

from pybag.enum import MinLenMode

from bag.util.immutable import Param
from bag.layout.template import TemplateDB
from bag.layout.routing.base import TrackID

from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase
from bag3_digital.layout.resdac.gates import PassGateCore
from xbase.layout.enum import MOSWireType


class NTo1PassGateMux(MOSBase):
    """an N to 1 pass gate multiplexer

    Assumes:
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='seg of each pass gate switch',
            n='number of n',
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

        n: int = self.params['n']
        show_pins: bool = self.show_pins

        passgate_params = self.params.copy(append={'show_pins': False})
        passgate_master = self.new_template(PassGateCore, params=passgate_params)

        passgate_ncols = passgate_master.num_cols
        sep_cols = self.min_sep_col

        inst_list = []
        for i in range(n):
            inst_list.append(self.add_tile(passgate_master, 0, i*(passgate_ncols + sep_cols)))

        for i, inst in enumerate(inst_list):
            en_tid = self._get_next_tid(inst.get_pin("in"), "in", "en", up=False)
            enb_tid = self._get_next_tid(inst.get_pin("in"), "in", "enb", up=True)
            en_m3 = self.connect_to_tracks(inst.get_pin("en"), en_tid,
                                           min_len_mode=MinLenMode.LOWER)
            enb_m3 = self.connect_to_tracks(inst.get_pin("enb"), enb_tid,
                                            min_len_mode=MinLenMode.LOWER)
            self.add_pin("en_{}".format(i), en_m3, show=show_pins)
            self.add_pin("enb_{}".format(i), enb_m3, show=show_pins)
            self.reexport(inst.get_port("in"), net_name="in_{}".format(i), show=show_pins)

        out_m2 = self.connect_wires([inst.get_pin('out') for inst in inst_list])
        self.add_pin("out", out_m2, show=show_pins)

        self.set_mos_size()

    def _get_next_tid(self, wire_track_object: Union[TrackID, WireArray], cur_type: Union[str,
                      int], next_type: Union[str, int], up: bool = True) -> TrackID:
        """
        computes next track id given the WireArray or TrackID object and given track types
        """

        layer_id = wire_track_object.layer_id
        cur_idx = wire_track_object.base_index if isinstance(wire_track_object, TrackID) else \
            wire_track_object.track_id.base_index

        thtr = self.tr_manager.get_next_track(layer_id, cur_idx, cur_type, next_type, up)

        return TrackID(layer_id, thtr, width=self.tr_manager.get_width(layer_id, next_type),
                       pitch=self.tr_manager.get_sep(layer_id, (cur_type, next_type)))


class SwitchArray(MOSBase):
    """
    Passgate multiplexer array used for res dac

    Assumes:
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='seg of each pass gate switch',
            n_cols='number of columns in res dac unit',
            n_rows='number of rows in res dac unit',
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

        n_cols: int = self.params['n_cols']
        n_rows: int = self.params['n_rows']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        tap_rate: int = self.params['tap_rate']
        show_pins: bool = self.show_pins

        passgate_params = self.params.copy(append={'show_pins': False})
        passgate_master = self.new_template(PassGateCore, params=passgate_params)

        passgate_ncols = passgate_master.num_cols

        # col mux array
        col_inst_list = []
        col_warr_dict = {}
        out_m2_list = []

        vdd_list, vss_list = [], []

        sub_sep = self.sub_sep_col
        col_delta = passgate_ncols + self.min_sep_col
        last_col = 0
        for r in range(n_rows):
            col_inst_list.append([])
            cur_col = 0
            for c in range(n_cols):
                if c % tap_rate == 0:
                    if cur_col != 0:
                        cur_col += sub_sep - self.min_sep_col
                    tap_col = self.add_tap(cur_col, vdd_list, vss_list, tile_idx=r)
                    cur_col += tap_col + sub_sep

                inst = self.add_tile(passgate_master, r, cur_col)
                cur_col += col_delta
                col_inst_list[r].append(inst)

                en_tid = self._get_next_tid(inst.get_pin("d"), "d", "en",
                                            up=False)
                enb_tid = self._get_next_tid(inst.get_pin("d"), "d",
                                             "enb", up=True)
                en_m3 = self.connect_to_tracks(inst.get_pin("en"), en_tid)

                enb_m3 = self.connect_to_tracks(inst.get_pin("enb"), enb_tid)
                en_key = "c_en<{}>".format(c)
                enb_key = "c_enb<{}>".format(c)
                in_key = "in<{}>".format(c + r*n_cols)
                self.add_pin(en_key, en_m3, show=show_pins)
                self.add_pin(enb_key, enb_m3, show=show_pins)
                self.reexport(inst.get_port("d"), net_name=in_key, show=show_pins)
                self.update_warr_dict(col_warr_dict,
                                      **{en_key: en_m3, enb_key: enb_m3,
                                         in_key: inst.get_pin("d")})

            if cur_col > last_col:
                last_col = cur_col
            # connect horizental output in col muxes in each row together
            out_m2 = self.connect_wires([inst.get_pin('s') for inst in col_inst_list[r]])[0]
            out_m2_list.append(out_m2)

        # connect vertical columns in col muxes and put down the pins
        for warr_key, warr_list in col_warr_dict.items():
            warr = self.connect_wires(warr_list)
            self.add_pin(warr_key, warr, show=show_pins)

        # row mux array
        row_inst_list = []
        row_warr_dict = {}
        output_switch_arr = None
        tap_col = 0
        for r in range(n_rows):
            inst = self.add_tile(passgate_master, r, last_col)
            row_inst_list.append(inst)

            self.reexport(inst.get_port('en'), net_name='r_en<{}>'.format(r))
            self.reexport(inst.get_port('enb'), net_name='r_enb<{}>'.format(r))

            self.connect_wires([inst.get_pin("s"), out_m2_list[r]])
            if not output_switch_arr:
                output_switch_arr = inst.get_pin("d", layer=self.conn_layer+2)
            else:
                output_switch_arr = self.connect_wires([output_switch_arr,
                                                        inst.get_pin("d")])
                assert len(output_switch_arr) == 1, "Multiple output wire arrays have been found"
                output_switch_arr = output_switch_arr[0]

            tap_col = self.add_tap(last_col + passgate_ncols + sub_sep, vdd_list, vss_list,
                                   flip_lr=True, tile_idx=r)

        last_col += passgate_ncols + sub_sep + tap_col

        vdd_list = self.connect_wires(vdd_list)
        vss_list = self.connect_wires(vss_list)

        # connect vertical columns in col muxes
        for warr_list in row_warr_dict.values():
            self.connect_wires(warr_list)

        upper = self.grid.track_to_coord(self.conn_layer, last_col)
        vdd_m2_list, vss_m2_list = [], []
        for r in range(n_rows):
            vdd_tid = self.get_track_id(ridx_p, MOSWireType.DS_MATCH, tile_idx=r, wire_name='sup')
            vss_tid = self.get_track_id(ridx_n, MOSWireType.DS_MATCH, tile_idx=r, wire_name='sup')
            vdd_m2_list.append(self.add_wires(self.conn_layer+1, vdd_tid.base_index, lower=0,
                                              upper=upper, width=vdd_tid.width, num=vdd_tid.num))
            vss_m2_list.append(self.add_wires(self.conn_layer+1, vss_tid.base_index, lower=0,
                                              upper=upper, width=vdd_tid.width, num=vdd_tid.num))

        self.connect_to_track_wires(vdd_list, vdd_m2_list)
        self.connect_to_track_wires(vss_list, vss_m2_list)

        self.add_pin('VDD', vdd_m2_list, show=show_pins)
        self.add_pin('VSS', vss_m2_list, show=show_pins)
        self.add_pin('out', output_switch_arr, show=show_pins)

        self.set_mos_size()

        self.sch_params = dict(
            n_cols=n_cols,
            n_rows=n_rows,
            passgate_params=dict(
                seg=self.params['seg'],
                lch=self.place_info.lch,
                w_p=self.place_info.get_row_place_info(ridx_n).row_info.width if w_p == 0 else w_p,
                w_n=self.place_info.get_row_place_info(ridx_p).row_info.width if w_n == 0 else w_n,
                th_n=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
                th_p=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
            )
        )

    @classmethod
    def update_warr_dict(cls, warr_dict, **kwargs):
        for key, val in kwargs.items():
            if key in warr_dict:
                warr_dict[key].append(val)
            else:
                warr_dict[key] = [val]

    def _get_next_tid(self, wire_track_object: Union[TrackID, WireArray], cur_type: Union[str, int],
                      next_type: Union[str, int], up: bool = True) -> TrackID:
        """
        computes next track id given the WireArray or TrackID object and given track types
        """

        layer_id = wire_track_object.layer_id
        cur_idx = wire_track_object.base_index if isinstance(wire_track_object, TrackID) else \
            wire_track_object.track_id.base_index

        thtr = self.tr_manager.get_next_track(layer_id, cur_idx, cur_type, next_type, up)

        return TrackID(layer_id, thtr, width=self.tr_manager.get_width(layer_id, next_type),
                       pitch=self.tr_manager.get_sep(layer_id, (cur_type, next_type)))
