from typing import Any, Dict, Sequence, cast, Type

from pybag.enum import MinLenMode, RoundMode

from bag.util.immutable import Param
from bag.layout.template import TemplateDB
from bag.layout.routing.base import TrackID

from xbase.layout.enum import MOSWireType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase
from xbase.layout.mos.top import MOSBaseWrapper
from bag.util.importlib import import_class
from bag.layout.template import TemplateType


class InvChainCore(MOSBase):
    """An inverter chain.

    Assumes:

    1. PMOS row above NMOS row.
    2. PMOS gate connections on bottom, NMOS gate connections on top.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_list='List of segments per stage.',
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

        seg_list: Sequence[int] = self.params['seg_list']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']

        nstage = len(seg_list)
        if nstage == 0:
            raise ValueError('Must have at least one inverter.')
        if nstage != 1:
            raise ValueError('Right now only supply one inverter.')

        self.draw_base(pinfo)

        seg = seg_list[0]
        nports = self.add_mos(ridx_n, 0, seg, w=w_n)
        pports = self.add_mos(ridx_p, 0, seg, w=w_p)

        self.set_mos_size(seg)

        # ng_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sig')
        nd_tid = self.get_track_id(ridx_n, MOSWireType.DS_GATE, wire_name='sig')
        ns_tid = self.get_track_id(ridx_n, MOSWireType.DS_MATCH, wire_name='sup')
        pd_tid = self.get_track_id(ridx_p, MOSWireType.DS_GATE, wire_name='sig')
        ps_tid = self.get_track_id(ridx_p, MOSWireType.DS_MATCH, wire_name='sup')

        in_vm = self.connect_wires([nports.g, pports.g])
        # in_warr = self.connect_to_tracks(in_vm, ng_tid)
        xr = self.bound_box.xh
        pout = self.connect_to_tracks(pports.d, pd_tid, min_len_mode=MinLenMode.MIDDLE)
        nout = self.connect_to_tracks(nports.d, nd_tid, min_len_mode=MinLenMode.MIDDLE)
        vdd = self.connect_to_tracks(pports.s, ps_tid, track_lower=0, track_upper=xr)
        vss = self.connect_to_tracks(nports.s, ns_tid, track_lower=0, track_upper=xr)

        vm_layer = self.conn_layer + 2
        vm_tidx = self.grid.coord_to_track(vm_layer, pout.middle, mode=RoundMode.GREATER_EQ)
        out = self.connect_to_tracks([pout, nout], TrackID(vm_layer, vm_tidx))

        show_pins = self.show_pins
        self.add_pin('VDD', vdd, show=show_pins)
        self.add_pin('VSS', vss, show=show_pins)
        # self.add_pin('in', in_warr, show=show_pins)
        # self.add_pin('out', out, show=show_pins)
        self.add_pin('out', nout, show=show_pins)
        self.add_pin('out', pout, show=show_pins)
        self.add_pin('out_vm', out, show=False)
        self.add_pin('in_vm', in_vm, show=False)

        self.sch_params = dict(
            seg=seg,
            lch=self.place_info.lch,
            w_p=self.place_info.get_row_place_info(ridx_n).row_info.width if w_p == 0 else w_p,
            w_n=self.place_info.get_row_place_info(ridx_p).row_info.width if w_n == 0 else w_n,
            th_n=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
            th_p=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
        )


class NAND2Core(MOSBase):
    """A 2-input NAND gate.

    Assumes:

    1. PMOS row above NMOS row.
    2. PMOS gate connections on bottom, NMOS gate connections on top.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='Number of segments.',
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

        seg: int = self.params['seg']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']

        self.draw_base(pinfo)

        self.set_mos_size(seg * 2)

        pports = self.add_mos(ridx_p, 0, 2 * seg, w=w_p, g_on_s=True, sep_g=True)
        nports = self.add_mos(ridx_n, 0, seg, w=w_n, g_on_s=True, stack=2, sep_g=True)

        # in0_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=in0_index)
        # in1_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=in1_index)
        nd_tid = self.get_track_id(ridx_n, MOSWireType.DS_GATE, wire_name='sig')
        ns_tid = self.get_track_id(ridx_n, MOSWireType.DS_MATCH, wire_name='sup')
        ps_tid = self.get_track_id(ridx_p, MOSWireType.DS_MATCH, wire_name='sup')

        xr = self.bound_box.xh
        vdd = self.connect_to_tracks(pports.s, ps_tid, track_lower=0, track_upper=xr)
        vss = self.connect_to_tracks(nports.s, ns_tid, track_lower=0, track_upper=xr)
        # in0 = self.connect_to_tracks([pports.g0, nports.g0], in0_tid)
        # in1 = self.connect_to_tracks([pports.g1, nports.g1], in1_tid)
        in0 = self.connect_wires([pports.g0, nports.g0])
        in1 = self.connect_wires([pports.g1, nports.g1])

        out_vm_list = []
        if self.can_short_adj_tracks:
            # we can short adjacent source/drain tracks together, so we can avoid going to vm_layer
            out = self.connect_to_tracks([pports.d, nports.d], nd_tid, ret_wire_list=out_vm_list)
        else:
            # need to use vm_layer to short adjacent tracks.
            pd_tid = self.get_track_id(ridx_p, MOSWireType.DS_GATE, wire_name='sig')
            pout = self.connect_to_tracks(pports.d, pd_tid, min_len_mode=MinLenMode.MIDDLE)
            vm_layer = self.conn_layer + 2
            vm_tidx = self.grid.coord_to_track(vm_layer, pout.middle, mode=RoundMode.GREATER_EQ)
            out_vm = self.connect_to_tracks(pout, TrackID(vm_layer, vm_tidx),
                                            min_len_mode=MinLenMode.LOWER)
            out = self.connect_to_tracks([out_vm, nports.d], nd_tid, ret_wire_list=out_vm_list)

            # add hidden pins so user can query for their location
            # self.add_pin('out_vm', out_vm, show=False)
            self.add_pin('pout', pout, show=False)

        show_pins = self.show_pins
        self.add_pin('VDD', vdd, show=show_pins)
        self.add_pin('VSS', vss, show=show_pins)
        self.add_pin('out', out, show=show_pins)
        self.add_pin('out_vm', out_vm_list, show=False)
        self.add_pin('in0', in0, show=show_pins)
        self.add_pin('in1', in1, show=show_pins)

        self.sch_params = dict(
            seg=seg,
            lch=self.place_info.lch,
            w_p=self.place_info.get_row_place_info(ridx_n).row_info.width if w_p == 0 else w_p,
            w_n=self.place_info.get_row_place_info(ridx_p).row_info.width if w_n == 0 else w_n,
            th_n=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
            th_p=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
        )


class NOR2Core(MOSBase):
    """A 2-input NOR gate.

    Assumes:

    1. PMOS row above NMOS row.
    2. PMOS gate connections on bottom, NMOS gate connections on top.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='Number of segments.',
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

        seg: int = self.params['seg']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']

        self.draw_base(pinfo)

        self.set_mos_size(seg * 2)

        pports = self.add_mos(ridx_p, 0, seg, w=w_p, g_on_s=True, stack=2, sep_g=True)
        nports = self.add_mos(ridx_n, 0, 2 * seg, w=w_n, g_on_s=True, sep_g=True)

        # in0_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sig')
        # in1_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sig')
        pd_tid = self.get_track_id(ridx_p, MOSWireType.DS_GATE, wire_name='sig')
        ns_tid = self.get_track_id(ridx_n, MOSWireType.DS_MATCH, wire_name='sup')
        ps_tid = self.get_track_id(ridx_p, MOSWireType.DS_MATCH, wire_name='sup')

        xr = self.bound_box.xh
        vdd = self.connect_to_tracks(pports.s, ps_tid, track_lower=0, track_upper=xr)
        vss = self.connect_to_tracks(nports.s, ns_tid, track_lower=0, track_upper=xr)

        # in0 = self.connect_to_tracks([pports.g0, nports.g0], in0_tid)
        # in1 = self.connect_to_tracks([pports.g1, nports.g1], in1_tid)
        in0 = self.connect_wires([pports.g0, nports.g0])
        in1 = self.connect_wires([pports.g1, nports.g1])

        out_vm_list = []
        if self.can_short_adj_tracks:
            # we can short adjacent source/drain tracks together, so we can avoid going to vm_layer
            out = self.connect_to_tracks([pports.d, nports.d], pd_tid, ret_wire_list=out_vm_list)
        else:
            # need to use vm_layer to short adjacent tracks.
            nd_tid = self.get_track_id(ridx_n, MOSWireType.DS_GATE, wire_name='sig')
            nout = self.connect_to_tracks(nports.d, nd_tid, min_len_mode=MinLenMode.MIDDLE)
            vm_layer = self.conn_layer + 2
            vm_tidx = self.grid.coord_to_track(vm_layer, nout.middle, mode=RoundMode.GREATER_EQ)
            out_vm = self.connect_to_tracks(nout, TrackID(vm_layer, vm_tidx),
                                            min_len_mode=MinLenMode.UPPER)
            out = self.connect_to_tracks([out_vm, pports.d], pd_tid, ret_wire_list=out_vm_list)

            # add hidden pins so user can query for their location
            # self.add_pin('out_vm', out_vm, show=False)
            self.add_pin('nout', nout, show=False)

        show_pins = self.show_pins
        self.add_pin('VDD', vdd, show=show_pins)
        self.add_pin('VSS', vss, show=show_pins)
        self.add_pin('out', out, show=show_pins)
        self.add_pin('out_vm', out_vm_list, show=False)
        self.add_pin('in0', in0, show=show_pins)
        self.add_pin('in1', in1, show=show_pins)

        self.sch_params = dict(
            seg=seg,
            lch=self.place_info.lch,
            w_p=self.place_info.get_row_place_info(ridx_n).row_info.width if w_p == 0 else w_p,
            w_n=self.place_info.get_row_place_info(ridx_p).row_info.width if w_n == 0 else w_n,
            th_n=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
            th_p=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
        )


class PassGateCore(MOSBase):
    """CMOS pass gate

    Assumes:

    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='Number of segments.',
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

        seg: int = self.params['seg']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']

        self.draw_base(pinfo)

        self.set_mos_size(seg)

        pports = self.add_mos(ridx_p, 0, seg, w=w_p)
        nports = self.add_mos(ridx_n, 0, seg, w=w_n)

        self.connect_wires([nports.s, pports.s])

        # input will connect on the track id aligned with transistor's source
        ng_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sig')
        pg_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sig')
        nd_tid = self.get_track_id(ridx_n, MOSWireType.DS_GATE, wire_name='sig', wire_idx=1)
        pd_tid = self.get_track_id(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=1)

        en_warr = self.connect_to_tracks(nports.g, ng_tid)
        enb_warr = self.connect_to_tracks(pports.g, pg_tid)

        nd_warr = self.connect_to_tracks(nports.d, nd_tid)
        pd_warr = self.connect_to_tracks(pports.d, pd_tid)

        in_tid = TrackID(nd_warr.layer_id+1,
                         self.grid.coord_to_track(nd_warr.layer_id + 1, nd_warr.middle),
                         width=self.tr_manager.get_width(nd_warr.layer_id + 1, 'sig'))

        in_warr = self.connect_to_tracks([nd_warr, pd_warr], track_id=in_tid)

        mid_htr = self.grid.get_middle_track(ng_tid.base_index, pg_tid.base_index)
        mid_tid = TrackID(nports.s.layer_id + 1, mid_htr,
                          width=self.tr_manager.get_width(nports.s.layer_id + 1, 's'))
        out_warr = self.connect_to_tracks([nports.s, pports.s], mid_tid)

        show_pins = self.show_pins
        self.add_pin('d', in_warr, show=show_pins)  # pin on vm layer
        self.add_pin('s', out_warr, show=show_pins)
        self.add_pin('en', en_warr, show=show_pins)
        self.add_pin('enb', enb_warr, show=show_pins)

        self.sch_params = dict(
            seg=seg,
            lch=self.place_info.lch,
            w_p=self.place_info.get_row_place_info(ridx_n).row_info.width if w_p == 0 else w_p,
            w_n=self.place_info.get_row_place_info(ridx_p).row_info.width if w_n == 0 else w_n,
            th_n=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
            th_p=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
        )


class STDCellWithTap(MOSBase):
    """
    Used for running LVS and DRC on individual gates
    """
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            cls_name='wrapped class name',
            params='parameters for the class to be surrounded by taps'
        )

    def draw_layout(self):

        show_pins: bool = self.show_pins
        gen_cls = cast(Type[TemplateType], import_class(self.params['cls_name']))
        master_params = self.params['params'].copy(append=dict(show_pins=False))
        master = self.new_template(gen_cls, params=master_params)
        self.draw_base(master.place_info)

        tap_n_cols = self.get_tap_ncol()
        sub_sep = master.sub_sep_col
        seg_master = master.num_cols
        seg_tot = seg_master + 2 * sub_sep + 2 * tap_n_cols

        vdd_list, vss_list = [], []
        inst = self.add_tile(master, 0, sub_sep + tap_n_cols)

        _, n_tiles = master.tile_size
        for i in range(n_tiles):
            self.add_tap(0, vdd_list, vss_list, tile_idx=i)
            self.add_tap(seg_tot, vdd_list, vss_list, flip_lr=True, tile_idx=i)

        self.set_mos_size(seg_tot)
        # re-export pins
        for name in inst.port_names_iter():
            if name in ['VSS', 'VDD', 'out_vm']:
                continue
            if name == 'in_vm':
                self.reexport(inst.get_port(name), label='in')
                continue
            self.reexport(inst.get_port(name), show=show_pins)

        vdd = self.connect_to_tracks(vdd_list, self.get_track_id(1, MOSWireType.DS_GATE,
                                                                 wire_name='sup'))

        vss = self.connect_to_tracks(vss_list, self.get_track_id(0, MOSWireType.DS_GATE,
                                                                 wire_name='sup'))

        self.add_pin('VSS', vss, show=show_pins)
        self.add_pin('VDD', vdd, show=show_pins)
        self.sch_params = master.sch_params


class STDCellWrapper(MOSBaseWrapper):
    """A MOSArrayWrapper that works with any given generator class."""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBaseWrapper.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            cls_name='wrapped class name.',
            params='parameters for the wrapped class.',
        )

    def get_layout_basename(self) -> str:
        cls_name: str = self.params['cls_name']
        cls_name = cls_name.split('.')[-1]
        if cls_name.endswith('Core'):
            return cls_name[:-4]
        return cls_name + 'Wrap'

    def draw_layout(self):
        gen_cls = cast(Type[TemplateType], import_class(self.params['cls_name']))

        master_params = self.params['params'].copy(append=dict(show_pins=False))
        master = self.new_template(gen_cls, params=master_params)
        inst = self.draw_boundaries(master, master.top_layer)
        self.sch_params = master.sch_params

        show_pins: bool = self.params['show_pins']
        # re-export pins
        for name in inst.port_names_iter():
            self.reexport(inst.get_port(name), show=show_pins)
