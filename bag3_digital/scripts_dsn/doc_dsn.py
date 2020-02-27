# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any
from pybag.enum import DesignOutput

from bag.core import BagProject
from bag.io import read_yaml
from bag.layout.template import TemplateDB
from bag.design.database import ModuleDB
from bag.simulation.data import netlist_info_from_dict, AnalysisType, SimData

from bag3_digital.design.sampler.doc_dsn import get_db, get_vb_crossing_g
from bag3_digital.schematic.samp_doc import bag3_digital__samp_doc
from bag3_digital.schematic.samp2_doc import bag3_digital__samp2_doc
from bag3_digital.schematic.samp3_doc import bag3_digital__samp3_doc
from bag3_digital.schematic.samp4_doc import bag3_digital__samp4_doc
from bag3_digital.schematic.tb_doc import bag3_digital__tb_doc
from bag3_digital.schematic.tb2_doc import bag3_digital__tb2_doc


def gen_scs_netlist(prj: BagProject, specs: Dict[str, Any]) -> Path:
    impl_lib = specs['impl_lib']
    dut_cell = specs['dut_cell']
    # dut_params_file = specs['dut_params']
    sch_params = specs['sch_params']
    tb_cell = specs['tb_cell']
    view_name = specs['view_name']
    if view_name != 'schematic':
        raise NotImplementedError('Post extracted simulation not yet implemented')

    # dut_params = read_yaml(Path('specs_test', 'bag3_digital', dut_params_file))

    # # DUT layout
    # lay_db = TemplateDB(prj.grid, impl_lib, prj=prj)
    # print('creating new template')
    # master = lay_db.new_template(DOCWrapper, dut_params)
    # print('creating batch layout')
    # lay_db.batch_layout([(master, dut_doc_cell)], DesignOutput.LAYOUT)
    # print('layout done')

    # DUT schematic
    sch_db = ModuleDB(prj.tech_info, impl_lib, prj=prj)
    # sch_master = sch_db.new_master(bag3_digital__samp_doc, sch_params)
    # sch_master = sch_db.new_master(bag3_digital__samp2_doc, sch_params)
    # sch_master = sch_db.new_master(bag3_digital__samp3_doc, sch_params)
    sch_master = sch_db.new_master(bag3_digital__samp4_doc, sch_params)

    print('creating DUT schematic')
    sch_db.batch_schematic([(sch_master, dut_cell)])
    print('DUT schematic creation done')

    # DUT scs netlist
    dut_cv_info_list = []
    dut_scs = Path('spectre_run', impl_lib, dut_cell, dut_cell + '.scs')
    print('creating Spectre netlist for DUT')
    sch_db.batch_schematic([(sch_master, dut_cell)], output=DesignOutput.SPECTRE, top_subckt=True,
                           fname=str(dut_scs), cv_info_out=dut_cv_info_list)
    print('Spectre netlist creation for DUT done')

    # TB schematic
    tb_params = dict(
        dut_lib=impl_lib,
        dut_cell=dut_cell,
    )
    # tb_master = sch_db.new_master(bag3_digital__tb_doc, tb_params)
    tb_master = sch_db.new_master(bag3_digital__tb2_doc, tb_params)

    print('creating TB schematic')
    sch_db.batch_schematic([(tb_master, tb_cell)])
    print('TB schematic creation done')

    # TB scs netlist
    tb_scs = Path('spectre_run', impl_lib, tb_cell, tb_cell + '.scs')
    print('creating Spectre netlist for TB')
    sch_db.batch_schematic([(tb_master, tb_cell)], output=DesignOutput.SPECTRE, top_subckt=False,
                           fname=str(tb_scs), cv_info_list=dut_cv_info_list,
                           cv_netlist=str(dut_scs))
    print('Spectre netlist creation for TB done')

    return tb_scs


def run_simulation(prj: BagProject, tb_scs: Path, tb_params: Dict[str, Any]) -> SimData:
    sim_setup = dict(
        sim_envs=tb_params['env_list'],
        analyses=[dict(type='TRAN',
                       start=0.0,
                       stop=tb_params['tsim'],
                       ),
                  ],
        params=dict(tper=tb_params['tper'],
                    tr=tb_params['tr'],
                    tdelay=tb_params['tdelay'],
                    tsim=tb_params['tsim'],
                    rst=tb_params['rst'],
                    vinit=tb_params['vinit'],
                    vfinal=tb_params['vfinal'],
                    vref=tb_params['vref'],
                    vincm=tb_params['vincm'],
                    vdd=tb_params['vdd'],
                    Vb=0.0,
                    vclk=tb_params['vclk'],
                    cload=tb_params['cload'],
                    cpre=tb_params['cpre'],
                    cc=tb_params['cc'],
                    rin=tb_params['rin'],
                    rref=tb_params['rref'],
                    ),
    )

    sim_info = netlist_info_from_dict(sim_setup)
    sim_tag = 'ComparatorTran'
    netlist_path = tb_scs.with_name(tb_scs.stem + '_sim.scs')
    prj.sim_access.create_netlist(netlist_path, tb_scs, sim_info)
    prj.sim_access.run_simulation(netlist_path, sim_tag)
    sim_data = prj.sim_access.load_sim_data(netlist_path.parent, sim_tag)
    sim_data.open_analysis(AnalysisType.TRAN)

    return sim_data


if __name__ == "__main__":
    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()
    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    # fname = 'doc_dsn.yaml'
    fname = 'doc_dsn4.yaml'
    specs = read_yaml(Path('specs_design', 'bag3_digital', fname))

    tb_netlist = gen_scs_netlist(bprj, specs)

    # nch_db = get_db('nch', specs)
    # pch_db = get_db('pch', specs)

    tb_params = specs['tb_params']
    # doc_seg_dict = specs['sch_params']['doc_wrap_params']['doc_params']['seg_dict']
    # tb_params['Vb'], vdd_g = get_vb_crossing_g(nch_db, pch_db, doc_seg_dict['gm2n'],
    #                                            doc_seg_dict['gm2p'], doc_seg_dict['gm1'],
    #                                            doc_seg_dict['pow'], tb_params['vdd'])
    # print(tb_params['Vb'])

    results = run_simulation(bprj, tb_netlist, tb_params)

    time = results['time']
    Vin = results['Vin'][0]
    ViP, ViM = results['ViP'][0], results['ViM'][0]
    VoP, VoM = results['VoP'][0], results['VoM'][0]
    V1p, V1m = results['V1p'][0], results['V1m'][0]
    sw_p, sw_m = results['sw_p'][0], results['sw_m'][0]
    clk, clkd = results['clk'][0], results['clkd'][0]

    mode = 0

    if mode:
        plt.subplot(6, 1, 1)
        plt.plot(time, Vin)
        plt.ylabel('Input')

        plt.subplot(6, 1, 2)
        plt.plot(time, ViP)
        # plt.plot(time, ViM)
        plt.ylabel('Sampled Input')

        plt.subplot(6, 1, 3)
        plt.plot(time, V1p)
        # plt.plot(time, V1m)
        plt.ylabel('After AC cap')

        plt.subplot(6, 1, 4)
        plt.plot(time, clk)
        plt.plot(time, clkd)
        plt.ylabel('Clock')

        plt.subplot(6, 1, 5)
        plt.plot(time, VoP)
        plt.plot(time, VoM)
        plt.ylabel('Output')

        plt.subplot(6, 1, 6)
        plt.plot(time, sw_p)
        plt.plot(time, sw_m)
        plt.ylabel('Output before cross cap')
    else:
        plt.plot(time, clk)
        plt.plot(time, clkd)
        plt.plot(time, ViP)
        # plt.plot(time, ViM)
        plt.plot(time, V1p)
        # plt.plot(time, V1m)
        plt.plot(time, sw_p)
        plt.plot(time, sw_m)
        plt.plot(time, VoP)
        plt.plot(time, VoM)
        plt.legend(['clk', 'clkd', 'sampp', 'inp', 'gp', 'gn', 'outp', 'outn'])
        # plt.legend(['clk', 'clkd', 'sampp', 'sampn', 'inp', 'inn', 'gp', 'gn', 'outp', 'outn'])

    plt.show()
