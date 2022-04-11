# -*- coding: utf-8 -*-

from pathlib import Path
from pybag.enum import DesignOutput

from bag.core import BagProject
from bag.io import read_yaml
from bag.layout.template import TemplateDB
from bag.design.database import ModuleDB

from bag3_digital.layout.sampler.sampler import SingleEndedNSampWrapper
from bag3_digital.schematic.sampler_wrap import bag3_digital__sampler_wrap


def run_main(prj: BagProject, gen_sch: bool = True, gen_cdl: bool = True, run_lvs: bool = True) \
        -> None:
    lib_name = 'AAA_BAG3_Sampler'
    fname = 'sampler.yaml'

    impl_cell = 'sampler_wrap'
    fname_cdl = Path('cdl_netlist', lib_name, impl_cell, 'schematic.net')

    params = read_yaml(Path('specs_test', 'bag3_digital', fname))

    db = TemplateDB(prj.grid, lib_name, prj=prj)

    print('creating new template')
    master = db.new_template(SingleEndedNSampWrapper, params)
    print('creating batch layout')
    db.batch_layout([(master, impl_cell)], DesignOutput.LAYOUT)
    print('done')

    if gen_sch or gen_cdl:
        sch_db = ModuleDB(prj.tech_info, lib_name, prj=prj)

        sch_master = sch_db.new_master(bag3_digital__sampler_wrap, master.sch_params)
        cv_info_list = []

        print('creating schematic')
        sch_db.batch_schematic([(sch_master, impl_cell)], cv_info_out=cv_info_list)
        print('schematic creation done')

        print('creating CDL netlist')
        sch_db.batch_schematic([(sch_master, impl_cell)], output=DesignOutput.CDL,
                               fname=str(fname_cdl), cv_info_list=cv_info_list)
        print('netlist creation done')

    if run_lvs:
        print('Running LVS ...')
        lvs_passed, lvs_log = prj.run_lvs(lib_name, impl_cell, netlist=str(fname_cdl))
        print('LVS log file:' + lvs_log)
        if lvs_passed:
            print('LVS passed!')
        else:
            print('LVS failed :(')


if __name__ == '__main__':
    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()
    else:
        print('loading BAG project')
        bprj = local_dict['bprj']

    run_main(bprj, gen_sch=False, gen_cdl=False, run_lvs=False)
    # run_main(bprj, gen_sch=True, gen_cdl=True, run_lvs=True)
