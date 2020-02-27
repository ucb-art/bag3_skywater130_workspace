# -*- coding: utf-8 -*-

from pathlib import Path
from pybag.enum import DesignOutput

from bag.core import BagProject
from bag.io import read_yaml
from bag.layout.template import TemplateDB

# from bag3_digital.layout.resdac.decoder import NColDecoder as lay_gen
from bag3_digital.layout.resdac.decoder import NRowDecoder as lay_gen
from xbase.layout.mos.top import GenericWrapper
from bag.design.database import ModuleDB
from bag3_digital.schematic.col_decoder import bag3_digital__col_decoder as sch_gen


def run_main(prj: BagProject, gen_sch: bool = False, run_lvs: bool = False) -> None:
    lib_name = 'AAA_decoder'
    fname = 'unit_tests_resdac/row_decoder.yaml'
    # fname = 'unit_tests_resdac/col_decoder.yaml'

    impl_cell = 'decoder'
    fname_cdl = Path('pvs_run', 'lvs_run_dir', lib_name, impl_cell, 'schematic.net')

    params = read_yaml(Path('specs_test', 'bag3_digital', fname))

    wrap_params = dict(
        cls_name=lay_gen.get_qualified_name(),
        params=params,
    )

    db = TemplateDB(prj.grid, lib_name, prj=prj)

    print('creating new template')
    master = db.new_template(GenericWrapper, wrap_params)
    print('creating batch layout')
    db.batch_layout([(master, impl_cell)], DesignOutput.LAYOUT)
    print('done')

    if gen_sch:

        sch_params = master.sch_params
        sch_db = ModuleDB(prj.tech_info, lib_name, prj=prj)

        sch_master = sch_db.new_master(sch_gen, sch_params)
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
        lvs_passed, lvs_log = prj.run_lvs(lib_name, impl_cell, netlist=fname_cdl)
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

    run_main(bprj, gen_sch=True, run_lvs=True)

