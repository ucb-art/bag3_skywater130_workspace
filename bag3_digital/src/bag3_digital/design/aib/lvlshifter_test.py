
from typing import Dict, Any, Tuple, Sequence, Type, cast, List, Optional

from bag.core import BagProject
from bag.io.file import read_yaml
from pathlib import Path

from bag.simulation.core import DesignManager

from bag3_digital.design.aib.output_driver import OutputDriverTBManager, OutputDriverMeasManager
import pdb


def test(prj: BagProject):

    pdir = Path(__file__).parent
    yaml_f = Path(str(pdir), 'lvl_shifter.yaml')
    yaml_content: Dict[str, Any] = read_yaml(yaml_f)

    dsn_man = DesignManager(prj, spec_dict=yaml_content)
    dsn_man.characterize_designs(generate=True, measure=True, load_from_file=False)
    pdb.set_trace()


if __name__ == '__main__':

    local_dict = locals()
    if 'prj' not in local_dict:
        print('creating bag project')
        prj = BagProject()
    else:
        print('loading bag project')
        prj = local_dict['prj']

    test(prj)