import argparse
from bag.core import BagProject
from bag3_testbenches.measurement.mos.db import DigitalDB


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('specs', help='YAML specs file name.')
    args = parser.parse_args()

    local_dict = locals()
    prj = local_dict.get('prj', BagProject())
    db = DigitalDB(prj, args.specs)

    if not db.has_results:
        db.characterize()

    db.get_cg()
    db.get_cd()
    db.get_res()