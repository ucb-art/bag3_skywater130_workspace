
from bag.core import BagProject
from bag3_digital.design.digital_db.db import Characterizer

if __name__ == '__main__':
    config_file = 'data/bag3_digital/specs_db/db_config.yaml'

    local_dict = locals()
    prj = local_dict.get('prj', BagProject())
    db_char = Characterizer(prj, config_file, dut_type='pch')
    params = dict(
        intent='lvt',
        lch=36,
        mos_type='pch',
        seg=32,
        w=4,
        vdd=1.0,)
    results = db_char.get_char_point(params, hash_value=1, sim_envs=['tt_25', 'ss_25'])
    print(results)
