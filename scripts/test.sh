set -e 

# BAG Workspace Setup Tests 
# Generate cells from 'bag3_digital'

./run_bag.sh BAG_framework/run_scripts/gen_cell.py data/bag3_digital/specs_gen/stdcells/inv.yaml -v -raw
./run_bag.sh BAG_framework/run_scripts/gen_cell.py data/bag3_digital/specs_gen/stdcells/inv_tristate.yaml -v -raw
./run_bag.sh BAG_framework/run_scripts/gen_cell.py data/bag3_digital/specs_gen/stdcells/nand2.yaml -v -raw
./run_bag.sh BAG_framework/run_scripts/gen_cell.py data/bag3_digital/specs_gen/stdcells/nor2.yaml -v -raw
./run_bag.sh BAG_framework/run_scripts/gen_cell.py data/bag3_digital/specs_gen/stdcells/flop_scan_rstlb.yaml -v -raw
