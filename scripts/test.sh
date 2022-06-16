set -e 

# BAG Workspace Setup Tests 
# Generate cells from 'bag3_digital'

./gen_cell.sh data/bag3_digital/specs_gen/stdcells/inv.yaml -v
./gen_cell.sh data/bag3_digital/specs_gen/stdcells/inv_tristate.yaml -v
./gen_cell.sh data/bag3_digital/specs_gen/stdcells/nand2.yaml -v
./gen_cell.sh data/bag3_digital/specs_gen/stdcells/nor2.yaml -v
