
# BAG3 SkyWater130 Template Workspace 

For use on BWRC infrastructure 


## Requirements 

This template workspace uses the pre-compiled version of BAG3 installed on the BWRC Linux servers. 

BAG3 will successfully run on a subset of the BWRC servers, which particularly include: 

* RedHat Linux v7. Running `uname -a` should yield a string that includes `el7`. 
* RedHat DevTools v8, and several related packages. On relevant machines, enable these byt running `source scl_source enable devtoolset-8 rh-git29 httpd24`
* The one known-good server as of commit-time is `bwrcr740-8`. It comes highly recommended. 


## Initial Setup 

After cloning, run: 

```
bash scripts/setup.sh
```

[Scripts/setup.sh](scripts/setup.sh) will perform a number of one-time setup activities, including cloning submodules, linking several binary libraries, 
and initializing Virtuoso libraries. Note invoking this script with `bash` (and not `source`) *will not* modify the existing environment. 


## Working 

To start working, run:

```
source scripts/work.sh
```

This [much-shorter script](scripts/work.sh) loads required packages and configures the run-environment. 
(It may be just as easy to run "manually".) 
Note it is intended to be `source`-ed, and to modify its parent environment. 


## Testing 

To (at any point) check for valid installation and configuration, run the `scripts/test.sh` script. 
This will generate a set of simple schematics and layouts from `bag3_digital`. 
If successful,  you'll be met with output like so: 

```
# ... 
creating BAG project
*WARNING* [Errno 2] No such file or directory: '/tools/B/dan_fritchman/sky130/bag3_skywater130_workspace/BAG_server_port.txt'.  Operating without Virtuoso.
computing schematic...
computation done.
creating netlist...
netlisting done.
```

Generated netlists and GDS layouts can be found in the `gen_outputs` directory. 
The content of [scripts/test.sh](scripts/test.sh) includes the run-commands, which can also be run on their own. 


## Caveats 

SkyWater130, like most planar technologies, *does not* produce LVS-clean BAG schematics. 
*Templates* are still designed in Virtuoso schematics, but their *generated output* is relevant only in netlist form. 
Note each of the runs in [scripts/test.sh](scripts/test.sh) uses BAG's `-raw` option to directly produce GDS. 
Attempts to use generated Virtuoso schematics will typically produce (intractable) LVS errors. 


## Licensing

This library is licensed under the Apache-2.0 license.  See [here](LICENSE) for full text of the
Apache license.


