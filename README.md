# BAG3 SkyWater130 Template Workspace 

For use on BWRC infrastructure 


## Requirements 

This template workspace uses the pre-compiled version of BAG3 installed on the BWRC Linux servers. 

BAG3 will successfully run on a subset of the BWRC servers, which particularly include: 

* RedHat Linux v7. Running `uname -a` should yield a string that includes `el7`. 
* RedHat DevTools v8, and several related packages. On relevant machines, enable these byt running `source scl_source enable devtoolset-8 rh-git29 httpd24`
* The one known-good server as of commit-time is `bwrcr740-8`. It comes highly recommended. 


## Initial Setup 

After cloning, do the following:

1. ssh to a server that supports RHEL7, or use bsub on the rhel7i queue.
2. add this to your home .bashrc so that devtoolset is enabled everytime a new terminal is opened: `source scl_source enable devtoolset-8 rh-git29 httpd24`
3. Start bash
4. `git submodule update --init --recursive`
5. Create a new cds.lib as follows: `echo "INCLUDE cds.lib.core" >> cds.lib`
6. `source .bashrc`
7. `cd BAG_framework/pybag`
8. `./run_test.sh`
9. `cd ../..`
10. Run just lines 55 and 56 of [setup_script.sh](setup_script.sh) to create symlinks to storage places in /tools/scratch.

## Working 

To start working, run:

```
source scripts/work.sh
```

This [much-shorter script](scripts/work.sh) loads required packages and configures the run-environment. 
(It may be just as easy to run "manually".) 
Note it is intended to be `source`-ed, and to modify its parent environment. 


## Testing 

To (at any point) check for valid installation and configuration, run the [scripts/test.sh](scripts/test.sh) script. 
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

If you would like to use abstract generation, there are a few issues to be aware of. More details available of the issues surrounding abstract generation and additional setup instructions in [the tech plugin abstract setup README.](skywater130/abstract_setup/README.md)

## Licensing

This library is licensed under the Apache-2.0 license.  See [here](LICENSE) for full text of the
Apache license.


