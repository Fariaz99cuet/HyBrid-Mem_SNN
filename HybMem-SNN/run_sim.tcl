# run_sim.tcl  -  run the SNN accelerator simulation from the Vivado terminal
#
#   vivado -mode batch -source run_sim.tcl          (from a normal terminal)
#   source run_sim.tcl                              (inside the Vivado Tcl Shell)
#
# Put this file in the same folder as snn_accel.v, tb_snn_accel.v,
# weights_init.vh and test_data.vh.

set here [file dirname [file normalize [info script]]]
cd $here

# close a project left open by a previous run, otherwise create_project fails
catch {close_sim -quiet}
catch {close_project -quiet}

create_project -force sim_proj $here/sim_proj

add_files -norecurse [list $here/snn_accel.v $here/weights_init.vh]
add_files -fileset sim_1 -norecurse [list $here/tb_snn_accel.v $here/test_data.vh]

# let `include "weights_init.vh" / "test_data.vh" be found
set_property include_dirs [list $here] [get_filesets sources_1]
set_property include_dirs [list $here] [get_filesets sim_1]

set_property top snn_accel       [get_filesets sources_1]
set_property top tb_snn_accel    [get_filesets sim_1]
set_property top_lib xil_defaultlib [get_filesets sim_1]

# run until the testbench calls $finish
set_property -name {xsim.simulate.runtime} -value {-all} -objects [get_filesets sim_1]

launch_simulation

catch {close_sim}
puts "simulation finished - see the table above (RESULT: PASS / FAIL)"
