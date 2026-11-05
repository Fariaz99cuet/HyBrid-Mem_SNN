# synth_report.tcl - synthesis + place & route of snn_accel (out-of-context) with area / timing / power reports
#
#   Vivado Tcl Shell :   cd D:/Downloads/HybMem-SNN
#                        source synth_report.tcl
#   or from cmd      :   vivado -mode batch -source synth_report.tcl
#
# Needs snn_accel.v and weights_init.vh in the same folder as this script.
# Reports are written to the folder  synth_out/
#
# Change the target device with:   set ::env(PART) xc7a35tcpg236-1     (before 'source')
# Change the target clock with:    set ::env(CLK_NS) 8.0
# (7-series device support must be installed in your Vivado; the script lists installed parts if not.)

set here [file dirname [file normalize [info script]]]
cd $here
file mkdir $here/synth_out

# this script runs in non-project mode: close any project left open by run_sim.tcl
catch {close_sim -quiet}
catch {close_project -quiet}

set part   [expr {[info exists ::env(PART)]   ? $::env(PART)   : "xc7z020clg400-1"}]
set clk_ns [expr {[info exists ::env(CLK_NS)] ? $::env(CLK_NS) : 10.0}]

if {[llength [get_parts $part]] == 0} {
    puts "ERROR: part $part is not installed in this Vivado. A few installed parts:"
    puts [lrange [get_parts] 0 15]
    puts "Set one with:  set ::env(PART) <part>   and source this script again."
    return
}

# ---------------------------------------------------------------- synthesis
read_verilog $here/snn_accel.v
synth_design -top snn_accel -part $part -mode out_of_context -include_dirs [list $here]

create_clock -name clk -period $clk_ns [get_ports clk]

report_utilization -file $here/synth_out/synth_utilization.rpt
report_timing_summary -file $here/synth_out/synth_timing.rpt

# ---------------------------------------------------------------- implementation
opt_design
place_design
route_design

report_utilization    -file $here/synth_out/impl_utilization.rpt
report_utilization    -hierarchical -file $here/synth_out/impl_utilization_hier.rpt
report_timing_summary -delay_type min_max -file $here/synth_out/impl_timing.rpt
report_power          -file $here/synth_out/impl_power.rpt
report_clock_utilization -file $here/synth_out/impl_clock.rpt

# ---------------------------------------------------------------- short summary
set wns [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
set fmax [expr {1000.0 / ($clk_ns - $wns)}]
puts ""
puts "=================== SUMMARY ($part, target clock $clk_ns ns) ==================="
set f [open $here/synth_out/impl_utilization.rpt r]
set txt [read $f]
close $f
foreach line [split $txt "\n"] {
    if {[regexp {^\| +(Slice LUTs|CLB LUTs|LUT as Logic|LUT as Memory|Slice Registers|CLB Registers|Block RAM Tile|RAMB36/FIFO|RAMB18|DSPs)\*?\s} $line]} {
        puts $line
    }
}
puts [format "Worst setup slack (WNS): %.3f ns   ->  estimated Fmax = %.1f MHz" $wns $fmax]
if {$wns < 0} { puts "Timing NOT met at $clk_ns ns - set ::env(CLK_NS) to a larger value or use the Fmax above." }
puts "Power: see synth_out/impl_power.rpt  (vectorless estimate - not a measurement)"
puts "All reports are in $here/synth_out"
