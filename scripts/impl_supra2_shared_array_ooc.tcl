set repo [file normalize [file join [file dirname [info script]] ..]]
set period_ns 2.500
if {[llength $argv] > 0} { set period_ns [lindex $argv 0] }
set run_tag [clock format [clock seconds] -format %Y%m%d_%H%M%S]
if {[llength $argv] > 1} { set run_tag [lindex $argv 1] }
set out_dir [file join $repo build supra2_block_ddr_system_ooc "shared_array_${run_tag}"]
set_param general.maxThreads 1
open_checkpoint [file join $out_dir synthesized.dcp]
create_clock -name sys_clk -period $period_ns [get_ports clk]
opt_design
place_design -directive Explore
phys_opt_design -directive Explore
route_design -directive Explore
write_checkpoint -force [file join $out_dir routed.dcp]
report_utilization -file [file join $out_dir routed_utilization.rpt]
report_timing_summary -delay_type min_max -max_paths 30 \
    -file [file join $out_dir routed_timing_summary.rpt]
report_drc -file [file join $out_dir routed_drc.rpt]
exit
