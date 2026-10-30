set repo [file normalize [file join [file dirname [info script]] ..]]
set clock_period_ns 2.500
if {[llength $argv] > 0} { set clock_period_ns [lindex $argv 0] }
set run_tag [clock format [clock seconds] -format %Y%m%d_%H%M%S]
if {[llength $argv] > 1} { set run_tag [lindex $argv 1] }
set period_tag [string map {. p} [format %.3f $clock_period_ns]]
set out_dir [file join $repo build supra2_block_ddr_system_ooc \
    "vivado_ooc_${period_tag}ns_${run_tag}"]

set_param general.maxThreads 1
open_checkpoint [file join $out_dir synthesized.dcp]
read_xdc [file join $out_dir clock.xdc]

report_methodology -file [file join $out_dir synth_methodology.rpt]
report_drc -checks [get_drc_checks LUTLP-1] \
    -file [file join $out_dir synth_combinational_loops.rpt]
check_timing -verbose -file [file join $out_dir synth_check_timing.rpt]
report_timing_summary -delay_type max -max_paths 30 \
    -file [file join $out_dir synth_timing_summary.rpt]

opt_design
place_design -directive Explore
phys_opt_design -directive Explore
route_design -directive Explore
write_checkpoint -force [file join $out_dir routed.dcp]
report_utilization -file [file join $out_dir routed_utilization.rpt]
report_timing_summary -delay_type max -max_paths 30 \
    -file [file join $out_dir routed_timing_summary.rpt]
report_drc -file [file join $out_dir routed_drc.rpt]
exit
