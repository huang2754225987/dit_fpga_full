set repo [file normalize [file join [file dirname [info script]] ..]]
set period_ns [lindex $argv 0]
set run_tag [lindex $argv 1]
set out_dir [file join $repo build supra2_block_ddr_system_ooc "components_${run_tag}"]
set_param general.maxThreads 1
open_checkpoint [file join $out_dir system_synth.dcp]
read_xdc [file join $out_dir clock.xdc]
report_methodology -file [file join $out_dir system_synth_methodology.rpt]
report_drc -checks [get_drc_checks LUTLP-1] -file [file join $out_dir system_synth_loops.rpt]
check_timing -verbose -file [file join $out_dir system_synth_check_timing.rpt]
report_timing_summary -delay_type max -max_paths 30 \
    -file [file join $out_dir system_synth_timing.rpt]
opt_design
place_design -directive Explore
phys_opt_design -directive Explore
route_design -directive Explore
write_checkpoint -force [file join $out_dir system_routed.dcp]
report_utilization -file [file join $out_dir system_routed_utilization.rpt]
report_timing_summary -delay_type min_max -max_paths 50 \
    -file [file join $out_dir system_routed_timing.rpt]
report_drc -file [file join $out_dir system_routed_drc.rpt]
exit
