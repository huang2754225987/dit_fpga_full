set repo [file normalize [file join [file dirname [info script]] ..]]
if {[llength $argv] < 3} {
    error "Usage: implement_supra2_pv_ooc.tcl <period_ns> <run_tag> <synth_dcp>"
}
set clock_period_ns [lindex $argv 0]
set run_tag [lindex $argv 1]
set synth_dcp [file normalize [lindex $argv 2]]
set period_tag [string map {. p} [format %.3f $clock_period_ns]]
set out_dir [file join $repo build supra2_pv_ooc "vivado_impl_${period_tag}ns_${run_tag}"]
file mkdir $out_dir
if {![file exists $synth_dcp]} { error "Synthesis checkpoint not found: $synth_dcp" }

open_checkpoint $synth_dcp
set_param general.maxThreads 1
report_clocks -file [file join $out_dir clocks.rpt]
opt_design -directive Explore
place_design -directive Explore
phys_opt_design -directive Explore
route_design -directive Explore
write_checkpoint -force [file join $out_dir routed.dcp]
report_utilization -file [file join $out_dir routed_utilization.rpt]
report_timing_summary -delay_type max -max_paths 30 \
    -file [file join $out_dir routed_timing_summary.rpt]
report_drc -file [file join $out_dir routed_drc.rpt]
exit
