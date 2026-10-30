set repo [file normalize [file join [file dirname [info script]] ..]]
if {[llength $argv] != 3} {
    error "usage: implement_supra2_dit_compute_bridge_ooc.tcl period_ns synth_run_tag impl_run_tag"
}
set clock_period_ns [lindex $argv 0]
set synth_run_tag [lindex $argv 1]
set run_tag [lindex $argv 2]
set period_tag [string map {. p} [format %.3f $clock_period_ns]]
set synth_dir [file join $repo build supra2_dit_compute_bridge \
    "vivado_ooc_${period_tag}ns_${synth_run_tag}"]
set out_dir [file join $repo build supra2_dit_compute_bridge \
    "vivado_ooc_${period_tag}ns_${run_tag}"]
set synth_checkpoint [file join $synth_dir synthesized.dcp]
if {![file exists $synth_checkpoint]} {
    error "synthesized checkpoint not found: $synth_checkpoint"
}
file mkdir $out_dir
read_checkpoint $synth_checkpoint
set bridge_clock [get_clocks -quiet bridge_clk]
if {[llength $bridge_clock] != 1} {
    error "expected exactly one bridge_clk in synthesized checkpoint"
}
set_property PERIOD $clock_period_ns $bridge_clock
write_xdc -force [file join $out_dir constraints.xdc]
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
