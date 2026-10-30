set repo [file normalize [file join [file dirname [info script]] ..]]
set clock_period_ns 2.500
if {[llength $argv] > 0} {
    set clock_period_ns [lindex $argv 0]
}
set run_tag [clock format [clock seconds] -format %Y%m%d_%H%M%S]
if {[llength $argv] > 1} {
    set run_tag [lindex $argv 1]
}
set period_tag [string map {. p} [format %.3f $clock_period_ns]]
set out_dir [file join $repo build supra2_hadamard_ooc "vivado_ooc_${period_tag}ns_${run_tag}"]
file mkdir $out_dir

read_verilog -sv [file join $repo rtl pkg fp16_utils_pkg.sv]
read_verilog -sv [file join $repo rtl supra2 fp32_add_rne_pipeline.sv]
read_verilog -sv [file join $repo rtl supra2 hadamard_transform.sv]
set clock_file [file join $out_dir clock.xdc]
set clock_stream [open $clock_file w]
puts $clock_stream [format {create_clock -name hadamard_clk -period %s [get_ports clk]} $clock_period_ns]
close $clock_stream
read_xdc $clock_file
set_param general.maxThreads 1
synth_design -top supra2_hadamard_transform -part xczu7ev-ffvc1156-2-e \
    -mode out_of_context -flatten_hierarchy none -directive RuntimeOptimized
write_checkpoint -force [file join $out_dir synthesized.dcp]
report_utilization -file [file join $out_dir synth_utilization.rpt]
write_xdc -force -exclude_physical [file join $out_dir constraints.xdc]
report_methodology -file [file join $out_dir synth_methodology.rpt]
report_drc -checks [get_drc_checks LUTLP-1] -file [file join $out_dir synth_combinational_loops.rpt]
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
