set repo [file normalize [file join [file dirname [info script]] ..]]
set run_tag [lindex $argv 0]
set out_dir [file join $repo build supra2_fp32_pipeline_probe $run_tag]
file mkdir $out_dir
read_verilog -sv [file join $repo rtl pkg fp16_utils_pkg.sv]
read_verilog -sv [file join $repo rtl supra2 fp32_add_rne_pipeline.sv]
read_verilog -sv [file join $repo rtl sim fp32_add_synth_probe.sv]
set clock_file [file join $out_dir clock.xdc]
set clock_stream [open $clock_file w]
puts $clock_stream {create_clock -name pipeline_probe_clk -period 2.500 [get_ports clk]}
close $clock_stream
read_xdc $clock_file
set_param general.maxThreads 1
synth_design -top fp32_add_pipeline_synth_probe -part xczu7ev-ffvc1156-2-e \
    -mode out_of_context -flatten_hierarchy none -directive RuntimeOptimized
write_checkpoint -force [file join $out_dir synthesized.dcp]
report_utilization -file [file join $out_dir synth_utilization.rpt]
report_drc -checks [get_drc_checks LUTLP-1] -file [file join $out_dir combinational_loops.rpt]
check_timing -verbose -file [file join $out_dir check_timing.rpt]
report_timing_summary -delay_type max -max_paths 10 \
    -file [file join $out_dir timing_summary.rpt]
exit
