set repo [file normalize [file join [file dirname [info script]] ..]]
set clock_period_ns 2.500
if {[llength $argv] > 0} { set clock_period_ns [lindex $argv 0] }
set run_tag [clock format [clock seconds] -format %Y%m%d_%H%M%S]
if {[llength $argv] > 1} { set run_tag [lindex $argv 1] }
set period_tag [string map {. p} [format %.3f $clock_period_ns]]
set out_dir [file join $repo build supra2_pv_ooc "vivado_ooc_${period_tag}ns_${run_tag}"]
file mkdir $out_dir

foreach source [list \
    rtl/pkg/isa_pkg.sv \
    rtl/pkg/fp16_utils_pkg.sv \
    rtl/ops/graph_exp_lut_fp164k.sv \
    rtl/ops/recip_lut4k.sv \
    rtl/ops/softmax_engine.sv \
    rtl/gemm/mac_int8.sv \
    rtl/gemm/pe.sv \
    rtl/gemm/systolic_array.sv \
    rtl/supra2/supra2_sync_word_ram.sv \
    rtl/supra2/attention_online_pv_batch.sv] {
    read_verilog -sv [file join $repo $source]
}

set clock_file [file join $out_dir clock.xdc]
set clock_stream [open $clock_file w]
puts $clock_stream [format {create_clock -name pv_clk -period %s [get_ports clk]} $clock_period_ns]
close $clock_stream
read_xdc $clock_file
set_param general.maxThreads 1
synth_design -top supra2_attention_online_pv_batch \
    -part xczu7ev-ffvc1156-2-e -mode out_of_context \
    -generic {TOKENS=256 HEAD_DIM=64 EXTERNAL_ARRAY=1} \
    -flatten_hierarchy none -directive RuntimeOptimized
write_checkpoint -force [file join $out_dir synthesized.dcp]
report_utilization -file [file join $out_dir synth_utilization.rpt]
report_timing_summary -delay_type max -max_paths 30 \
    -file [file join $out_dir synth_timing_summary.rpt]
report_drc -file [file join $out_dir synth_drc.rpt]
exit
