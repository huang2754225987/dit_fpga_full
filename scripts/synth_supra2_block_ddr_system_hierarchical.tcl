set repo [file normalize [file join [file dirname [info script]] ..]]
set period_ns 2.500
if {[llength $argv] > 0} { set period_ns [lindex $argv 0] }
set run_tag [clock format [clock seconds] -format %Y%m%d_%H%M%S]
if {[llength $argv] > 1} { set run_tag [lindex $argv 1] }
set out_dir [file join $repo build supra2_block_ddr_system_ooc "hierarchical_${run_tag}"]
file mkdir $out_dir
set_param general.maxThreads 1
foreach source [list \
    rtl/pkg/isa_pkg.sv \
    rtl/pkg/fp16_utils_pkg.sv \
    rtl/ops/graph_exp_lut_fp164k.sv \
    rtl/ops/recip_lut4k.sv \
    rtl/ops/softmax_engine.sv \
    rtl/supra2/fp32_add_rne_pipeline.sv \
    rtl/supra2/hadamard_transform.sv \
    rtl/supra2/activation_quantizer.sv \
    rtl/supra2/qlinear_epilogue.sv \
    rtl/supra2/layernorm_modulate_32lane.sv \
    rtl/supra2/gelu_tanh_32lane.sv \
    rtl/supra2/residual_gate_32lane.sv \
    rtl/supra2/attention_online_pv_batch.sv \
    rtl/supra2/supra2_sync_word_ram.sv \
    rtl/supra2/supra2_gemm_engine.sv \
    rtl/supra2/supra2_block_top.sv \
    build/supra2_block13_ddr_traffic/block13_ddr_layout.sv \
    rtl/supra2/supra2_block_dma.sv \
    rtl/axi_to_mig_bridge.sv \
    scripts/systolic_array_blackbox.sv \
    rtl/supra2/supra2_block_ddr_system_top.sv] {
    read_verilog -sv [file join $repo $source]
}
set clock_file [file join $out_dir clock.xdc]
set clock_stream [open $clock_file w]
puts $clock_stream [format {create_clock -name system_clk -period %s [get_ports clk]} $period_ns]
close $clock_stream
synth_design -top supra2_block_ddr_system_top -part xczu7ev-ffvc1156-2-e \
    -generic {AXI_DATA_W=512} -mode out_of_context \
    -flatten_hierarchy none -directive RuntimeOptimized -no_timing_driven
set array_cell [get_cells -hier -quiet -filter {ORIG_REF_NAME == systolic_array || REF_NAME == systolic_array}]
if {[llength $array_cell] != 1} {
    error "Expected exactly one black-box systolic_array cell, found [llength $array_cell]: $array_cell"
}
read_checkpoint -cell $array_cell [file join $repo build supra2_block_ddr_system_ooc "shared_array_${run_tag}" routed.dcp]
write_checkpoint -force [file join $out_dir synthesized.dcp]
report_utilization -file [file join $out_dir synth_utilization.rpt]
exit
