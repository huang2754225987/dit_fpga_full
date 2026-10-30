set repo [file normalize [file join [file dirname [info script]] ..]]
set clock_period_ns 2.500
if {[llength $argv] > 0} { set clock_period_ns [lindex $argv 0] }
set run_tag [clock format [clock seconds] -format %Y%m%d_%H%M%S]
if {[llength $argv] > 1} { set run_tag [lindex $argv 1] }
set period_tag [string map {. p} [format %.3f $clock_period_ns]]
set out_dir [file join $repo build supra2_block_ddr_system_ooc \
    "vivado_ooc_${period_tag}ns_${run_tag}"]
file mkdir $out_dir

set sources [list \
    rtl/pkg/isa_pkg.sv \
    rtl/pkg/fp16_utils_pkg.sv \
    rtl/ops/graph_exp_lut_fp164k.sv \
    rtl/ops/recip_lut4k.sv \
    rtl/ops/softmax_engine.sv \
    rtl/gemm/mac_int8.sv \
    rtl/gemm/pe.sv \
    rtl/gemm/systolic_array.sv \
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
    rtl/supra2/supra2_tile_program_sequencer.sv \
    rtl/supra2/supra2_tile_program_ddr_loader.sv \
    build/supra2_block13_ddr_traffic/block13_ddr_layout.sv \
    rtl/supra2/supra2_block_dma.sv \
    rtl/axi_to_mig_bridge.sv \
    rtl/supra2/supra2_block_ddr_system_top.sv]
foreach source $sources {
    read_verilog -sv [file join $repo $source]
}

set clock_file [file join $out_dir clock.xdc]
set clock_stream [open $clock_file w]
puts $clock_stream [format {create_clock -name system_clk -period %s [get_ports clk]} $clock_period_ns]
close $clock_stream
set_param general.maxThreads 4
synth_design -top supra2_block_ddr_system_top -part xczu7ev-ffvc1156-2-e \
    -generic {AXI_DATA_W=512 USE_TILE_PROGRAM=1 TILE_PROGRAM_WORDS=259 TILE_DEPENDENCY_WORDS=1052} -mode out_of_context \
    -flatten_hierarchy none -directive RuntimeOptimized -no_timing_driven
# Save the complete synthesized hierarchy before loading timing constraints.
# Timing graph construction previously exhausted the host commit limit in the
# same process that elaborated this large top. The implementation stage opens
# this checkpoint in a fresh Vivado process and applies the 2.5 ns clock there.
write_checkpoint -force [file join $out_dir synthesized.dcp]
report_utilization -file [file join $out_dir synth_utilization.rpt]
exit
