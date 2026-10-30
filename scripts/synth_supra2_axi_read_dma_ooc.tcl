set repo [file normalize [file join [file dirname [info script]] ..]]
set out_dir [file join $repo build supra2_axi_read_dma vivado_ooc]
file mkdir $out_dir

read_verilog -sv [file join $repo rtl supra2 supra2_axi_read_dma.sv]
synth_design -top supra2_axi_read_dma -part xczu7ev-ffvc1156-2-e \
    -mode out_of_context
create_clock -name dma_clk -period 5.000 [get_ports clk]

report_utilization -file [file join $out_dir synth_utilization.rpt]
report_timing_summary -delay_type max -max_paths 20 \
    -file [file join $out_dir synth_timing_summary.rpt]

opt_design
place_design
phys_opt_design
route_design
report_utilization -file [file join $out_dir routed_utilization.rpt]
report_timing_summary -delay_type max -max_paths 20 \
    -file [file join $out_dir routed_timing_summary.rpt]
report_drc -file [file join $out_dir routed_drc.rpt]
write_checkpoint -force [file join $out_dir supra2_axi_read_dma_ooc_routed.dcp]
exit
