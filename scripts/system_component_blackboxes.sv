`default_nettype none
(* black_box = "yes" *)
module supra2_block_top(
    input wire clk, input wire rst_n, input wire start,
    input wire [4:0] start_stage, input wire [3:0] start_head,
    input wire [15:0] start_ni, input wire [15:0] stop_ni,
    output logic busy, output logic done,
    input wire ld_en, input wire [3:0] ld_target, input wire [20:0] ld_addr,
    input wire [15:0] ld_data, input wire ld_vec_en,
    input wire [3:0] ld_vec_target, input wire [15:0] ld_vec_beat,
    input wire [511:0] ld_vec_data,
    input wire param_vec_wr_en, input wire [7:0] param_vec_wr_beat,
    input wire [511:0] param_vec_wr_data,
    input wire wt_wr_en, input wire [15:0] wt_wr_beat,
    input wire [255:0] wt_wr_data, input wire sb_wr_en,
    input wire sb_wr_sel, input wire [15:0] sb_wr_addr,
    input wire [31:0] sb_wr_data, input wire [14:0] result_rd_beat,
    output wire [127:0] result_rd_data,
    input wire [12:0] result_vec_rd_beat,
    output wire [511:0] result_vec_rd_data,
    output logic wload_req, output logic [3:0] wload_stage,
    input wire wload_ack, output logic [4:0] stage_id
);
endmodule

(* black_box = "yes" *)
module supra2_block_dma #(
    parameter logic [31:0] DDR_BASE = 32'h0010_0000,
    parameter logic [31:0] OUTPUT_BASE = 32'h0080_0000,
    parameter integer AXI_DATA_W = 128
)(
    input wire clk, input wire rst_n, input wire start,
    output logic busy, output logic done, output logic dma_error,
    output logic block_start, input wire block_done,
    output logic ld_en, output logic [3:0] ld_target,
    output logic [20:0] ld_addr, output logic [15:0] ld_data,
    output logic wt_wr_en, output logic [15:0] wt_wr_beat,
    output logic [255:0] wt_wr_data, output logic sb_wr_en,
    output logic sb_wr_sel, output logic [15:0] sb_wr_addr,
    output logic [31:0] sb_wr_data, input wire wload_req,
    input wire [3:0] wload_stage, output logic wload_ack,
    output logic [14:0] result_rd_beat,
    input wire [127:0] result_rd_data,
    output wire [3:0] m_axi_awid, output logic [31:0] m_axi_awaddr,
    output logic [7:0] m_axi_awlen, output wire [2:0] m_axi_awsize,
    output wire [1:0] m_axi_awburst, output logic m_axi_awvalid,
    input wire m_axi_awready, output wire [AXI_DATA_W-1:0] m_axi_wdata,
    output wire [AXI_DATA_W/8-1:0] m_axi_wstrb,
    output wire m_axi_wlast, output logic m_axi_wvalid,
    input wire m_axi_wready, input wire [3:0] m_axi_bid,
    input wire [1:0] m_axi_bresp, input wire m_axi_bvalid,
    output logic m_axi_bready, output wire [3:0] m_axi_arid,
    output logic [31:0] m_axi_araddr, output logic [7:0] m_axi_arlen,
    output wire [2:0] m_axi_arsize, output wire [1:0] m_axi_arburst,
    output logic m_axi_arvalid, input wire m_axi_arready,
    input wire [3:0] m_axi_rid,
    input wire [AXI_DATA_W-1:0] m_axi_rdata,
    input wire [1:0] m_axi_rresp, input wire m_axi_rlast,
    input wire m_axi_rvalid, output logic m_axi_rready
);
endmodule

(* black_box = "yes" *)
module axi_to_mig_bridge #(
    parameter int AXI_DATA_W = 128,
    parameter int AXI_ADDR_W = 32,
    parameter int AXI_ID_W = 4,
    parameter int MIG_DATA_W = 64,
    parameter int MIG_ADDR_W = 29,
    parameter int MIG_MASK_W = 8,
    parameter int RATIO = AXI_DATA_W / MIG_DATA_W,
    parameter int READ_FIFO_BEATS = 8
)(
    input wire clk, input wire rst_n,
    input wire [AXI_ID_W-1:0] s_axi_awid,
    input wire [AXI_ADDR_W-1:0] s_axi_awaddr,
    input wire [7:0] s_axi_awlen, input wire [2:0] s_axi_awsize,
    input wire [1:0] s_axi_awburst, input wire s_axi_awvalid,
    output wire s_axi_awready,
    input wire [AXI_DATA_W-1:0] s_axi_wdata,
    input wire [AXI_DATA_W/8-1:0] s_axi_wstrb,
    input wire s_axi_wlast, input wire s_axi_wvalid,
    output wire s_axi_wready,
    output wire [AXI_ID_W-1:0] s_axi_bid,
    output wire [1:0] s_axi_bresp, output wire s_axi_bvalid,
    input wire s_axi_bready,
    input wire [AXI_ID_W-1:0] s_axi_arid,
    input wire [AXI_ADDR_W-1:0] s_axi_araddr,
    input wire [7:0] s_axi_arlen, input wire [2:0] s_axi_arsize,
    input wire [1:0] s_axi_arburst, input wire s_axi_arvalid,
    output wire s_axi_arready,
    output wire [AXI_ID_W-1:0] s_axi_rid,
    output wire [AXI_DATA_W-1:0] s_axi_rdata,
    output wire [1:0] s_axi_rresp, output wire s_axi_rlast,
    output wire s_axi_rvalid, input wire s_axi_rready,
    output wire [MIG_ADDR_W-1:0] app_addr, output wire [2:0] app_cmd,
    output wire app_en, output wire app_hi_pri,
    output wire [MIG_DATA_W-1:0] app_wdf_data,
    output wire app_wdf_end, output wire [MIG_MASK_W-1:0] app_wdf_mask,
    output wire app_wdf_wren,
    input wire [MIG_DATA_W-1:0] app_rd_data,
    input wire app_rd_data_end, input wire app_rd_data_valid,
    input wire app_rdy, input wire app_wdf_rdy
);
endmodule
`default_nettype wire
