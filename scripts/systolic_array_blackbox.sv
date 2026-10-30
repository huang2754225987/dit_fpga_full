`default_nettype none
(* black_box = "yes", syn_black_box = 1 *)
module systolic_array #(
    parameter int M = 32,
    parameter int N = 32,
    parameter int DATA_W = 8,
    parameter int ACC_W = 32,
    parameter bit UNSIGNED_A = 1'b0,
    parameter bit RUNTIME_UNSIGNED_A = 1'b0,
    parameter int PIPE_STAGE = 32
)(
    input wire clk,
    input wire rst_n,
    input wire clear_acc,
    input wire en,
    input wire a_unsigned,
    input wire [5:0] active_rows,
    input wire signed [DATA_W-1:0] a_col [M],
    input wire signed [DATA_W-1:0] b_row [N],
    output logic signed [ACC_W-1:0] acc_out [M][N],
    output logic acc_valid
);
endmodule
`default_nettype wire
