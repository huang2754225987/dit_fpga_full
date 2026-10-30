module probe_systolic_array_parent(
    input wire clk,
    input wire rst_n,
    input wire clear_acc,
    input wire en,
    input wire a_unsigned,
    input wire [5:0] active_rows,
    input wire signed [7:0] a_col [32],
    input wire signed [7:0] b_row [32],
    output wire signed [31:0] acc_out [32][32],
    output wire acc_valid
);
    systolic_array #(
        .M(32), .N(32), .DATA_W(8), .ACC_W(32),
        .UNSIGNED_A(1'b0), .RUNTIME_UNSIGNED_A(1'b1),
        .PIPE_STAGE(32)
    ) u_array (
        .clk(clk),
        .rst_n(rst_n),
        .clear_acc(clear_acc),
        .en(en),
        .a_unsigned(a_unsigned),
        .active_rows(active_rows),
        .a_col(a_col),
        .b_row(b_row),
        .acc_out(acc_out),
        .acc_valid(acc_valid)
    );
endmodule
