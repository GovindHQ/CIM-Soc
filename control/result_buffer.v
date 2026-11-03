// =============================================================================
// result_buffer.v -- MVM output storage (future: partial-sum buffer).
//
// One word = one MVM result (COLS column sums of ACC_W bits each).
//
//   write port : array_controller  (STORE_RESULT: result_base + token_idx)
//   read  port : CPU / testbench   (reads the result after DONE)
// =============================================================================
module result_buffer #(
    parameter COLS  = 32,
    parameter ACC_W = 32,
    parameter DEPTH = 4,
    parameter AW    = 8
) (
    input  wire                  clk,

    // controller write port
    input  wire                  wr_en,
    input  wire [AW-1:0]         wr_addr,
    input  wire [COLS*ACC_W-1:0] wr_data,

    // host read port
    input  wire [AW-1:0]         rd_addr,
    output wire [COLS*ACC_W-1:0] rd_data
);

    reg [COLS*ACC_W-1:0] mem [0:DEPTH-1];

    always @(posedge clk)
        if (wr_en) mem[wr_addr] <= wr_data;

    assign rd_data = mem[rd_addr];

endmodule
