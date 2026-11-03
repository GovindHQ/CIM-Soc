// =============================================================================
// weight_memory.v -- weight storage.
//
// One word = one weight ROW (COLS elements of W_BITS each). A 32x32 weight
// tile occupies ROWS consecutive words starting at its base address:
//
//     mem[weight_base + r] = W[r][0..COLS-1]      (r = contraction index K)
//
//   write port : CPU / testbench   (fills the memory before START)
//   read  port : array_controller  (drives rd_addr = weight_base + w_row)
//
// Read is combinational (register array). A real SRAM would add one cycle.
// =============================================================================
module weight_memory #(
    parameter COLS   = 32,
    parameter W_BITS = 8,
    parameter DEPTH  = 64,       // rows held (64 = two 32x32 tiles)
    parameter AW     = 8
) (
    input  wire                   clk,

    // host write port
    input  wire                   wr_en,
    input  wire [AW-1:0]          wr_addr,
    input  wire [COLS*W_BITS-1:0] wr_data,

    // controller read port
    input  wire [AW-1:0]          rd_addr,
    output wire [COLS*W_BITS-1:0] rd_data
);

    reg [COLS*W_BITS-1:0] mem [0:DEPTH-1];

    always @(posedge clk)
        if (wr_en) mem[wr_addr] <= wr_data;

    assign rd_data = mem[rd_addr];

endmodule
