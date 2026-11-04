// =============================================================================
// token_buffer.v -- input (token) memory.
//
// One word = one DEPTH-BLOCK SLICE of one token: ROWS elements of ACT_BITS each.
// The CPU lays out X[N,K] as
//
//     mem[input_base + t*n_db + db] = X[t, db*ROWS .. db*ROWS+ROWS-1]
//
// with ragged last slices zero-padded. 196 tokens x 7 depth blocks = 1372 words.
//
//   write port : CPU / testbench   (fills the memory before START)
//   read  port : array_controller  (drives rd_addr)
//
// Read is combinational (register array). A real SRAM would add one cycle.
// =============================================================================
module token_buffer #(
    parameter ROWS     = 32,
    parameter ACT_BITS = 8,
    parameter DEPTH    = 4096,   // slices held
    parameter AW       = 16      // address width
) (
    input  wire                     clk,

    // host write port
    input  wire                     wr_en,
    input  wire [AW-1:0]            wr_addr,
    input  wire [ROWS*ACT_BITS-1:0] wr_data,

    // controller read port
    input  wire [AW-1:0]            rd_addr,
    output wire [ROWS*ACT_BITS-1:0] rd_data
);

    reg [ROWS*ACT_BITS-1:0] mem [0:DEPTH-1];

    always @(posedge clk)
        if (wr_en) mem[wr_addr] <= wr_data;

    assign rd_data = mem[rd_addr];

endmodule
