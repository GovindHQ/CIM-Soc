// =============================================================================
// token_buffer.v -- activation / token memory.
//
// One word = one whole token (ROWS elements of ACT_BITS each), so a token is
// moved into the CIM array in a single cycle.
//
//   write port : CPU / testbench   (fills the buffer before START)
//   read  port : array_controller  (drives rd_addr = token_base + token_idx)
//
// Read is combinational (register array). A real SRAM would add one cycle.
// =============================================================================
module token_buffer #(
    parameter ROWS     = 32,
    parameter ACT_BITS = 8,
    parameter DEPTH    = 4,      // number of tokens held
    parameter AW       = 8       // address width
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
