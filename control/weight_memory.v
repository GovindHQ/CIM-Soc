// =============================================================================
// weight_memory.v -- weight storage.
//
// One word = one weight ROW of one tile (COLS elements of W_BITS each). One tile
// is ROWS consecutive words. The CPU decides the layout and puts each tile's
// base address in the CMD_LOAD_WGT command.
//
//   write port : CPU / testbench   (fills the memory before START)
//   read  port : array_controller  (drives rd_addr = weight_addr + row)
//
// Read is combinational (register array). A real SRAM would add one cycle.
// =============================================================================
module weight_memory #(
    parameter COLS   = 32,
    parameter W_BITS = 8,
    parameter DEPTH  = 4096,     // rows held (4096 = 128 tiles of 32 rows)
    parameter AW     = 16
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
