// =============================================================================
// command_table.v -- the operation table the CPU generates.
//
// One word = one command (see command_defs.vh). The controller reads from
// address 0 upward until it reaches CMD_END.
//
//   write port : CPU / testbench   (fills the table before START)
//   read  port : array_controller  (drives rd_addr = pc)
// =============================================================================
`include "command_defs.vh"

module command_table #(
    parameter DEPTH = 4096,
    parameter AW    = 16
) (
    input  wire              clk,

    // host write port
    input  wire              wr_en,
    input  wire [AW-1:0]     wr_addr,
    input  wire [`CMD_W-1:0] wr_data,

    // controller read port
    input  wire [AW-1:0]     rd_addr,
    output wire [`CMD_W-1:0] rd_data
);

    reg [`CMD_W-1:0] mem [0:DEPTH-1];

    always @(posedge clk)
        if (wr_en) mem[wr_addr] <= wr_data;

    assign rd_data = mem[rd_addr];

endmodule
