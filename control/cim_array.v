// =============================================================================
// cim_array.v -- behavioural model of ONE 32x32 CIM macro.
//
//   weight registers : wrow[r] = W[r][0..COLS-1], written one row per cycle
//   input register   : x[0..ROWS-1], latched on x_load (models the DAC inputs)
//   MVM              : on mvm_start,  y[c] = sum_r x[r] * W[r][c]   (signed)
//
// This is an IDEAL MAC: no DAC, charge sharing or ADC quantisation. The result
// is ready one cycle after mvm_start; the controller nevertheless waits its
// own MVM_LATENCY, which is the timing contract with the (future) analog macro.
// =============================================================================
module cim_array #(
    parameter ROWS     = 32,
    parameter COLS     = 32,
    parameter ACT_BITS = 8,
    parameter W_BITS   = 8,
    parameter ACC_W    = 32,
    // derived -- do not override
    parameter RIDX_W   = (ROWS > 1) ? $clog2(ROWS) : 1
) (
    input  wire                     clk,

    // weight write (one row per cycle)
    input  wire                     w_we,
    input  wire [RIDX_W-1:0]        w_row,
    input  wire [COLS*W_BITS-1:0]   w_data,

    // input token
    input  wire                     x_load,
    input  wire [ROWS*ACT_BITS-1:0] x_data,

    // compute
    input  wire                     mvm_start,
    output reg  [COLS*ACC_W-1:0]    y
);

    reg [COLS*W_BITS-1:0]   wrow [0:ROWS-1];
    reg [ROWS*ACT_BITS-1:0] x;

    integer r, c;
    reg signed [ACC_W-1:0] sum;

    always @(posedge clk) begin
        if (w_we)   wrow[w_row] <= w_data;
        if (x_load) x <= x_data;

        if (mvm_start) begin
            for (c = 0; c < COLS; c = c + 1) begin
                sum = 0;
                for (r = 0; r < ROWS; r = r + 1)
                    sum = sum + $signed(x[r*ACT_BITS +: ACT_BITS])
                              * $signed(wrow[r][c*W_BITS +: W_BITS]);
                y[c*ACC_W +: ACC_W] <= sum;
            end
        end
    end

endmodule
