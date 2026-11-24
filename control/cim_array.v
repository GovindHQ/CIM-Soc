// =============================================================================
// cim_array.v -- behavioural model of ONE 32x32 analog CIM macro.
//
//   weights    : wrow[r] = W[r][0..COLS-1], written one row per cycle
//   array_start: the input held on v_ia is applied to the array. The column
//                outputs are UNKNOWN (X) while they settle, for ARRAY_LATENCY
//                cycles, and then hold until the next start or weight write.
//
// The macro has no "compute" operation of its own: the column outputs simply
// develop once the input is applied. array_start is a control abstraction that
// gives the controller a clean sequencing point.
//
// v_oa is the analog-equivalent column output, in units of K_MAC:
//
//     v_oa[c] = sum_r  v_ia[r] * W[r][c]
//
// so physical V_OA = K_MAC * v_oa, K_MAC = v_per_unit / (m * 2^B), m = ROWS.
// Zero-padded rows still count in m (cim_sim/analog/macro.py); that is why the
// scale in adc.v depends only on ROWS and never on how full a tile is.
//
// BEHAVIOURAL ONLY: arithmetic on real numbers standing in for charge sharing.
// Signed weights are one more abstraction here; how silicon realises the sign
// is an open hardware question (CLAUDE.md section 9).
// =============================================================================
module cim_array #(
    parameter ROWS          = 32,
    parameter COLS          = 32,
    parameter ACT_BITS      = 8,
    parameter W_BITS        = 8,
    parameter ANALOG_W      = 22,
    parameter ARRAY_LATENCY = 4,
    // derived -- do not override
    parameter RIDX_W        = (ROWS > 1) ? $clog2(ROWS) : 1
) (
    input  wire                         clk,

    // weight write (one row per cycle)
    input  wire                         weight_we,
    input  wire [RIDX_W-1:0]            weight_row,
    input  wire [COLS*W_BITS-1:0]       weight_data,

    // analog input from the DAC bank
    input  wire [ROWS*(ACT_BITS+1)-1:0] v_ia,
    input  wire                         array_start,

    // analog output to the ADC bank
    output reg  [COLS*ANALOG_W-1:0]     v_oa
);

    reg [COLS*W_BITS-1:0]       wrow [0:ROWS-1];
    reg [ROWS*(ACT_BITS+1)-1:0] applied;
    reg [15:0]                  cnt = 0;

    integer r, c;
    reg signed [ANALOG_W-1:0] sum;

    always @(posedge clk) begin
        if (weight_we) begin
            wrow[weight_row] <= weight_data;
            v_oa <= {COLS*ANALOG_W{1'bx}};   // output no longer reflects the weights
        end

        if (array_start) begin
            applied <= v_ia;                 // X here if the DAC had not settled
            v_oa    <= {COLS*ANALOG_W{1'bx}};
            cnt     <= ARRAY_LATENCY;
        end else if (cnt != 0) begin
            cnt <= cnt - 1'b1;
            if (cnt == 1) begin
                for (c = 0; c < COLS; c = c + 1) begin
                    sum = 0;
                    for (r = 0; r < ROWS; r = r + 1)
                        sum = sum + $signed(applied[r*(ACT_BITS+1) +: ACT_BITS+1])
                                  * $signed(wrow[r][c*W_BITS +: W_BITS]);
                    v_oa[c*ANALOG_W +: ANALOG_W] <= sum;
                end
            end
        end
    end

endmodule
