// =============================================================================
// adc.v -- behavioural pre-ADC gain + SAR ADC bank of ONE array (one per column).
//
//   adc_start : sample v_oa. adc_code is UNKNOWN (X) for ADC_LATENCY cycles while
//               the search runs, then holds until the next adc_start.
//
// Same transfer as cim_sim/analog/adc.py (capacitive DAC, Vref_dac = Vref_adc):
//
//     V_ADC_IN = G * V_OA = G * K_MAC * v_oa
//     d = floor(V_ADC_IN / LSB + 1/2),  LSB = Vref/2^N,  clipped to [-2^(N-1), 2^(N-1)-1]
//
//     LSB / K_MAC = m * 2^B * 2^DAC_BITS / 2^ADC_BITS = 2^ADC_SHIFT  (=2^13 at 8b/8b/8b)
//
// so d = floor((G*v_oa + 2^(ADC_SHIFT-1)) / 2^ADC_SHIFT). The result is a SIGNED
// code (the offset-binary re-centring is already done): 0 V -> d = 0, so codes
// from different depth blocks add directly in the accumulator.
//
// G (adc_gain) is an integer configuration value set per operation by CMD_CONFIG.
// It is never derived from data. Clipping is counted, never hidden.
//
// BEHAVIOURAL ONLY: equals the SAR search's ideal round-to-nearest in one step.
// =============================================================================
module adc #(
    parameter COLS        = 32,
    parameter ANALOG_W    = 22,
    parameter ADC_BITS    = 8,
    parameter ADC_SHIFT   = 13,
    parameter ADC_LATENCY = 9
) (
    input  wire                     clk,
    input  wire                     adc_start,
    input  wire [7:0]               adc_gain,
    input  wire [COLS*ANALOG_W-1:0] v_oa,
    output reg  [COLS*ADC_BITS-1:0] adc_code,
    output reg  [31:0]              clip_count     // conversions outside +/-Vref/2
);

    localparam signed [47:0] HALF_LSB = 48'sd1 << (ADC_SHIFT-1);
    localparam signed [47:0] SWING    = 48'sd1 << (ADC_SHIFT + ADC_BITS-1);  // Vref/2
    localparam signed [47:0] CODE_MAX =  (48'sd1 << (ADC_BITS-1)) - 1;
    localparam signed [47:0] CODE_MIN = -(48'sd1 << (ADC_BITS-1));

    reg [COLS*ADC_BITS-1:0] held;
    reg [15:0]              cnt = 0;
    reg [31:0]              n_clip;

    integer c;
    reg signed [47:0] v, d;

    initial clip_count = 0;

    always @(posedge clk) begin
        if (adc_start) begin
            n_clip = clip_count;
            for (c = 0; c < COLS; c = c + 1) begin
                v = $signed({1'b0, adc_gain}) * $signed(v_oa[c*ANALOG_W +: ANALOG_W]);
                if (v > SWING || v < -SWING) n_clip = n_clip + 1;
                d = (v + HALF_LSB) >>> ADC_SHIFT;
                if (d > CODE_MAX)      d = CODE_MAX;
                else if (d < CODE_MIN) d = CODE_MIN;
                held[c*ADC_BITS +: ADC_BITS] <= d[ADC_BITS-1:0];
            end
            clip_count <= n_clip;
            adc_code   <= {COLS*ADC_BITS{1'bx}};
            cnt        <= ADC_LATENCY;
        end else if (cnt != 0) begin
            cnt <= cnt - 1'b1;
            if (cnt == 1) adc_code <= held;
        end
    end

endmodule
