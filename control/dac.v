// =============================================================================
// dac.v -- behavioural input DAC bank of ONE array (one channel per row).
//
//   input_load : latch the digital codes for this input vector
//   dac_start  : begin conversion. v_ia is UNKNOWN (X) for DAC_LATENCY cycles,
//                then holds the converted value until the next dac_start.
//
// v_ia is the signed analog-equivalent input, as in cim_sim/analog/dac.py:
// code 0 -> 0 V, in units of the DAC's v_per_unit. The topology is unconfirmed;
// this matches the offset-binary candidates (R2R / CAPACITIVE), the simulator's
// default, for which v_per_unit = Vref/2^N in both input domains.
//
// act_signed selects how the codes are read. Post-softmax inputs are unsigned
// [0, 2^N-1] and must not be reinterpreted as signed.
//
// BEHAVIOURAL ONLY: the X window makes a controller that waits too short fail
// in simulation. Nothing inside this module tells the controller it is ready.
// =============================================================================
module dac #(
    parameter ROWS        = 32,
    parameter ACT_BITS    = 8,
    parameter DAC_LATENCY = 2
) (
    input  wire                         clk,
    input  wire                         input_load,
    input  wire [ROWS*ACT_BITS-1:0]     input_code,
    input  wire                         dac_start,
    input  wire                         act_signed,
    output reg  [ROWS*(ACT_BITS+1)-1:0] v_ia
);

    reg [ROWS*ACT_BITS-1:0] code;
    reg [15:0]              cnt = 0;

    integer r;
    reg sign_bit;

    always @(posedge clk) begin
        if (input_load) code <= input_code;

        if (dac_start) begin
            v_ia <= {ROWS*(ACT_BITS+1){1'bx}};
            cnt  <= DAC_LATENCY;
        end else if (cnt != 0) begin
            cnt <= cnt - 1'b1;
            if (cnt == 1) begin
                for (r = 0; r < ROWS; r = r + 1) begin
                    sign_bit = act_signed & code[r*ACT_BITS + ACT_BITS-1];
                    v_ia[r*(ACT_BITS+1) +: ACT_BITS+1] <= {sign_bit, code[r*ACT_BITS +: ACT_BITS]};
                end
            end
        end
    end

endmodule
