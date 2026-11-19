// =============================================================================
// result_buffer.v -- digital accumulator + result memory.
//
// One word = one output tile of one token: COLS accumulators of ACC_W bits. The
// partial sums live here and accumulate in place, so the last ACCUMULATE of a
// word leaves the final result in it. The CPU lays out Y[N,M] as
//
//     mem[output_base + t*n_ob + ob] = Y[t, ob*COLS .. ob*COLS+COLS-1]
//
//   clear : mem[addr] <= 0
//   accum : mem[addr] <= mem[addr] + sign_extend(adc_code), per column
//
// Values are integer sums of signed ADC codes, exactly like `acc += d` in
// cim_sim/matmul.py. The single rescale to real units (LSB_P * S_X * S_W) belongs
// to the CPU after DONE; the two scales stay separate.
// =============================================================================
module result_buffer #(
    parameter COLS     = 32,
    parameter ADC_BITS = 8,
    parameter ACC_W    = 32,
    parameter DEPTH    = 8192,
    parameter AW       = 16
) (
    input  wire                     clk,

    // controller port
    input  wire                     clear,
    input  wire                     accum,
    input  wire [AW-1:0]            addr,
    input  wire [COLS*ADC_BITS-1:0] adc_code,

    // host read port
    input  wire [AW-1:0]            rd_addr,
    output wire [COLS*ACC_W-1:0]    rd_data
);

    reg [COLS*ACC_W-1:0] mem [0:DEPTH-1];

    wire [COLS*ACC_W-1:0] current = mem[addr];
    wire [COLS*ACC_W-1:0] sum;

    genvar c;
    generate
        for (c = 0; c < COLS; c = c + 1) begin : g_col
            assign sum[c*ACC_W +: ACC_W] = current[c*ACC_W +: ACC_W]
                + {{(ACC_W-ADC_BITS){adc_code[c*ADC_BITS + ADC_BITS-1]}},
                   adc_code[c*ADC_BITS +: ADC_BITS]};
        end
    endgenerate

    always @(posedge clk) begin
        if (clear)      mem[addr] <= {COLS*ACC_W{1'b0}};
        else if (accum) mem[addr] <= sum;
    end

    assign rd_data = mem[rd_addr];

endmodule
