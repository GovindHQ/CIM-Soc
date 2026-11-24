// =============================================================================
// cim_top.v -- wires the controller, the memories and NUM_ARRAYS analog channels.
//
//   CPU --> command_table --> array_controller (addresses, enables, pulses)
//
//   CPU --> token_buffer  --> dac[a] --> cim_array[a] --> adc[a] --+
//   CPU --> weight_memory ------------------^                      |
//                                                                  v
//   CPU <-- result_buffer (accumulates the selected array's codes) <
//
// One analog channel (DAC bank + array + ADC bank) per array, built by the
// generate loop below. With NUM_ARRAYS = 1 the muxes and masks disappear.
//
//   active_array : the array addressed one at a time by the shared memory ports
//                  (LOAD_WEIGHT, LOAD_INPUT, ACCUMULATE)
//   group_mask   : the arrays that get the DAC / array / ADC pulses together
// =============================================================================
`include "command_defs.vh"

module cim_top #(
    parameter NUM_ARRAYS       = 1,
    parameter ROWS             = 32,
    parameter COLS             = 32,
    parameter ACT_BITS         = 8,
    parameter W_BITS           = 8,
    parameter ADC_BITS         = 8,
    parameter ACC_W            = 32,
    parameter AW               = 16,
    parameter CMD_DEPTH        = 4096,
    parameter INPUT_DEPTH      = 4096,
    parameter WEIGHT_DEPTH     = 4096,
    parameter RESULT_DEPTH     = 8192,   // TinyViT stage 3 qkv+QK^T+AV needs 4508
    // timing: all PLACEHOLDERS, in clock cycles (see README)
    parameter DAC_LATENCY      = 2,
    parameter ARRAY_LATENCY    = 4,
    parameter ADC_LATENCY      = 9,
    parameter ARRAY_ACCUM_WAIT = 0,
    // derived -- do not override
    parameter IN_W  = ROWS * ACT_BITS,    // one input slice
    parameter WR_W  = COLS * W_BITS,      // one weight row
    parameter RES_W = COLS * ACC_W        // one result word
) (
    input  wire              clk,
    input  wire              rst_n,

    // CPU: fill the command table
    input  wire              cmd_wr_en,
    input  wire [AW-1:0]     cmd_wr_addr,
    input  wire [`CMD_W-1:0] cmd_wr_data,

    // CPU: fill the token buffer
    input  wire              input_wr_en,
    input  wire [AW-1:0]     input_wr_addr,
    input  wire [IN_W-1:0]   input_wr_data,

    // CPU: fill the weight memory
    input  wire              weight_wr_en,
    input  wire [AW-1:0]     weight_wr_addr,
    input  wire [WR_W-1:0]   weight_wr_data,

    // CPU: run / status
    input  wire              start,
    output wire              busy,
    output wire              done,
    output wire              error,
    output reg  [31:0]       adc_clip_count,

    // CPU: read results
    input  wire [AW-1:0]     result_rd_addr,
    output wire [RES_W-1:0]  result_rd_data
);

    localparam RIDX_W    = (ROWS > 1) ? $clog2(ROWS) : 1;
    localparam ANALOG_W  = ACT_BITS + W_BITS + RIDX_W + 1;   // holds any column sum
    localparam CODE_W    = COLS * ADC_BITS;                  // one array's codes
    // LSB / K_MAC = m * 2^B * 2^DAC_BITS / 2^ADC_BITS, see adc.v
    localparam ADC_SHIFT = RIDX_W + W_BITS + ACT_BITS - ADC_BITS;

    // controller <-> datapath
    wire [`CMD_W-1:0]     cmd_rd_data;
    wire [AW-1:0]         cmd_rd_addr, input_rd_addr, weight_rd_addr, result_addr;
    wire [IN_W-1:0]       input_rd_data;
    wire [WR_W-1:0]       weight_rd_data;
    wire                  result_clear, result_accum;
    wire [7:0]            active_array;
    wire [NUM_ARRAYS-1:0] group_mask;
    wire                  weight_we, input_load, dac_start, array_start, adc_start;
    wire [RIDX_W-1:0]     weight_row;
    wire [7:0]            adc_gain;
    wire                  act_signed;

    wire [NUM_ARRAYS*CODE_W-1:0] code_all;      // every array's ADC codes, packed
    wire [NUM_ARRAYS*32-1:0]     clip_all;
    wire [CODE_W-1:0]            code_active = code_all[active_array*CODE_W +: CODE_W];

    // ------------------------------------------------------------ control --
    array_controller #(
        .ROWS(ROWS), .NUM_ARRAYS(NUM_ARRAYS), .AW(AW),
        .DAC_LATENCY(DAC_LATENCY), .ARRAY_LATENCY(ARRAY_LATENCY),
        .ADC_LATENCY(ADC_LATENCY), .ARRAY_ACCUM_WAIT(ARRAY_ACCUM_WAIT)
    ) u_ctrl (
        .clk(clk), .rst_n(rst_n),
        .start(start), .busy(busy), .done(done), .error(error),
        .cmd_rd_addr(cmd_rd_addr), .cmd_rd_data(cmd_rd_data),
        .input_rd_addr(input_rd_addr), .weight_rd_addr(weight_rd_addr),
        .result_addr(result_addr), .result_clear(result_clear), .result_accum(result_accum),
        .active_array(active_array), .group_mask(group_mask),
        .weight_we(weight_we), .weight_row(weight_row), .input_load(input_load),
        .dac_start(dac_start), .array_start(array_start), .adc_start(adc_start),
        .adc_gain(adc_gain), .act_signed(act_signed)
    );

    // ----------------------------------------------------------- memories --
    command_table #(.DEPTH(CMD_DEPTH), .AW(AW)) u_cmd (
        .clk(clk),
        .wr_en(cmd_wr_en), .wr_addr(cmd_wr_addr), .wr_data(cmd_wr_data),
        .rd_addr(cmd_rd_addr), .rd_data(cmd_rd_data)
    );

    token_buffer #(.ROWS(ROWS), .ACT_BITS(ACT_BITS), .DEPTH(INPUT_DEPTH), .AW(AW)) u_input (
        .clk(clk),
        .wr_en(input_wr_en), .wr_addr(input_wr_addr), .wr_data(input_wr_data),
        .rd_addr(input_rd_addr), .rd_data(input_rd_data)
    );

    weight_memory #(.COLS(COLS), .W_BITS(W_BITS), .DEPTH(WEIGHT_DEPTH), .AW(AW)) u_wmem (
        .clk(clk),
        .wr_en(weight_wr_en), .wr_addr(weight_wr_addr), .wr_data(weight_wr_data),
        .rd_addr(weight_rd_addr), .rd_data(weight_rd_data)
    );

    result_buffer #(.COLS(COLS), .ADC_BITS(ADC_BITS), .ACC_W(ACC_W),
                    .DEPTH(RESULT_DEPTH), .AW(AW)) u_res (
        .clk(clk),
        .clear(result_clear), .accum(result_accum), .addr(result_addr),
        .adc_code(code_active),
        .rd_addr(result_rd_addr), .rd_data(result_rd_data)
    );

    // ---------------------------------- one analog channel per array -------
    genvar g;
    generate
        for (g = 0; g < NUM_ARRAYS; g = g + 1) begin : g_arr
            wire [ROWS*(ACT_BITS+1)-1:0] v_ia;
            wire [COLS*ANALOG_W-1:0]     v_oa;

            dac #(.ROWS(ROWS), .ACT_BITS(ACT_BITS), .DAC_LATENCY(DAC_LATENCY)) u_dac (
                .clk(clk),
                .input_load(input_load && (active_array == g)),
                .input_code(input_rd_data),
                .dac_start(dac_start && group_mask[g]),
                .act_signed(act_signed),
                .v_ia(v_ia)
            );

            cim_array #(.ROWS(ROWS), .COLS(COLS), .ACT_BITS(ACT_BITS), .W_BITS(W_BITS),
                        .ANALOG_W(ANALOG_W), .ARRAY_LATENCY(ARRAY_LATENCY)) u_arr (
                .clk(clk),
                .weight_we(weight_we && (active_array == g)),
                .weight_row(weight_row),
                .weight_data(weight_rd_data),
                .v_ia(v_ia),
                .array_start(array_start && group_mask[g]),
                .v_oa(v_oa)
            );

            adc #(.COLS(COLS), .ANALOG_W(ANALOG_W), .ADC_BITS(ADC_BITS),
                  .ADC_SHIFT(ADC_SHIFT), .ADC_LATENCY(ADC_LATENCY)) u_adc (
                .clk(clk),
                .adc_start(adc_start && group_mask[g]),
                .adc_gain(adc_gain),
                .v_oa(v_oa),
                .adc_code(code_all[g*CODE_W +: CODE_W]),
                .clip_count(clip_all[g*32 +: 32])
            );
        end
    endgenerate

    integer i;
    always @* begin
        adc_clip_count = 0;
        for (i = 0; i < NUM_ARRAYS; i = i + 1)
            adc_clip_count = adc_clip_count + clip_all[i*32 +: 32];
    end

endmodule
