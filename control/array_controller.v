// =============================================================================
// array_controller.v -- the ONE controller of the CIM control path.
//
// Reads commands from the table the CPU wrote and executes them in order:
//
//   IDLE -> FETCH -> DECODE -+-> (CONFIG: set gain + input domain) ----> FETCH
//             ^              +-> LOAD_WEIGHT (ROWS cycles) ------------> FETCH
//             |              +-> CLEAR_OUTPUT (count cycles) ----------> FETCH
//             |              +-> END / illegal command ----------------> DONE
//             |              +-> COMPUTE:
//             |
//             |    LOAD_INPUT    (one cycle per array in the group)
//             |    DAC_START  -> WAIT_DAC    (DAC_LATENCY)
//             |    ARRAY_START-> WAIT_ARRAY  (ARRAY_LATENCY)
//             |    ADC_START  -> WAIT_ADC    (ADC_LATENCY)
//             |    WAIT_ACCUM                (ARRAY_ACCUM_WAIT, skipped if 0)
//             |    ACCUMULATE    (one cycle per array in the group)
//             |        |
//             |        +-- more tokens --> LOAD_INPUT
//             +------------ last token
//
// It only produces addresses, enables, start pulses and array selection. No
// token, weight or result data passes through this module. The waits are fixed
// counters: nothing comes back from the analog blocks.
// =============================================================================
`include "command_defs.vh"

module array_controller #(
    parameter ROWS             = 32,
    parameter NUM_ARRAYS       = 1,
    parameter AW               = 16,
    // all PLACEHOLDERS: the simulator holds no timing data (see README)
    parameter DAC_LATENCY      = 2,   // >= 1
    parameter ARRAY_LATENCY    = 4,   // >= 1
    parameter ADC_LATENCY      = 9,   // >= 1
    parameter ARRAY_ACCUM_WAIT = 0,   // >= 0
    // derived -- do not override
    parameter RIDX_W = (ROWS > 1) ? $clog2(ROWS) : 1
) (
    input  wire                  clk,
    input  wire                  rst_n,

    // ---------------- host ---------------------------------------------------
    input  wire                  start,
    output wire                  busy,
    output wire                  done,
    output reg                   error,      // illegal command or array range

    // ---------------- command table -----------------------------------------
    output wire [AW-1:0]         cmd_rd_addr,
    input  wire [`CMD_W-1:0]     cmd_rd_data,

    // ---------------- memories ----------------------------------------------
    output wire [AW-1:0]         input_rd_addr,
    output wire [AW-1:0]         weight_rd_addr,
    output wire [AW-1:0]         result_addr,
    output wire                  result_clear,
    output wire                  result_accum,

    // ---------------- arrays ------------------------------------------------
    output wire [7:0]            active_array,   // the one array addressed now
    output reg  [NUM_ARRAYS-1:0] group_mask,     // arrays taking part in COMPUTE
    output wire                  weight_we,
    output wire [RIDX_W-1:0]     weight_row,
    output wire                  input_load,
    output wire                  dac_start,
    output wire                  array_start,
    output wire                  adc_start,

    // ---------------- analog domain (set by CMD_CONFIG, broadcast) -----------
    output reg  [7:0]            adc_gain,
    output reg                   act_signed
);

    // ------------------------------------------------------------- states --
    localparam S_IDLE         = 4'd0;
    localparam S_FETCH        = 4'd1;
    localparam S_DECODE       = 4'd2;
    localparam S_LOAD_WEIGHT  = 4'd3;
    localparam S_CLEAR_OUTPUT = 4'd4;
    localparam S_LOAD_INPUT   = 4'd5;
    localparam S_DAC_START    = 4'd6;
    localparam S_WAIT_DAC     = 4'd7;
    localparam S_ARRAY_START  = 4'd8;
    localparam S_WAIT_ARRAY   = 4'd9;
    localparam S_ADC_START    = 4'd10;
    localparam S_WAIT_ADC     = 4'd11;
    localparam S_WAIT_ACCUM   = 4'd12;
    localparam S_ACCUMULATE   = 4'd13;
    localparam S_DONE         = 4'd14;

    reg [3:0]        state;
    reg [AW-1:0]     pc;          // next command to fetch
    reg [`CMD_W-1:0] cmd;         // command being executed
    reg [15:0]       token_idx;   // COMPUTE: token 0..count-1; CLEAR: word 0..count-1
    reg [7:0]        arr_idx;     // COMPUTE: array in the group, 0..array_count-1
    reg [RIDX_W-1:0] row;         // LOAD_WEIGHT: row 0..ROWS-1
    reg [15:0]       wait_cnt;    // cycles left in a WAIT_* state, minus one

    // ---------------------------------------------------- command fields --
    wire [3:0]  op            = cmd[`F_OP];
    wire [7:0]  array_first   = cmd[`F_ARRAY_FIRST];
    wire [7:0]  array_count   = cmd[`F_ARRAY_COUNT];
    wire [15:0] count         = cmd[`F_COUNT];
    wire        in_step       = cmd[`F_IN_STEP];
    wire        out_step      = cmd[`F_OUT_STEP];

    wire last_token = (token_idx == count - 1'b1);
    wire last_array = (arr_idx == array_count - 1'b1);

    // A COMPUTE group must lie inside the arrays that exist.
    wire [8:0] group_end = array_first + array_count;
    wire compute_ok = (array_count != 0) && (count != 0) && (group_end <= NUM_ARRAYS);

    // ------------------------------------------------------- FSM sequence --
    integer g;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            pc         <= 0;
            cmd        <= 0;
            token_idx  <= 0;
            arr_idx    <= 0;
            row        <= 0;
            wait_cnt   <= 0;
            error      <= 0;
            group_mask <= 0;
            adc_gain   <= 1;
            act_signed <= 1;
        end else begin
            case (state)
                S_IDLE, S_DONE:
                    if (start) begin
                        pc    <= 0;
                        error <= 0;
                        state <= S_FETCH;
                    end

                // ---------------------------------------------- next command
                S_FETCH: begin
                    cmd   <= cmd_rd_data;
                    pc    <= pc + 1'b1;
                    state <= S_DECODE;
                end

                S_DECODE: begin
                    token_idx <= 0;
                    arr_idx   <= 0;
                    row       <= 0;
                    case (op)
                        `CMD_END:
                            state <= S_DONE;
                        `CMD_CONFIG: begin
                            adc_gain   <= cmd[`F_GAIN];
                            act_signed <= cmd[`F_ACT_SIGNED];
                            state      <= S_FETCH;
                        end
                        `CMD_LOAD_WGT:
                            if (array_first < NUM_ARRAYS) state <= S_LOAD_WEIGHT;
                            else begin error <= 1; state <= S_DONE; end
                        `CMD_CLEAR:
                            state <= S_CLEAR_OUTPUT;
                        `CMD_COMPUTE:
                            if (compute_ok) begin
                                for (g = 0; g < NUM_ARRAYS; g = g + 1)
                                    group_mask[g] <= (g >= array_first) && (g < group_end);
                                state <= S_LOAD_INPUT;
                            end else begin error <= 1; state <= S_DONE; end
                        default: begin
                            error <= 1;
                            state <= S_DONE;
                        end
                    endcase
                end

                // ---------------------------------------------- weight reload
                S_LOAD_WEIGHT: begin
                    row <= row + 1'b1;
                    if (row == ROWS-1) state <= S_FETCH;
                end

                // ---------------------------------------- zero the result words
                S_CLEAR_OUTPUT: begin
                    token_idx <= token_idx + 1'b1;
                    if (last_token) state <= S_FETCH;
                end

                // ---------------------------------------------- one token, DAC
                S_LOAD_INPUT: begin
                    if (last_array) begin
                        arr_idx <= 0;
                        state   <= S_DAC_START;
                    end else
                        arr_idx <= arr_idx + 1'b1;
                end

                S_DAC_START: begin
                    wait_cnt <= DAC_LATENCY - 1;
                    state    <= S_WAIT_DAC;
                end

                S_WAIT_DAC: begin
                    wait_cnt <= wait_cnt - 1'b1;
                    if (wait_cnt == 0) state <= S_ARRAY_START;
                end

                // ------------------------------------------- array, analog settle
                S_ARRAY_START: begin
                    wait_cnt <= ARRAY_LATENCY - 1;
                    state    <= S_WAIT_ARRAY;
                end

                S_WAIT_ARRAY: begin
                    wait_cnt <= wait_cnt - 1'b1;
                    if (wait_cnt == 0) state <= S_ADC_START;
                end

                // ------------------------------------------------------- ADC
                S_ADC_START: begin
                    wait_cnt <= ADC_LATENCY - 1;
                    state    <= S_WAIT_ADC;
                end

                S_WAIT_ADC: begin
                    wait_cnt <= wait_cnt - 1'b1;
                    if (wait_cnt == 0) begin
                        wait_cnt <= ARRAY_ACCUM_WAIT - 1;
                        state    <= (ARRAY_ACCUM_WAIT == 0) ? S_ACCUMULATE : S_WAIT_ACCUM;
                    end
                end

                S_WAIT_ACCUM: begin
                    wait_cnt <= wait_cnt - 1'b1;
                    if (wait_cnt == 0) state <= S_ACCUMULATE;
                end

                // ------------------------------------------------ accumulation
                S_ACCUMULATE: begin
                    if (last_array) begin
                        arr_idx <= 0;
                        if (last_token)
                            state <= S_FETCH;
                        else begin
                            token_idx <= token_idx + 1'b1;
                            state     <= S_LOAD_INPUT;
                        end
                    end else
                        arr_idx <= arr_idx + 1'b1;
                end

                default:
                    state <= S_IDLE;
            endcase
        end
    end

    // --------------------------------------------------- address generation --
    assign cmd_rd_addr    = pc;
    assign weight_rd_addr = cmd[`F_WEIGHT_ADDR] + row;
    assign input_rd_addr  = cmd[`F_INPUT_ADDR]
                          + token_idx * cmd[`F_INPUT_STRIDE]
                          + (in_step ? arr_idx : 8'd0);
    assign result_addr    = (state == S_CLEAR_OUTPUT)
                          ? cmd[`F_OUTPUT_ADDR] + token_idx
                          : cmd[`F_OUTPUT_ADDR]
                            + token_idx * cmd[`F_OUTPUT_STRIDE]
                            + (out_step ? arr_idx : 8'd0);

    assign active_array   = array_first + arr_idx;
    assign weight_row     = row;

    // -------------------------------------------------------- control pulses --
    assign weight_we    = (state == S_LOAD_WEIGHT);
    assign input_load   = (state == S_LOAD_INPUT);
    assign dac_start    = (state == S_DAC_START);
    assign array_start  = (state == S_ARRAY_START);
    assign adc_start    = (state == S_ADC_START);
    assign result_accum = (state == S_ACCUMULATE);
    assign result_clear = (state == S_CLEAR_OUTPUT);

    assign busy = (state != S_IDLE) && (state != S_DONE);
    assign done = (state == S_DONE);

endmodule
