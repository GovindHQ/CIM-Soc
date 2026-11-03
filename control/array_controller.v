// =============================================================================
// array_controller.v -- the ONE controller of the simplified CIM control path.
//
// The host sets cfg_* and pulses start. The controller then runs, on its own:
//
//   IDLE -> LOAD_WEIGHT (ROWS cycles) -> LOAD_TOKEN -> START_MVM
//        -> WAIT_MVM (MVM_LATENCY cycles) -> STORE_RESULT -> DONE
//                         ^                                |
//                         +---- next token (num_tokens) ---+
//
// Weights are loaded ONCE and stay resident (weight-stationary); every token
// then runs LOAD_TOKEN .. STORE_RESULT against the same weights.
//
// It only produces ADDRESSES and STROBES. Token, weight and result data flow
// directly between the memories and the array (see cim_top.v).
//
// Address generation (TOKEN_SIZE = 1 memory word per token):
//   token  address = token_base  + token_idx
//   weight address = weight_base + w_row
//   result address = result_base + token_idx
// =============================================================================
module array_controller #(
    parameter ROWS        = 32,
    parameter NUM_ARRAYS  = 1,
    parameter AW          = 8,
    parameter MVM_LATENCY = 10,   // cycles from mvm_start to result valid (>= 1, PLACEHOLDER)
    // derived -- do not override
    parameter AID_W  = (NUM_ARRAYS > 1) ? $clog2(NUM_ARRAYS) : 1,
    parameter RIDX_W = (ROWS > 1) ? $clog2(ROWS) : 1
) (
    input  wire              clk,
    input  wire              rst_n,

    // ---------------- host: configuration + start / status -----------------
    input  wire              start,
    input  wire [AW-1:0]     cfg_token_base,
    input  wire [AW-1:0]     cfg_weight_base,
    input  wire [AW-1:0]     cfg_result_base,
    input  wire [AW-1:0]     cfg_num_tokens,    // tokens to process (0 behaves as 1)
    input  wire [AID_W-1:0]  cfg_array_id,      // must be < NUM_ARRAYS
    output wire              busy,
    output wire              done,

    // ---------------- memories ---------------------------------------------
    output wire [AW-1:0]     tok_rd_addr,
    output wire [AW-1:0]     w_rd_addr,
    output wire              res_we,
    output wire [AW-1:0]     res_wr_addr,

    // ---------------- CIM array(s) -----------------------------------------
    output wire [AID_W-1:0]  arr_sel,
    output wire              arr_w_we,
    output wire [RIDX_W-1:0] arr_w_row,
    output wire              arr_x_load,
    output wire              arr_mvm_start
);

    // ------------------------------------------------------------- states --
    localparam S_IDLE         = 3'd0;
    localparam S_LOAD_WEIGHT  = 3'd1;
    localparam S_LOAD_TOKEN   = 3'd2;
    localparam S_START_MVM    = 3'd3;
    localparam S_WAIT_MVM     = 3'd4;
    localparam S_STORE_RESULT = 3'd5;
    localparam S_DONE         = 3'd6;

    reg [2:0]        state;

    // configuration registers (latched on start)
    reg [AW-1:0]     token_base, weight_base, result_base, num_tokens;
    reg [AID_W-1:0]  array_id;

    // loop counters
    reg [AW-1:0]     token_idx;   // token being processed, 0 .. num_tokens-1
    reg [RIDX_W-1:0] w_row;       // weight row being written
    reg [15:0]       lat_cnt;     // MVM latency counter

    // ------------------------------------------------------- FSM sequence --
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            token_base  <= 0;
            weight_base <= 0;
            result_base <= 0;
            num_tokens  <= 0;
            array_id    <= 0;
            token_idx   <= 0;
            w_row       <= 0;
            lat_cnt     <= 0;
        end else begin
            case (state)
                S_IDLE, S_DONE: begin
                    if (start) begin
                        token_base  <= cfg_token_base;
                        weight_base <= cfg_weight_base;
                        result_base <= cfg_result_base;
                        num_tokens  <= cfg_num_tokens;
                        array_id    <= cfg_array_id;
                        token_idx   <= 0;
                        w_row       <= 0;
                        state       <= S_LOAD_WEIGHT;
                    end
                end

                S_LOAD_WEIGHT: begin                 // one weight row per cycle
                    w_row <= w_row + 1'b1;
                    if (w_row == ROWS-1)
                        state <= S_LOAD_TOKEN;
                end

                S_LOAD_TOKEN:                        // token -> array input register
                    state <= S_START_MVM;

                S_START_MVM: begin                   // analog start
                    lat_cnt <= 0;
                    state   <= S_WAIT_MVM;
                end

                S_WAIT_MVM: begin                    // fixed latency
                    lat_cnt <= lat_cnt + 1'b1;
                    if (lat_cnt == MVM_LATENCY-1)    // future: replace with arr_mvm_done
                        state <= S_STORE_RESULT;
                end

                S_STORE_RESULT: begin                // array result -> result buffer
                    if (token_idx + 1'b1 >= num_tokens)
                        state <= S_DONE;             // last token
                    else begin
                        token_idx <= token_idx + 1'b1;
                        state     <= S_LOAD_TOKEN;   // next token, same weights
                    end
                end

                default:
                    state <= S_IDLE;
            endcase
        end
    end

    // ---------------------------------------------------- address generation --
    assign tok_rd_addr = token_base  + token_idx;
    assign w_rd_addr   = weight_base + w_row;
    assign res_wr_addr = result_base + token_idx;

    // ------------------------------------------------------------- strobes --
    assign arr_sel       = array_id;
    assign arr_w_we      = (state == S_LOAD_WEIGHT);
    assign arr_w_row     = w_row;
    assign arr_x_load    = (state == S_LOAD_TOKEN);
    assign arr_mvm_start = (state == S_START_MVM);
    assign res_we        = (state == S_STORE_RESULT);

    assign busy = (state != S_IDLE) && (state != S_DONE);
    assign done = (state == S_DONE);

endmodule
