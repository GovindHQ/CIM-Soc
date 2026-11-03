// =============================================================================
// cim_top.v -- wires the controller, the three memories and NUM_ARRAYS arrays.
//
//   host --> token_buffer  --tok_rd_data--> cim_array[*].x_data
//   host --> weight_memory --w_rd_data----> cim_array[*].w_data
//   cim_array[arr_sel].y -----------------> result_buffer --> host
//   array_controller drives every address and strobe.
//
// NUM_ARRAYS: the generate loop below instantiates that many arrays. Today
// only the array selected by cfg_array_id (arr_sel) receives strobes and has
// its result stored; the others stay idle.
// =============================================================================
module cim_top #(
    parameter NUM_ARRAYS  = 1,
    parameter ROWS        = 32,
    parameter COLS        = 32,
    parameter ACT_BITS    = 8,
    parameter W_BITS      = 8,
    parameter ACC_W       = 32,
    parameter AW          = 8,
    parameter TOK_DEPTH   = 16,
    parameter W_DEPTH     = 64,
    parameter RES_DEPTH   = 16,
    parameter MVM_LATENCY = 10,
    // derived -- do not override
    parameter AID_W = (NUM_ARRAYS > 1) ? $clog2(NUM_ARRAYS) : 1,
    parameter X_W   = ROWS * ACT_BITS,    // one token
    parameter WR_W  = COLS * W_BITS,      // one weight row
    parameter Y_W   = COLS * ACC_W        // one MVM result
) (
    input  wire             clk,
    input  wire             rst_n,

    // host: fill token buffer
    input  wire             tok_wr_en,
    input  wire [AW-1:0]    tok_wr_addr,
    input  wire [X_W-1:0]   tok_wr_data,

    // host: fill weight memory (one row per write)
    input  wire             w_wr_en,
    input  wire [AW-1:0]    w_wr_addr,
    input  wire [WR_W-1:0]  w_wr_data,

    // host: configuration + start / status
    input  wire             start,
    input  wire [AW-1:0]    cfg_token_base,
    input  wire [AW-1:0]    cfg_weight_base,
    input  wire [AW-1:0]    cfg_result_base,
    input  wire [AW-1:0]    cfg_num_tokens,
    input  wire [AID_W-1:0] cfg_array_id,
    output wire             busy,
    output wire             done,

    // host: read results
    input  wire [AW-1:0]    res_rd_addr,
    output wire [Y_W-1:0]   res_rd_data
);

    localparam RIDX_W = (ROWS > 1) ? $clog2(ROWS) : 1;

    // controller <-> datapath
    wire [AW-1:0]     tok_rd_addr, w_rd_addr, res_wr_addr;
    wire              res_we;
    wire [AID_W-1:0]  arr_sel;
    wire              arr_w_we, arr_x_load, arr_mvm_start;
    wire [RIDX_W-1:0] arr_w_row;

    wire [X_W-1:0]    tok_rd_data;
    wire [WR_W-1:0]   w_rd_data;
    wire [NUM_ARRAYS*Y_W-1:0] y_all;      // results of every array, packed
    wire [Y_W-1:0]    y_sel = y_all[arr_sel*Y_W +: Y_W];

    // ------------------------------------------------------------ control --
    array_controller #(
        .ROWS(ROWS), .NUM_ARRAYS(NUM_ARRAYS), .AW(AW), .MVM_LATENCY(MVM_LATENCY)
    ) u_ctrl (
        .clk(clk), .rst_n(rst_n),
        .start(start),
        .cfg_token_base(cfg_token_base), .cfg_weight_base(cfg_weight_base),
        .cfg_result_base(cfg_result_base), .cfg_num_tokens(cfg_num_tokens),
        .cfg_array_id(cfg_array_id),
        .busy(busy), .done(done),
        .tok_rd_addr(tok_rd_addr), .w_rd_addr(w_rd_addr),
        .res_we(res_we), .res_wr_addr(res_wr_addr),
        .arr_sel(arr_sel), .arr_w_we(arr_w_we), .arr_w_row(arr_w_row),
        .arr_x_load(arr_x_load), .arr_mvm_start(arr_mvm_start)
    );

    // ----------------------------------------------------------- memories --
    token_buffer #(.ROWS(ROWS), .ACT_BITS(ACT_BITS), .DEPTH(TOK_DEPTH), .AW(AW)) u_tok (
        .clk(clk),
        .wr_en(tok_wr_en), .wr_addr(tok_wr_addr), .wr_data(tok_wr_data),
        .rd_addr(tok_rd_addr), .rd_data(tok_rd_data)
    );

    weight_memory #(.COLS(COLS), .W_BITS(W_BITS), .DEPTH(W_DEPTH), .AW(AW)) u_wmem (
        .clk(clk),
        .wr_en(w_wr_en), .wr_addr(w_wr_addr), .wr_data(w_wr_data),
        .rd_addr(w_rd_addr), .rd_data(w_rd_data)
    );

    result_buffer #(.COLS(COLS), .ACC_W(ACC_W), .DEPTH(RES_DEPTH), .AW(AW)) u_res (
        .clk(clk),
        .wr_en(res_we), .wr_addr(res_wr_addr), .wr_data(y_sel),
        .rd_addr(res_rd_addr), .rd_data(res_rd_data)
    );

    // ---------------------------------------------------- CIM array(s) --
    genvar g;
    generate
        for (g = 0; g < NUM_ARRAYS; g = g + 1) begin : g_arr
            cim_array #(
                .ROWS(ROWS), .COLS(COLS), .ACT_BITS(ACT_BITS),
                .W_BITS(W_BITS), .ACC_W(ACC_W)
            ) u_arr (
                .clk(clk),
                .w_we(arr_w_we && (arr_sel == g)),
                .w_row(arr_w_row),
                .w_data(w_rd_data),
                .x_load(arr_x_load && (arr_sel == g)),
                .x_data(tok_rd_data),
                .mvm_start(arr_mvm_start && (arr_sel == g)),
                .y(y_all[g*Y_W +: Y_W])
            );
        end
    endgenerate

endmodule
