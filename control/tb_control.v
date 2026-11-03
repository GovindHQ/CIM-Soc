// =============================================================================
// tb_control.v -- N tokens, ONE weight matrix, ONE array, N MVMs, N results.
//
//   1. reset
//   2. write NUM_TOKENS tokens into the token buffer (+ decoys around them)
//   3. write one 32x32 weight matrix                 (+ a decoy tile)
//   4. set cfg_* (token/weight/result base, number of tokens, array id)
//   5. pulse START
//   6. wait for DONE
//   7. read every result and compare with an independent golden loop;
//      check that nothing was written before / after the result range
//
// NUM_TOKENS = 1 is the original one-token demonstration.
// To change the test data, edit tok_val() and w_val() below.
// =============================================================================
// No `timescale: time is counted in clock cycles only, and leaving it out keeps
// XSim from demanding one in every RTL file.
module tb_control;

    parameter NUM_ARRAYS  = 1;
    parameter NUM_TOKENS  = 4;            // 1 .. 13 with the buffer depths below
    parameter MVM_LATENCY = 10;

    localparam ROWS = 32, COLS = 32, ACT_BITS = 8, W_BITS = 8, ACC_W = 32, AW = 8;
    localparam AID_W = (NUM_ARRAYS > 1) ? $clog2(NUM_ARRAYS) : 1;

    // where the test puts things (non-zero on purpose, to exercise addressing)
    localparam TOKEN_BASE  = 1;           // tokens at 1 .. NUM_TOKENS
    localparam WEIGHT_BASE = 32;          // second 32-row tile
    localparam RESULT_BASE = 2;           // results at 2 .. NUM_TOKENS+1
    localparam ARRAY_ID    = NUM_ARRAYS - 1;

    // expected start -> done: weights once, then (LOAD_TOKEN, START, WAIT, STORE) per token
    localparam EXP_CYCLES  = ROWS + NUM_TOKENS * (MVM_LATENCY + 3);

    // -------------------------------------------------------- test data ----
    // token t: x_t[r] = (r + 1) - 8t        (t = 0: 1, 2, ..., 32)
    function integer tok_val(input integer t, input integer r);
        tok_val = (r + 1) - 8 * t;
    endfunction

    // W[r][c] = +2 on the diagonal, -1 on the anti-diagonal, else 0
    //   => y_t[c] = 2*x_t[c] - x_t[31-c] = 3c - 30 - 8t
    function integer w_val(input integer r, input integer c);
        if (r == c)                w_val = 2;
        else if (r == ROWS-1 - c)  w_val = -1;
        else                       w_val = 0;
    endfunction

    // ---------------------------------------------------------------- DUT --
    reg                      clk = 0, rst_n = 0;
    reg                      tok_wr_en = 0;
    reg  [AW-1:0]            tok_wr_addr = 0;
    reg  [ROWS*ACT_BITS-1:0] tok_wr_data = 0;
    reg                      w_wr_en = 0;
    reg  [AW-1:0]            w_wr_addr = 0;
    reg  [COLS*W_BITS-1:0]   w_wr_data = 0;
    reg                      start = 0;
    reg  [AW-1:0]            cfg_token_base = 0, cfg_weight_base = 0, cfg_result_base = 0;
    reg  [AW-1:0]            cfg_num_tokens = 0;
    reg  [AID_W-1:0]         cfg_array_id = 0;
    wire                     busy, done;
    reg  [AW-1:0]            res_rd_addr = 0;
    wire [COLS*ACC_W-1:0]    res_rd_data;

    cim_top #(.NUM_ARRAYS(NUM_ARRAYS), .MVM_LATENCY(MVM_LATENCY)) dut (
        .clk(clk), .rst_n(rst_n),
        .tok_wr_en(tok_wr_en), .tok_wr_addr(tok_wr_addr), .tok_wr_data(tok_wr_data),
        .w_wr_en(w_wr_en), .w_wr_addr(w_wr_addr), .w_wr_data(w_wr_data),
        .start(start),
        .cfg_token_base(cfg_token_base), .cfg_weight_base(cfg_weight_base),
        .cfg_result_base(cfg_result_base), .cfg_num_tokens(cfg_num_tokens),
        .cfg_array_id(cfg_array_id),
        .busy(busy), .done(done),
        .res_rd_addr(res_rd_addr), .res_rd_data(res_rd_data)
    );

    always #5 clk = ~clk;

    // ------------------------------------------------------ FSM trace ------
    integer cycle = 0;
    always @(posedge clk) cycle <= cycle + 1;

    function [8*12-1:0] state_name(input [2:0] s);
        case (s)
            3'd0: state_name = "IDLE";
            3'd1: state_name = "LOAD_WEIGHT";
            3'd2: state_name = "LOAD_TOKEN";
            3'd3: state_name = "START_MVM";
            3'd4: state_name = "WAIT_MVM";
            3'd5: state_name = "STORE_RESULT";
            3'd6: state_name = "DONE";
            default: state_name = "???";
        endcase
    endfunction

    reg [2:0] prev_state = 3'd0;
    always @(posedge clk) begin
        if (dut.u_ctrl.state != prev_state)
            $display("  cycle %4d  %-12s -> %-12s  token_idx=%0d", cycle,
                     state_name(prev_state), state_name(dut.u_ctrl.state),
                     dut.u_ctrl.token_idx);
        prev_state <= dut.u_ctrl.state;
    end

    // ---------------------------------------------------- host tasks -------
    // decoy = 1 writes a constant 7 instead of real token data
    task write_token(input integer addr, input integer t, input integer decoy);
        integer r;
        begin
            @(negedge clk);
            tok_wr_en = 1; tok_wr_addr = addr;
            for (r = 0; r < ROWS; r = r + 1)
                tok_wr_data[r*ACT_BITS +: ACT_BITS] = decoy ? 7 : tok_val(t, r);
            @(negedge clk);
            tok_wr_en = 0;
        end
    endtask

    task write_weights(input integer base, input integer decoy);
        integer r, c;
        begin
            for (r = 0; r < ROWS; r = r + 1) begin
                @(negedge clk);
                w_wr_en = 1; w_wr_addr = base + r;
                for (c = 0; c < COLS; c = c + 1)
                    w_wr_data[c*W_BITS +: W_BITS] = decoy ? 1 : w_val(r, c);
            end
            @(negedge clk);
            w_wr_en = 0;
        end
    endtask

    // -------------------------------------------------------------- test ---
    integer t, r, c, xv, wv, exp_y, got_y, errors, tok_errors, t_start, t_done;

    initial begin
        errors = 0;
        $display("==== tb_control: NUM_TOKENS=%0d NUM_ARRAYS=%0d ARRAY_ID=%0d MVM_LATENCY=%0d ROWS=%0d COLS=%0d",
                 NUM_TOKENS, NUM_ARRAYS, ARRAY_ID, MVM_LATENCY, ROWS, COLS);
        $display("     token_base=%0d weight_base=%0d result_base=%0d",
                 TOKEN_BASE, WEIGHT_BASE, RESULT_BASE);

        // 1. reset
        repeat (3) @(negedge clk);
        rst_n = 1;

        // 2-3. data, with decoys just before and just after the token range
        write_token(TOKEN_BASE - 1, 0, 1);
        for (t = 0; t < NUM_TOKENS; t = t + 1)
            write_token(TOKEN_BASE + t, t, 0);
        write_token(TOKEN_BASE + NUM_TOKENS, 0, 1);
        write_weights(0, 1);
        write_weights(WEIGHT_BASE, 0);

        // 4. configure
        @(negedge clk);
        cfg_token_base  = TOKEN_BASE;
        cfg_weight_base = WEIGHT_BASE;
        cfg_result_base = RESULT_BASE;
        cfg_num_tokens  = NUM_TOKENS;
        cfg_array_id    = ARRAY_ID;

        // 5. START (one cycle)
        $display("---- START");
        start = 1;
        @(negedge clk);
        start = 0;
        t_start = cycle;

        // 6. wait for DONE (with timeout)
        fork : wait_done
            begin @(posedge done); disable wait_done; end
            begin repeat (10000) @(posedge clk);
                  $display("ERROR: timeout waiting for done"); errors = errors + 1;
                  disable wait_done; end
        join
        @(negedge clk);
        t_done = cycle;
        $display("---- DONE after %0d cycles (expected ROWS + NUM_TOKENS*(MVM_LATENCY+3) = %0d)",
                 t_done - t_start, EXP_CYCLES);
        if (t_done - t_start != EXP_CYCLES) begin
            $display("ERROR: unexpected latency"); errors = errors + 1;
        end

        // 7. read and check every token's result
        @(posedge clk);                 // let the trace print the last transition
        for (t = 0; t < NUM_TOKENS; t = t + 1) begin
            res_rd_addr = RESULT_BASE + t;
            #1;
            if (t == 0) $display("   token 0:  col |   got | expected");
            tok_errors = 0;
            for (c = 0; c < COLS; c = c + 1) begin
                exp_y = 0;
                for (r = 0; r < ROWS; r = r + 1) begin
                    xv = tok_val(t, r);
                    wv = w_val(r, c);
                    exp_y = exp_y + xv * wv;
                end
                got_y = $signed(res_rd_data[c*ACC_W +: ACC_W]);
                if (t == 0)
                    $display("             %3d | %5d | %5d %s", c, got_y, exp_y,
                             (got_y === exp_y) ? "" : "<-- MISMATCH");
                else if (got_y !== exp_y)
                    $display("   token %0d col %0d: got %0d expected %0d  <-- MISMATCH",
                             t, c, got_y, exp_y);
                if (got_y !== exp_y) tok_errors = tok_errors + 1;
            end
            $display("   token %2d @ result addr %2d: %0d/%0d columns match  (y[0]=%0d, y[31]=%0d)",
                     t, RESULT_BASE + t, COLS - tok_errors, COLS,
                     $signed(res_rd_data[0 +: ACC_W]),
                     $signed(res_rd_data[(COLS-1)*ACC_W +: ACC_W]));
            errors = errors + tok_errors;
        end

        // nothing written outside the result range (unwritten memory reads X)
        res_rd_addr = RESULT_BASE - 1;
        #1;
        if (^res_rd_data !== 1'bx) begin
            $display("ERROR: result addr %0d (before range) was written", RESULT_BASE - 1);
            errors = errors + 1;
        end
        res_rd_addr = RESULT_BASE + NUM_TOKENS;
        #1;
        if (^res_rd_data !== 1'bx) begin
            $display("ERROR: result addr %0d (after range) was written -- extra token?",
                     RESULT_BASE + NUM_TOKENS);
            errors = errors + 1;
        end

        if (errors == 0) $display("RESULT: PASS");
        else             $display("RESULT: FAIL (%0d errors)", errors);
        $finish;
    end

endmodule
