// =============================================================================
// tb_control.v -- plays the CPU: load, START, wait, check.
//
//   1. reset
//   2. write the command table, the token buffer and the weight memory
//   3. pulse START
//   4. wait for DONE
//   5. read every result word and compare with the golden values
//   6. compare cycle count and activity counters with what the mapper predicted
//   7. check nothing was written just outside the result range
//
// Everything specific to a test comes from files in the current directory,
// written by run_tests.py (the CPU's mapper):
//
//   tb_params.vh  parameters and expected counters
//   cmd.hex  input.hex  weight.hex  expected.hex
// =============================================================================
// No `timescale: time is counted in clock cycles only.
`include "command_defs.vh"

module tb_control;

    `include "tb_params.vh"

    localparam ROWS = 32, COLS = 32, ACT_BITS = 8, W_BITS = 8, ADC_BITS = 8, ACC_W = 32, AW = 16;
    localparam IN_W = ROWS*ACT_BITS, WR_W = COLS*W_BITS, RES_W = COLS*ACC_W;

    // what the CPU will load, as read from the files
    reg [`CMD_W-1:0] cmd_img      [0:N_CMD-1];
    reg [IN_W-1:0]   input_img    [0:N_INPUT-1];
    reg [WR_W-1:0]   weight_img   [0:N_WEIGHT-1];
    reg [RES_W-1:0]  expected_img [0:N_EXPECTED-1];

    // ---------------------------------------------------------------- DUT --
    reg               clk = 0, rst_n = 0, start = 0;
    reg               cmd_wr_en = 0, input_wr_en = 0, weight_wr_en = 0;
    reg  [AW-1:0]     cmd_wr_addr = 0, input_wr_addr = 0, weight_wr_addr = 0;
    reg  [`CMD_W-1:0] cmd_wr_data = 0;
    reg  [IN_W-1:0]   input_wr_data = 0;
    reg  [WR_W-1:0]   weight_wr_data = 0;
    wire              busy, done, error;
    wire [31:0]       adc_clip_count;
    reg  [AW-1:0]     result_rd_addr = 0;
    wire [RES_W-1:0]  result_rd_data;

    cim_top #(
        .NUM_ARRAYS(NUM_ARRAYS),
        .DAC_LATENCY(DAC_LATENCY), .ARRAY_LATENCY(ARRAY_LATENCY),
        .ADC_LATENCY(ADC_LATENCY), .ARRAY_ACCUM_WAIT(ARRAY_ACCUM_WAIT)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .cmd_wr_en(cmd_wr_en), .cmd_wr_addr(cmd_wr_addr), .cmd_wr_data(cmd_wr_data),
        .input_wr_en(input_wr_en), .input_wr_addr(input_wr_addr), .input_wr_data(input_wr_data),
        .weight_wr_en(weight_wr_en), .weight_wr_addr(weight_wr_addr), .weight_wr_data(weight_wr_data),
        .start(start), .busy(busy), .done(done), .error(error),
        .adc_clip_count(adc_clip_count),
        .result_rd_addr(result_rd_addr), .result_rd_data(result_rd_data)
    );

    always #5 clk = ~clk;

    // ---------------------------------------------------- activity counters --
    integer n_weight_rows = 0, n_array_ops = 0, n_timesteps = 0, n_adc_groups = 0, g;
    always @(posedge clk) begin
        if (dut.weight_we) n_weight_rows = n_weight_rows + 1;
        if (dut.array_start) begin
            n_timesteps = n_timesteps + 1;
            for (g = 0; g < NUM_ARRAYS; g = g + 1)
                if (dut.group_mask[g]) n_array_ops = n_array_ops + 1;
        end
    end

    // --------------------------------------------- readable trace (small tests) --
    function [8*14-1:0] state_name(input [3:0] s);
        case (s)
            4'd0:  state_name = "IDLE";
            4'd1:  state_name = "FETCH";
            4'd2:  state_name = "DECODE";
            4'd3:  state_name = "LOAD_WEIGHT";
            4'd4:  state_name = "CLEAR_OUTPUT";
            4'd5:  state_name = "LOAD_INPUT";
            4'd6:  state_name = "DAC_START";
            4'd7:  state_name = "WAIT_DAC";
            4'd8:  state_name = "ARRAY_START";
            4'd9:  state_name = "WAIT_ARRAY";
            4'd10: state_name = "ADC_START";
            4'd11: state_name = "WAIT_ADC";
            4'd12: state_name = "WAIT_ACCUM";
            4'd13: state_name = "ACCUMULATE";
            4'd14: state_name = "DONE";
            default: state_name = "???";
        endcase
    endfunction

    integer cycle = 0;
    reg [3:0] prev_state = 0;
    always @(posedge clk) begin
        cycle <= cycle + 1;
        if (VERBOSE && dut.u_ctrl.state != prev_state)
            $display("  cycle %6d  %0s", cycle, state_name(dut.u_ctrl.state));
        prev_state <= dut.u_ctrl.state;
        if (VERBOSE && dut.result_accum)
            $display("                accumulate: array %0d code[0]=%0d  result word %0d col 0: %0d -> %0d",
                     dut.active_array,
                     $signed(dut.code_active[ADC_BITS-1:0]),
                     dut.result_addr,
                     $signed(dut.u_res.current[ACC_W-1:0]),
                     $signed(dut.u_res.sum[ACC_W-1:0]));
    end

    // -------------------------------------------------------------- test ---
    integer i, c, errors, shown, t_start, t_done, got, want;

    initial begin
        errors = 0;
        shown  = 0;
        $readmemh("cmd.hex",      cmd_img);
        $readmemh("input.hex",    input_img);
        $readmemh("weight.hex",   weight_img);
        $readmemh("expected.hex", expected_img);

        $display("==== %0s: NUM_ARRAYS=%0d DAC_LATENCY=%0d ARRAY_LATENCY=%0d ADC_LATENCY=%0d ARRAY_ACCUM_WAIT=%0d commands=%0d",
                 TEST_NAME, NUM_ARRAYS, DAC_LATENCY, ARRAY_LATENCY, ADC_LATENCY, ARRAY_ACCUM_WAIT, N_CMD);

        // 1. reset
        repeat (3) @(negedge clk);
        rst_n = 1;

        // 2. the CPU fills the memories
        cmd_wr_en = 1;
        for (i = 0; i < N_CMD; i = i + 1) begin
            cmd_wr_addr = i; cmd_wr_data = cmd_img[i];
            @(negedge clk);
        end
        cmd_wr_en = 0;

        input_wr_en = 1;
        for (i = 0; i < N_INPUT; i = i + 1) begin
            input_wr_addr = i; input_wr_data = input_img[i];
            @(negedge clk);
        end
        input_wr_en = 0;

        weight_wr_en = 1;
        for (i = 0; i < N_WEIGHT; i = i + 1) begin
            weight_wr_addr = i; weight_wr_data = weight_img[i];
            @(negedge clk);
        end
        weight_wr_en = 0;

        // 3. START (one cycle)
        start = 1;
        @(negedge clk);
        start = 0;
        t_start = cycle;

        // 4. wait for DONE (with timeout)
        fork : wait_done
            begin @(posedge done); disable wait_done; end
            begin repeat (2*EXP_CYCLES + 1000) @(posedge clk);
                  $display("ERROR: timeout waiting for done"); errors = errors + 1;
                  disable wait_done; end
        join
        @(negedge clk);
        t_done = cycle;

        if (error !== (EXP_ERROR != 0)) begin
            $display("ERROR: controller error flag = %b, expected %0d", error, EXP_ERROR);
            errors = errors + 1;
        end

        // 5. results
        for (i = 0; i < N_EXPECTED; i = i + 1) begin
            result_rd_addr = OUTPUT_BASE + i;
            #1;
            for (c = 0; c < COLS; c = c + 1) begin
                got  = $signed(result_rd_data[c*ACC_W +: ACC_W]);
                want = $signed(expected_img[i][c*ACC_W +: ACC_W]);
                if (result_rd_data[c*ACC_W +: ACC_W] !== expected_img[i][c*ACC_W +: ACC_W]) begin
                    errors = errors + 1;
                    if (shown < 10) begin
                        $display("MISMATCH result word %0d col %0d: got %0d expected %0d",
                                 OUTPUT_BASE + i, c, got, want);
                        shown = shown + 1;
                    end
                end
            end
            if (VERBOSE) begin
                $display("  result word %0d, col 0..3: %0d %0d %0d %0d  (expected %0d %0d %0d %0d)",
                         OUTPUT_BASE + i,
                         $signed(result_rd_data[0*ACC_W +: ACC_W]),
                         $signed(result_rd_data[1*ACC_W +: ACC_W]),
                         $signed(result_rd_data[2*ACC_W +: ACC_W]),
                         $signed(result_rd_data[3*ACC_W +: ACC_W]),
                         $signed(expected_img[i][0*ACC_W +: ACC_W]),
                         $signed(expected_img[i][1*ACC_W +: ACC_W]),
                         $signed(expected_img[i][2*ACC_W +: ACC_W]),
                         $signed(expected_img[i][3*ACC_W +: ACC_W]));
            end
        end

        // 6. counters
        `define CHECK(name, got_v, exp_v) \
            $display("  %-16s %0d (expected %0d)", name, got_v, exp_v); \
            if ((got_v) !== (exp_v)) begin $display("ERROR: %0s", name); errors = errors + 1; end
        `CHECK("cycles",        t_done - t_start,   EXP_CYCLES)
        `CHECK("tile writes",   n_weight_rows/ROWS, EXP_TILE_WRITES)
        `CHECK("array ops",     n_array_ops,        EXP_ARRAY_OPS)
        `CHECK("timesteps",     n_timesteps,        EXP_TIMESTEPS)
        `CHECK("ADC clipped",   adc_clip_count,     EXP_CLIPS)

        // 7. nothing written outside the result range (unwritten memory reads X)
        result_rd_addr = OUTPUT_BASE - 1;
        #1;
        if (^result_rd_data !== 1'bx) begin
            $display("ERROR: result word %0d, before the range, was written", OUTPUT_BASE - 1);
            errors = errors + 1;
        end
        result_rd_addr = OUTPUT_BASE + N_EXPECTED;
        #1;
        if (^result_rd_data !== 1'bx) begin
            $display("ERROR: result word %0d, after the range, was written", OUTPUT_BASE + N_EXPECTED);
            errors = errors + 1;
        end

        if (errors == 0) $display("RESULT: PASS");
        else             $display("RESULT: FAIL (%0d errors)", errors);
        $finish;
    end

endmodule
