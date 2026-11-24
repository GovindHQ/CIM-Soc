// =============================================================================
// command_defs.vh -- the command format the CPU writes into the command table.
//
// Every command is one CMD_W-bit word. Every field sits at a fixed position
// whatever the opcode; a field a command does not use is simply 0.
//
//   op            meaning                               fields used
//   ------------  ------------------------------------  ----------------------
//   CMD_END       run finished, raise done              -
//   CMD_CONFIG    set analog domain for what follows    gain, act_signed
//   CMD_LOAD_WGT  write one weight tile into one array  array_first, weight_addr
//   CMD_CLEAR     zero 'count' consecutive result words output_addr, count
//   CMD_COMPUTE   stream 'count' tokens through the     array_first, array_count,
//                 resident weights:                     input_addr, input_stride,
//                 input -> DAC -> array -> ADC          in_step, output_addr,
//                 -> accumulate                         output_stride, out_step,
//                                                       count
//
// CMD_COMPUTE addresses, for token t = 0..count-1 and array a = 0..array_count-1:
//
//   input  word  = input_addr  + t*input_stride  + (in_step  ? a : 0)
//   result word  = output_addr + t*output_stride + (out_step ? a : 0)
//
//   in_step=1, out_step=0 : arrays hold different DEPTH blocks of one output
//                           tile; their codes add into the same result word.
//   in_step=0, out_step=1 : arrays hold different OUTPUT tiles of one depth
//                           block; one input word goes to all of them, and each
//                           accumulates into its own result word.
//
// The controller knows nothing about attention, linear layers or convolution.
// What the words mean is decided by whoever wrote the table.
// =============================================================================
`ifndef COMMAND_DEFS_VH
`define COMMAND_DEFS_VH

`define CMD_W          120

`define CMD_END        4'd0
`define CMD_CONFIG     4'd1
`define CMD_LOAD_WGT   4'd2
`define CMD_CLEAR      4'd3
`define CMD_COMPUTE    4'd4

`define F_OP            3:0
`define F_ARRAY_FIRST  11:4
`define F_ARRAY_COUNT  19:12
`define F_WEIGHT_ADDR  35:20
`define F_INPUT_ADDR   51:36
`define F_INPUT_STRIDE 63:52
`define F_OUTPUT_ADDR  79:64
`define F_OUTPUT_STRIDE 91:80
`define F_COUNT        107:92
`define F_IN_STEP      108
`define F_OUT_STEP     109
`define F_ACT_SIGNED   110
`define F_GAIN         118:111
// bit 119 reserved

`endif
