module top_level_accelerator(
    input clk,
    input rst,
    input start,
    input ext_audio_we,
    input [13:0] ext_audio_addr,
    input signed [7:0] ext_audio_data,
    output done,
    output signed [31:0] final_prediction,
    output threat_detected
);
    wire [13:0] fsm_addr;
    wire mac_en;
    wire signed [7:0] audio_val;
    wire signed [7:0] weight_val;
    
    // Multiplexer to allow external sensor to write to RAM
    wire [13:0] ram_addr = ext_audio_we ? ext_audio_addr : fsm_addr;

    control_fsm fsm_inst (
        .clk(clk), .rst(rst), .start(start), 
        .mem_addr(fsm_addr), .mac_en(mac_en), .done(done)
    );

    audio_ram ram_inst (
        .clk(clk), .we(ext_audio_we), .addr(ram_addr), 
        .din(ext_audio_data), .dout(audio_val)
    );

    weight_rom rom_inst (
        .clk(clk), .addr(fsm_addr), .dout(weight_val)
    );

    mac_unit mac_inst (
        .clk(clk), .rst(rst), .en(mac_en), 
        .audio_pixel(audio_val), .weight(weight_val), .accumulator(final_prediction)
    );

    // The Hardware Alarm Threshold (Trigger LoRa transmission)
    assign threat_detected = (final_prediction > 280) ? 1'b1 : 1'b0;

endmodule