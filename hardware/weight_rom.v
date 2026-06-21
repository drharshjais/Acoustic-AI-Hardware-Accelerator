module weight_rom(
    input clk,
    input [13:0] addr,
    output reg signed [7:0] dout
);
    reg signed [7:0] rom [0:10047];
    
    // Load the quantized AI weights from Python
    initial $readmemh("weights.mem", rom);
    
    always @(posedge clk) begin
        dout <= rom[addr];
    end
endmodule