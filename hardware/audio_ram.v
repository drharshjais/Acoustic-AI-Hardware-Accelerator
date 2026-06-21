module audio_ram(
    input clk,
    input we,
    input [13:0] addr,
    input signed [7:0] din,
    output reg signed [7:0] dout
);
    reg signed [7:0] ram [0:10047];
    integer i;
    
    // Prevent Poison X bug during simulation
    initial begin 
        for (i=0; i<10048; i=i+1) ram[i] = 0; 
    end
    
    always @(posedge clk) begin
        if (we) ram[addr] <= din;
        dout <= ram[addr];
    end
endmodule