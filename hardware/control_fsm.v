module control_fsm(
    input clk,
    input rst,
    input start,
    output reg [13:0] mem_addr,
    output reg mac_en,
    output reg done
);
    reg [1:0] state;
    parameter IDLE = 2'b00, COMPUTE = 2'b01, FINISH = 2'b10;

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            mem_addr <= 0;
            mac_en <= 0;
            done <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    mem_addr <= 0;
                    if (start) begin
                        state <= COMPUTE;
                        mac_en <= 1;
                    end
                end
                COMPUTE: begin
                    if (mem_addr == 14'd10047) begin
                        state <= FINISH;
                        mac_en <= 0;
                    end else begin
                        mem_addr <= mem_addr + 1;
                    end
                end
                FINISH: begin
                    done <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule