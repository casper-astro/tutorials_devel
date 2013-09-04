module traffic_lights
    #(parameter AMBER_TIME = 32'd10)
    (input clk,
    input ce,
    input reset,
    input toggle,
    //Give your output registers initial values to avoid Simulink
    //"undetermined output" errors
    output reg green_led = 1'b0,
    output reg amber_led = 1'b0,
    output reg red_led = 1'b0
    );


    // State encoding
    localparam STATE_RED = 2'd0;
    localparam STATE_GREEN = 2'd1;
    localparam STATE_GOING_RED = 2'd2;
    localparam STATE_GOING_GREEN = 2'd3;
    
    // State register
    reg [1:0] state = 2'b0;

    // Register for amber timer counter
    reg [31:0] amber_timer = 32'b0;
    
    always @(posedge clk) begin
        if (reset == 1'b1) begin
            state <= STATE_RED;
        end else begin
            green_led <= 1'b0;
            red_led   <= 1'b0;
            amber_led <= 1'b0;
            case(state)
                STATE_RED: begin
                    red_led <= 1'b1;
                    if (toggle) begin
                        state <= STATE_GOING_GREEN;
                    end else begin
                        state <= STATE_RED;
                    end
                end
                STATE_GREEN: begin
                    green_led <= 1'b1;
                    if (toggle) begin
                        state <= STATE_GOING_RED;
                    end else begin
                        state <= STATE_GREEN;
                    end
                end
                STATE_GOING_RED: begin
                    amber_led <= 1;
                    amber_timer <= amber_timer + 1;
                    if (amber_timer == (AMBER_TIME-1)) begin
                        amber_timer <= 32'b0;
                        state <= STATE_RED;
                    end else begin
                        state <= STATE_GOING_RED;
                    end
                end
                STATE_GOING_GREEN: begin
                    amber_led <= 1;
                    red_led   <= 1;
                    amber_timer <= amber_timer + 1;
                    if (amber_timer == (AMBER_TIME-1)) begin
                        amber_timer <= 32'b0;
                        state <= STATE_GREEN;
                    end else begin
                        state <= STATE_GOING_GREEN;
                    end
                end
                default: begin
                    state <= STATE_RED;
                end
            endcase
        end
    end

endmodule

