`timescale 1ns / 1ps

module traffic_lights_tb;

	// Inputs
	reg clk;
    reg ce;
	reg toggle;

	// Outputs
	wire green_led;
	wire red_led;
	wire amber_led;

    // parameters
    localparam AMBER_TIME = 32'd10;

    // Instantialte the Unit Under Test
    traffic_lights #(
        .AMBER_TIME(AMBER_TIME)
    ) uut (
        .clk(clk),
        .ce(ce),
        .toggle(toggle),
        .green_led(green_led),
        .red_led(red_led),
        .amber_led(amber_led)
    );

	initial begin
		// Initialize Inputs
		clk = 1'b0;
        ce  = 1'b1;
		toggle = 1'b0;

		// Wait 100 ns for global reset to finish
		#100;

        // Add stimulus here
        $display("toggling!");
        toggle = 1'b1;
        #20 toggle = 1'b0;

        #500;

        $display("toggling!");
        toggle = 1'b1;
        #20 toggle = 1'b0;

        #500;
        $finish;

	end
    
    initial begin
        forever #10 clk = !clk;        
    end

    // Display the states every clock cycle
    reg [31:0] ctr = 32'b0;
    always @(posedge clk) begin
        ctr <= ctr + 1'b1;
        $display("CLOCK: %d, {r,a,g} = {%d,%d,%d}", ctr, red_led, amber_led, green_led);
    end
    


endmodule

