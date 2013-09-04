module gpio_bidir(
    input    	     clk,
    inout      [7:0] dio_buf, //inout, NOT input(!)
    input      [7:0] din_i,
    output reg [7:0] dout_o,
    input            in_not_out_i
  );
  
  // A wire for the output data stream
  wire [7:0] dout_wire;

  // Buffer the in-out data line
  IOBUF iob_data[7:0] (
    .O (dout_wire),  //the data output
    .IO(dio_buf),    //the external in-out signal
    .I(din_i),       //the data input
    .T(in_not_out_i) //The control signal. 1 for input, 0 for output
  ); 
 
  //register the data output
  always @(posedge clk) begin
    dout_o <= dout_wire;
  end
  
endmodule
