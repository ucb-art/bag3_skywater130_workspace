`timescale 1ps/1ps 


module skywater_nand2__w_sup(
    input  wire [1:0] in,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

parameter DELAY = 20;

assign #DELAY out = VSS ? 1'bx : (~VDD ? 1'b0 : ~&in );

endmodule


module skywater_nand2(
    input  wire [1:0] in,
    output wire out
);

wire VDD_val;
wire VSS_val;

assign VDD_val = 1'b1;
assign VSS_val = 1'b0;

skywater_nand2__w_sup XDUT (
    .in( in ),
    .out( out ),
    .VDD( VDD_val ),
    .VSS( VSS_val )
);

endmodule
