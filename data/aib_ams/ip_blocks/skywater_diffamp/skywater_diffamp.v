`timescale 1ps/1ps 


module skywater_diffamp__w_sup(
    input  wire v_inn,
    input  wire v_inp,
    output wire v_out,
    inout  wire VDD,
    inout  wire VSS
);

parameter DELAY = 20;
logic temp;

always_comb begin
    casez ({v_inp, v_inn, VDD, VSS})
        4'b1010: temp = 1'b1;
        4'b0110: temp = 1'b0;
        4'b??00: temp = 1'b0;
        default: temp = 1'bx;
    endcase
end

assign #DELAY v_out = temp;

endmodule


module skywater_diffamp(
    input  wire v_inn,
    input  wire v_inp,
    output wire v_out
);

wire VDD_val;
wire VSS_val;

assign VDD_val = 1'b1;
assign VSS_val = 1'b0;

skywater_diffamp__w_sup XDUT (
    .v_inn( v_inn ),
    .v_inp( v_inp ),
    .v_out( v_out ),
    .VDD( VDD_val ),
    .VSS( VSS_val )
);

endmodule
