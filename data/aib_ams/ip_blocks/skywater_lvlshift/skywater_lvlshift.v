`timescale 1ps/1ps 


module skywater_lvlshift_inv_3__w_sup(
    input  wire in,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

parameter DELAY = 0;

assign #DELAY out = VSS ? 1'bx : (~VDD ? 1'b0 : ~in );

endmodule


module skywater_lvlshift_inv_4__w_sup(
    input  wire in,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

parameter DELAY = 0;

assign #DELAY out = VSS ? 1'bx : (~VDD ? 1'b0 : ~in );

endmodule


module skywater_lvlshift_inv_chain_2__w_sup(
    input  wire in,
    output wire out,
    output wire outb,
    inout  wire VDD,
    inout  wire VSS
);

skywater_lvlshift_inv_3__w_sup XINV0 (
    .in( in ),
    .out( outb ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_lvlshift_inv_4__w_sup XINV1 (
    .in( outb ),
    .out( out ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_lvlshift_inv_5__w_sup(
    input  wire in,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

parameter DELAY = 0;

assign #DELAY out = VSS ? 1'bx : (~VDD ? 1'b0 : ~in );

endmodule


module skywater_lvlshift_inv_chain_3__w_sup(
    input  wire in,
    output wire outb,
    inout  wire VDD,
    inout  wire VSS
);

skywater_lvlshift_inv_5__w_sup XINV (
    .in( in ),
    .out( outb ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_lvlshift_lvshift_core_1__w_sup(
    input  wire inn,
    input  wire inp,
    output wire outn,
    output wire outp,
    inout  wire VDD,
    inout  wire VSS
);

parameter DELAY = 0;
logic outp_temp;
logic outn_temp;

// add first two lines of casez to eliminate any X output.  This is done to debug innovus
always_comb begin
    casez ({inp, inn, VDD, VSS})
        4'b00_10: {outp_temp, outn_temp} = 2'b00;
        4'b11_10: {outp_temp, outn_temp} = 2'b11;
        4'b10_10: {outp_temp, outn_temp} = 2'b10;
        4'b01_10: {outp_temp, outn_temp} = 2'b01;
        4'b??_00: {outp_temp, outn_temp} = 2'b00;
        default: {outp_temp, outn_temp} = 2'bxx;
    endcase
end

assign #DELAY outp = outp_temp;
assign #DELAY outn = outn_temp;

endmodule


module skywater_lvlshift_lvshift_core_w_drivers_1__w_sup(
    input  wire in,
    input  wire inb,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

wire midn;
wire midp;

skywater_lvlshift_inv_chain_3__w_sup XBUFN (
    .in( midn ),
    .outb( out ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_lvlshift_lvshift_core_1__w_sup XCORE (
    .inn( inb ),
    .inp( in ),
    .outn( midn ),
    .outp( midp ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_lvlshift__w_sup(
    input  wire in,
    output wire out,
    inout  wire VDD,
    inout  wire VDD_in,
    inout  wire VSS
);

wire in_buf;
wire inb_buf;

skywater_lvlshift_inv_chain_2__w_sup XBUF (
    .in( in ),
    .out( in_buf ),
    .outb( inb_buf ),
    .VDD( VDD_in ),
    .VSS( VSS )
);

skywater_lvlshift_lvshift_core_w_drivers_1__w_sup XLEV (
    .in( in_buf ),
    .inb( inb_buf ),
    .out( out ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_lvlshift(
    input  wire in,
    output wire out
);

wire VDD_val;
wire VDD_in_val;
wire VSS_val;

assign VDD_val = 1'b1;
assign VDD_in_val = 1'b1;
assign VSS_val = 1'b0;

skywater_lvlshift__w_sup XDUT (
    .in( in ),
    .out( out ),
    .VDD( VDD_val ),
    .VDD_in( VDD_in_val ),
    .VSS( VSS_val )
);

endmodule
