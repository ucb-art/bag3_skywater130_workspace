`timescale 1ps/1ps 


module skywater_rxanlg_inv_7__w_sup(
    input  wire in,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

parameter DELAY = 0;

assign #DELAY out = VSS ? 1'bx : (~VDD ? 1'b0 : ~in );

endmodule


module skywater_rxanlg_inv_8__w_sup(
    input  wire in,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

parameter DELAY = 0;

assign #DELAY out = VSS ? 1'bx : (~VDD ? 1'b0 : ~in );

endmodule


module skywater_rxanlg_inv_chain_6__w_sup(
    input  wire in,
    output wire outb,
    inout  wire VDD,
    inout  wire VSS
);

skywater_rxanlg_inv_8__w_sup XINV (
    .in( in ),
    .out( outb ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_rxanlg_lvshift_core_3__w_sup(
    input  wire inn,
    input  wire inp,
    input  wire rst_casc,
    input  wire rst_outn,
    input  wire rst_outp,
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
    casez ({rst_outp, rst_outn, rst_casc, inp, inn, VDD, VSS})
        7'b00_1_00_10: {outp_temp, outn_temp} = 2'b00;
        7'b00_1_11_10: {outp_temp, outn_temp} = 2'b11;
        7'b10_0_??_10: {outp_temp, outn_temp} = 2'b01;
        7'b01_0_??_10: {outp_temp, outn_temp} = 2'b10;
        7'b00_1_10_10: {outp_temp, outn_temp} = 2'b10;
        7'b00_1_01_10: {outp_temp, outn_temp} = 2'b01;
        7'b10_1_10_10: {outp_temp, outn_temp} = 2'b01;
        7'b01_1_01_10: {outp_temp, outn_temp} = 2'b10;
        7'b??_?_??_00: {outp_temp, outn_temp} = 2'b00;
        default: {outp_temp, outn_temp} = 2'bxx;
    endcase
end

assign #DELAY outp = outp_temp;
assign #DELAY outn = outn_temp;

endmodule


module skywater_rxanlg_lvshift_core_w_drivers_3__w_sup(
    input  wire in,
    input  wire inb,
    input  wire rst_casc,
    input  wire rst_out,
    input  wire rst_outb,
    output wire out,
    output wire outb,
    inout  wire VDD,
    inout  wire VSS
);

wire midn;
wire midp;

skywater_rxanlg_inv_chain_6__w_sup XBUFN (
    .in( midn ),
    .outb( out ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_rxanlg_inv_chain_6__w_sup XBUFP (
    .in( midp ),
    .outb( outb ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_rxanlg_lvshift_core_3__w_sup XCORE (
    .inn( inb ),
    .inp( in ),
    .rst_casc( rst_casc ),
    .rst_outn( rst_outb ),
    .rst_outp( rst_out ),
    .outn( midn ),
    .outp( midp ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_rxanlg_inv_9__w_sup(
    input  wire in,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

parameter DELAY = 0;

assign #DELAY out = VSS ? 1'bx : (~VDD ? 1'b0 : ~in );

endmodule


module skywater_rxanlg_inv_chain_7__w_sup(
    input  wire in,
    output wire out,
    output wire outb,
    inout  wire VDD,
    inout  wire VSS
);

skywater_rxanlg_inv_9__w_sup XINV0 (
    .in( in ),
    .out( outb ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_rxanlg_inv_9__w_sup XINV1 (
    .in( outb ),
    .out( out ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_rxanlg_inv_chain_8__w_sup(
    input  wire in,
    output wire outb,
    inout  wire VDD,
    inout  wire VSS
);

skywater_rxanlg_inv_9__w_sup XINV (
    .in( in ),
    .out( outb ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_rxanlg_lvshift_core_4__w_sup(
    input  wire inn,
    input  wire inp,
    input  wire rst_casc,
    input  wire rst_outn,
    input  wire rst_outp,
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
    casez ({rst_outp, rst_outn, rst_casc, inp, inn, VDD, VSS})
        7'b00_1_00_10: {outp_temp, outn_temp} = 2'b00;
        7'b00_1_11_10: {outp_temp, outn_temp} = 2'b11;
        7'b10_0_??_10: {outp_temp, outn_temp} = 2'b01;
        7'b01_0_??_10: {outp_temp, outn_temp} = 2'b10;
        7'b00_1_10_10: {outp_temp, outn_temp} = 2'b10;
        7'b00_1_01_10: {outp_temp, outn_temp} = 2'b01;
        7'b10_1_10_10: {outp_temp, outn_temp} = 2'b01;
        7'b01_1_01_10: {outp_temp, outn_temp} = 2'b10;
        7'b??_?_??_00: {outp_temp, outn_temp} = 2'b00;
        default: {outp_temp, outn_temp} = 2'bxx;
    endcase
end

assign #DELAY outp = outp_temp;
assign #DELAY outn = outn_temp;

endmodule


module skywater_rxanlg_lvshift_core_w_drivers_4__w_sup(
    input  wire in,
    input  wire inb,
    input  wire rst_casc,
    input  wire rst_out,
    input  wire rst_outb,
    output wire out,
    output wire outb,
    inout  wire VDD,
    inout  wire VSS
);

wire midn;
wire midp;

skywater_rxanlg_inv_chain_8__w_sup XBUFN (
    .in( midn ),
    .outb( out ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_rxanlg_inv_chain_8__w_sup XBUFP (
    .in( midp ),
    .outb( outb ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_rxanlg_lvshift_core_4__w_sup XCORE (
    .inn( inb ),
    .inp( in ),
    .rst_casc( rst_casc ),
    .rst_outn( rst_outb ),
    .rst_outp( rst_out ),
    .outn( midn ),
    .outp( midp ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_rxanlg_lvshift_1__w_sup(
    input  wire in,
    input  wire rst_casc,
    input  wire rst_out,
    input  wire rst_outb,
    output wire out,
    output wire outb,
    inout  wire VDD,
    inout  wire VDD_in,
    inout  wire VSS
);

wire in_buf;
wire inb_buf;

skywater_rxanlg_inv_chain_7__w_sup XBUF (
    .in( in ),
    .out( in_buf ),
    .outb( inb_buf ),
    .VDD( VDD_in ),
    .VSS( VSS )
);

skywater_rxanlg_lvshift_core_w_drivers_4__w_sup XLEV (
    .in( in_buf ),
    .inb( inb_buf ),
    .rst_casc( rst_casc ),
    .rst_out( rst_out ),
    .rst_outb( rst_outb ),
    .out( out ),
    .outb( outb ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_rxanlg_inv_10__w_sup(
    input  wire in,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

parameter DELAY = 0;

assign #DELAY out = VSS ? 1'bx : (~VDD ? 1'b0 : ~in );

endmodule


module skywater_rxanlg_inv_chain_9__w_sup(
    input  wire in,
    output wire outb,
    inout  wire VDD,
    inout  wire VSS
);

skywater_rxanlg_inv_10__w_sup XINV (
    .in( in ),
    .out( outb ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_rxanlg_lvshift_core_5__w_sup(
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


module skywater_rxanlg_lvshift_core_w_drivers_5__w_sup(
    input  wire in,
    input  wire inb,
    output wire out,
    output wire outb,
    inout  wire VDD,
    inout  wire VSS
);

wire midn;
wire midp;

skywater_rxanlg_inv_chain_9__w_sup XBUFN (
    .in( midn ),
    .outb( out ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_rxanlg_inv_chain_9__w_sup XBUFP (
    .in( midp ),
    .outb( outb ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_rxanlg_lvshift_core_5__w_sup XCORE (
    .inn( inb ),
    .inp( in ),
    .outn( midn ),
    .outp( midp ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_rxanlg_inv_11__w_sup(
    input  wire in,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

parameter DELAY = 0;

assign #DELAY out = VSS ? 1'bx : (~VDD ? 1'b0 : ~in );

endmodule


module skywater_rxanlg_inv_chain_10__w_sup(
    input  wire in,
    output wire out,
    output wire outb,
    inout  wire VDD,
    inout  wire VSS
);

skywater_rxanlg_inv_7__w_sup XINV0 (
    .in( in ),
    .out( outb ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_rxanlg_inv_11__w_sup XINV1 (
    .in( outb ),
    .out( out ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_rxanlg_inv_12__w_sup(
    input  wire in,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

parameter DELAY = 0;

assign #DELAY out = VSS ? 1'bx : (~VDD ? 1'b0 : ~in );

endmodule


module skywater_rxanlg_inv_chain_11__w_sup(
    input  wire in,
    output wire outb,
    inout  wire VDD,
    inout  wire VSS
);

wire [0:0] mid;
wire out;

skywater_rxanlg_inv_9__w_sup XINV0 (
    .in( in ),
    .out( mid[0] ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_rxanlg_inv_7__w_sup XINV1 (
    .in( mid[0] ),
    .out( out ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_rxanlg_inv_12__w_sup XINV2 (
    .in( out ),
    .out( outb ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_rxanlg_nand_2__w_sup(
    input  wire [1:0] in,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

parameter DELAY = 0;

assign #DELAY out = VSS ? 1'bx : (~VDD ? 1'b0 : ~&in );

endmodule


module skywater_rxanlg_nor_2__w_sup(
    input  wire [1:0] in,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

parameter DELAY = 0;

   assign #DELAY out = VSS ? 1'bx : (~VDD ? 1'b0 : ~|in );
   

endmodule


module skywater_rxanlg_aib_se2diff_match_1__w_sup(
    input  wire en,
    input  wire enb,
    input  wire inn,
    input  wire inp,
    output wire outn,
    output wire outp,
    inout  wire VDD,
    inout  wire VSS
);

wire nand_out;
wire nor_out;

skywater_rxanlg_inv_chain_11__w_sup XBUFN (
    .in( nor_out ),
    .outb( outn ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_rxanlg_inv_chain_11__w_sup XBUFP (
    .in( nand_out ),
    .outb( outp ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_rxanlg_nand_2__w_sup XNAND (
    .in( {en,inp} ),
    .out( nand_out ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_rxanlg_nor_2__w_sup XNOR (
    .in( {enb,inn} ),
    .out( nor_out ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_rxanlg_inv_13__w_sup(
    input  wire in,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

parameter DELAY = 0;

assign #DELAY out = VSS ? 1'bx : (~VDD ? 1'b0 : ~in );

endmodule


module skywater_rxanlg_passgate_1__w_sup(
    input  wire en,
    input  wire enb,
    input  wire s,
    output trireg d,
    inout  wire VDD,
    inout  wire VSS
);

parameter DELAY = 0;
wire tmp;

assign #DELAY tmp = VSS ? 1'bx : (~VDD ? 1'b0 : s);

tranif1 XTRN1 (d, tmp, en );
tranif0 XTRN0 (d, tmp, enb);

endmodule


module skywater_rxanlg_se_to_diff_1__w_sup(
    input  wire in,
    output wire outn,
    output wire outp,
    inout  wire VDD,
    inout  wire VSS
);

wire midn_inv;
wire midn_pass0;
wire midn_pass1;
wire midp;

skywater_rxanlg_inv_9__w_sup XINVN0 (
    .in( in ),
    .out( midn_inv ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_rxanlg_inv_7__w_sup XINVN1 (
    .in( midn_inv ),
    .out( midp ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_rxanlg_inv_12__w_sup XINVN2 (
    .in( midp ),
    .out( outn ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_rxanlg_inv_13__w_sup XINVP0 (
    .in( in ),
    .out( midn_pass0 ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_rxanlg_inv_12__w_sup XINVP1 (
    .in( midn_pass1 ),
    .out( outp ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_rxanlg_passgate_1__w_sup XPASS (
    .en( VDD ),
    .enb( VSS ),
    .s( midn_pass0 ),
    .d( midn_pass1 ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_rxanlg_nor_3__w_sup(
    input  wire [1:0] in,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

parameter DELAY = 0;

   assign #DELAY out = VSS ? 1'bx : (~VDD ? 1'b0 : ~|in );
   

endmodule


module skywater_rxanlg_nand_3__w_sup(
    input  wire [1:0] in,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

parameter DELAY = 0;

assign #DELAY out = VSS ? 1'bx : (~VDD ? 1'b0 : ~&in );

endmodule


module skywater_rxanlg_aib_se2diff_1__w_sup(
    input  wire en,
    input  wire enb,
    input  wire in,
    output wire outn,
    output wire outp,
    inout  wire VDD,
    inout  wire VSS
);

wire inb;
wire nc;

skywater_rxanlg_se_to_diff_1__w_sup XCORE (
    .in( inb ),
    .outn( outp ),
    .outp( outn ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_rxanlg_nor_3__w_sup XDUM (
    .in( {enb,VDD} ),
    .out( nc ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_rxanlg_nand_3__w_sup XNAND (
    .in( {en,in} ),
    .out( inb ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_rxanlg__w_sup(
    input  wire clk_en,
    input  wire data_en,
    input  wire iclkn,
    input  wire iopad,
    input  wire por,
    output wire oclkn,
    output wire oclkp,
    output wire odat,
    output wire odat_async,
    output wire por_vccl,
    output wire porb_vccl,
    inout  wire VDDCore,
    inout  wire VDDIO,
    inout  wire VSS
);

wire clk_en_vccl;
wire clk_enb_vccl;
wire data_en_vccl;
wire data_enb_vccl;
wire dumn;
wire dump;
wire oclkn_vccl;
wire oclkp_vccl;
wire odatb;
wire odatn_vccl;
wire odatp_vccl;
wire por_buf;
wire porb_buf;
wire [2:0] unused;

skywater_rxanlg_inv_7__w_sup XDUM (
    .in( VSS ),
    .out( unused[0] ),
    .VDD( VDDCore ),
    .VSS( VSS )
);

skywater_rxanlg_inv_7__w_sup XINV (
    .in( odatb ),
    .out( odat_async ),
    .VDD( VDDCore ),
    .VSS( VSS )
);

skywater_rxanlg_lvshift_core_w_drivers_3__w_sup XLV_CLK (
    .in( oclkp_vccl ),
    .inb( oclkn_vccl ),
    .rst_casc( porb_buf ),
    .rst_out( por_buf ),
    .rst_outb( VSS ),
    .out( oclkp ),
    .outb( oclkn ),
    .VDD( VDDCore ),
    .VSS( VSS )
);

skywater_rxanlg_lvshift_1__w_sup XLV_CLK_EN (
    .in( clk_en ),
    .rst_casc( porb_vccl ),
    .rst_out( por_vccl ),
    .rst_outb( VSS ),
    .out( clk_en_vccl ),
    .outb( clk_enb_vccl ),
    .VDD( VDDIO ),
    .VDD_in( VDDCore ),
    .VSS( VSS )
);

skywater_rxanlg_lvshift_core_w_drivers_3__w_sup XLV_DATA (
    .in( odatp_vccl ),
    .inb( odatn_vccl ),
    .rst_casc( porb_buf ),
    .rst_out( por_buf ),
    .rst_outb( VSS ),
    .out( odat ),
    .outb( odatb ),
    .VDD( VDDCore ),
    .VSS( VSS )
);

skywater_rxanlg_lvshift_1__w_sup XLV_DATA_EN (
    .in( data_en ),
    .rst_casc( porb_vccl ),
    .rst_out( por_vccl ),
    .rst_outb( VSS ),
    .out( data_en_vccl ),
    .outb( data_enb_vccl ),
    .VDD( VDDIO ),
    .VDD_in( VDDCore ),
    .VSS( VSS )
);

skywater_rxanlg_lvshift_core_w_drivers_5__w_sup XLV_DUM (
    .in( dump ),
    .inb( dumn ),
    .out( unused[1] ),
    .outb( unused[2] ),
    .VDD( VDDIO ),
    .VSS( VSS )
);

skywater_rxanlg_lvshift_core_w_drivers_5__w_sup XLV_POR (
    .in( por_buf ),
    .inb( porb_buf ),
    .out( por_vccl ),
    .outb( porb_vccl ),
    .VDD( VDDIO ),
    .VSS( VSS )
);

skywater_rxanlg_inv_chain_10__w_sup XPOR (
    .in( por ),
    .out( por_buf ),
    .outb( porb_buf ),
    .VDD( VDDCore ),
    .VSS( VSS )
);

skywater_rxanlg_inv_chain_10__w_sup XPOR_DUM (
    .in( VSS ),
    .out( dump ),
    .outb( dumn ),
    .VDD( VDDCore ),
    .VSS( VSS )
);

skywater_rxanlg_aib_se2diff_match_1__w_sup XSE_CLK (
    .en( clk_en_vccl ),
    .enb( clk_enb_vccl ),
    .inn( iclkn ),
    .inp( iopad ),
    .outn( oclkn_vccl ),
    .outp( oclkp_vccl ),
    .VDD( VDDIO ),
    .VSS( VSS )
);

skywater_rxanlg_aib_se2diff_1__w_sup XSE_DATA (
    .en( data_en_vccl ),
    .enb( data_enb_vccl ),
    .in( iopad ),
    .outn( odatn_vccl ),
    .outp( odatp_vccl ),
    .VDD( VDDIO ),
    .VSS( VSS )
);

endmodule


module skywater_rxanlg(
    input  wire clk_en,
    input  wire data_en,
    input  wire iclkn,
    input  wire iopad,
    input  wire por,
    output wire oclkn,
    output wire oclkp,
    output wire odat,
    output wire odat_async
);

wire por_vccl_val;
wire porb_vccl_val;
wire VDDCore_val;
wire VDDIO_val;
wire VSS_val;

assign por_vccl_val = 1'b1;
assign porb_vccl_val = 1'b1;
assign VDDCore_val = 1'b1;
assign VDDIO_val = 1'b1;
assign VSS_val = 1'b0;

skywater_rxanlg__w_sup XDUT (
    .clk_en( clk_en ),
    .data_en( data_en ),
    .iclkn( iclkn ),
    .iopad( iopad ),
    .por( por ),
    .oclkn( oclkn ),
    .oclkp( oclkp ),
    .odat( odat ),
    .odat_async( odat_async ),
    .por_vccl( por_vccl_val ),
    .porb_vccl( porb_vccl_val ),
    .VDDCore( VDDCore_val ),
    .VDDIO( VDDIO_val ),
    .VSS( VSS_val )
);

endmodule
