`timescale 1ps/1ps 


module skywater_txanlg_aib_driver_pu_pd_2__w_sup(
    input  wire pden,
    input  wire puenb,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

    logic out_temp;

    always_comb begin
        // puenb connects to PMOS, pden connects to NMOS
        casez ({VDD, VSS, puenb, pden})
           4'b10_00: out_temp = 1'b1;
           4'b10_01: out_temp = 1'bx;
           4'b10_10: out_temp = 1'bz;
           4'b10_11: out_temp = 1'b0;
           4'b00_??: out_temp = 1'b0;
           default:  out_temp = 1'bx;
        endcase
    end

    assign (weak0, weak1) out = out_temp;

endmodule


module skywater_txanlg_current_summer_1__w_sup(
    input  wire [6:0] in,
    output wire out
);

    tran tr0(in[0], out);
    tran tr1(in[1], out);
    tran tr2(in[2], out);
    tran tr3(in[3], out);
    tran tr4(in[4], out);
    tran tr5(in[5], out);
    tran tr6(in[6], out);

endmodule


module skywater_txanlg_nand_1__w_sup(
    input  wire [1:0] in,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

parameter DELAY = 0;

assign #DELAY out = VSS ? 1'bx : (~VDD ? 1'b0 : ~&in );

endmodule


module skywater_txanlg_nor_1__w_sup(
    input  wire [1:0] in,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

parameter DELAY = 0;

   assign #DELAY out = VSS ? 1'bx : (~VDD ? 1'b0 : ~|in );
   

endmodule


module skywater_txanlg_aib_driver_pu_pd_3__w_sup(
    input  wire pden,
    input  wire puenb,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

    logic out_temp;

    always_comb begin
        // puenb connects to PMOS, pden connects to NMOS
        casez ({VDD, VSS, puenb, pden})
           4'b10_00: out_temp = 1'b1;
           4'b10_01: out_temp = 1'bx;
           4'b10_10: out_temp = 1'bz;
           4'b10_11: out_temp = 1'b0;
           4'b00_??: out_temp = 1'b0;
           default:  out_temp = 1'bx;
        endcase
    end

    assign out = out_temp;

endmodule


module skywater_txanlg_aib_driver_output_unit_cell_1__w_sup(
    input  wire en,
    input  wire enb,
    input  wire in,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

wire nand_pu;
wire nor_pd;

skywater_txanlg_nand_1__w_sup XNAND (
    .in( {en,in} ),
    .out( nand_pu ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_txanlg_nor_1__w_sup XNOR (
    .in( {enb,in} ),
    .out( nor_pd ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_txanlg_aib_driver_pu_pd_3__w_sup Xpupd (
    .pden( nor_pd ),
    .puenb( nand_pu ),
    .out( out ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_txanlg_aib_driver_output_driver_1__w_sup(
    input  wire din,
    input  wire [1:0] n_enb_drv,
    input  wire [1:0] p_en_drv,
    input  wire tristate,
    input  wire tristateb,
    input  wire weak_pden,
    input  wire weak_puenb,
    output wire txpadout,
    inout  wire VDD,
    inout  wire VSS
);

wire [6:0] txpadout_tmp;

skywater_txanlg_aib_driver_pu_pd_2__w_sup XPUPD (
    .pden( weak_pden ),
    .puenb( weak_puenb ),
    .out( txpadout_tmp[6] ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_txanlg_current_summer_1__w_sup XSUM (
    .in( txpadout_tmp[6:0] ),
    .out( txpadout )
);

skywater_txanlg_aib_driver_output_unit_cell_1__w_sup XUNIT_5 (
    .en( p_en_drv[0] ),
    .enb( n_enb_drv[0] ),
    .in( din ),
    .out( txpadout_tmp[5] ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_txanlg_aib_driver_output_unit_cell_1__w_sup XUNIT_4 (
    .en( p_en_drv[1] ),
    .enb( n_enb_drv[1] ),
    .in( din ),
    .out( txpadout_tmp[4] ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_txanlg_aib_driver_output_unit_cell_1__w_sup XUNIT_3 (
    .en( p_en_drv[1] ),
    .enb( n_enb_drv[1] ),
    .in( din ),
    .out( txpadout_tmp[3] ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_txanlg_aib_driver_output_unit_cell_1__w_sup XUNIT_2 (
    .en( tristateb ),
    .enb( tristate ),
    .in( din ),
    .out( txpadout_tmp[2] ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_txanlg_aib_driver_output_unit_cell_1__w_sup XUNIT_1 (
    .en( tristateb ),
    .enb( tristate ),
    .in( din ),
    .out( txpadout_tmp[1] ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_txanlg_aib_driver_output_unit_cell_1__w_sup XUNIT_0 (
    .en( tristateb ),
    .enb( tristate ),
    .in( din ),
    .out( txpadout_tmp[0] ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_txanlg_inv_3__w_sup(
    input  wire in,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

parameter DELAY = 0;

assign #DELAY out = VSS ? 1'bx : (~VDD ? 1'b0 : ~in );

endmodule


module skywater_txanlg_inv_chain_4__w_sup(
    input  wire in,
    output wire out,
    output wire outb,
    inout  wire VDD,
    inout  wire VSS
);

skywater_txanlg_inv_3__w_sup XINV0 (
    .in( in ),
    .out( outb ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_txanlg_inv_3__w_sup XINV1 (
    .in( outb ),
    .out( out ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_txanlg_inv_4__w_sup(
    input  wire in,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

parameter DELAY = 0;

assign #DELAY out = VSS ? 1'bx : (~VDD ? 1'b0 : ~in );

endmodule


module skywater_txanlg_inv_chain_5__w_sup(
    input  wire in,
    output wire outb,
    inout  wire VDD,
    inout  wire VSS
);

skywater_txanlg_inv_4__w_sup XINV (
    .in( in ),
    .out( outb ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_txanlg_lvshift_core_1__w_sup(
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


module skywater_txanlg_lvshift_core_w_drivers_4__w_sup(
    input  wire in,
    input  wire inb,
    input  wire rst_casc,
    input  wire rst_out,
    input  wire rst_outb,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

wire midn;
wire midp;

skywater_txanlg_inv_chain_5__w_sup XBUFN (
    .in( midn ),
    .outb( out ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_txanlg_lvshift_core_1__w_sup XCORE (
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


module skywater_txanlg_lvshift_4__w_sup(
    input  wire in,
    input  wire rst_casc,
    input  wire rst_out,
    input  wire rst_outb,
    output wire out,
    inout  wire VDD,
    inout  wire VDD_in,
    inout  wire VSS
);

wire in_buf;
wire inb_buf;

skywater_txanlg_inv_chain_4__w_sup XBUF (
    .in( in ),
    .out( in_buf ),
    .outb( inb_buf ),
    .VDD( VDD_in ),
    .VSS( VSS )
);

skywater_txanlg_lvshift_core_w_drivers_4__w_sup XLEV (
    .in( in_buf ),
    .inb( inb_buf ),
    .rst_casc( rst_casc ),
    .rst_out( rst_out ),
    .rst_outb( rst_outb ),
    .out( out ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_txanlg_inv_5__w_sup(
    input  wire in,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

parameter DELAY = 0;

assign #DELAY out = VSS ? 1'bx : (~VDD ? 1'b0 : ~in );

endmodule


module skywater_txanlg_inv_chain_6__w_sup(
    input  wire in,
    output wire out,
    output wire outb,
    inout  wire VDD,
    inout  wire VSS
);

skywater_txanlg_inv_5__w_sup XINV0 (
    .in( in ),
    .out( outb ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_txanlg_inv_5__w_sup XINV1 (
    .in( outb ),
    .out( out ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_txanlg_inv_chain_7__w_sup(
    input  wire in,
    output wire outb,
    inout  wire VDD,
    inout  wire VSS
);

skywater_txanlg_inv_5__w_sup XINV (
    .in( in ),
    .out( outb ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_txanlg_lvshift_core_w_drivers_5__w_sup(
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

skywater_txanlg_inv_chain_7__w_sup XBUFN (
    .in( midn ),
    .outb( out ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_txanlg_inv_chain_7__w_sup XBUFP (
    .in( midp ),
    .outb( outb ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_txanlg_lvshift_core_1__w_sup XCORE (
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


module skywater_txanlg_lvshift_5__w_sup(
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

skywater_txanlg_inv_chain_6__w_sup XBUF (
    .in( in ),
    .out( in_buf ),
    .outb( inb_buf ),
    .VDD( VDD_in ),
    .VSS( VSS )
);

skywater_txanlg_lvshift_core_w_drivers_5__w_sup XLEV (
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


module skywater_txanlg_lvshift_core_w_drivers_6__w_sup(
    input  wire in,
    input  wire inb,
    input  wire rst_casc,
    input  wire rst_out,
    input  wire rst_outb,
    output wire outb,
    inout  wire VDD,
    inout  wire VSS
);

wire midn;
wire midp;

skywater_txanlg_inv_chain_7__w_sup XBUFP (
    .in( midp ),
    .outb( outb ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_txanlg_lvshift_core_1__w_sup XCORE (
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


module skywater_txanlg_lvshift_6__w_sup(
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

skywater_txanlg_inv_chain_6__w_sup XBUF (
    .in( in ),
    .out( in_buf ),
    .outb( inb_buf ),
    .VDD( VDD_in ),
    .VSS( VSS )
);

skywater_txanlg_lvshift_core_w_drivers_6__w_sup XLEV (
    .in( in_buf ),
    .inb( inb_buf ),
    .rst_casc( rst_casc ),
    .rst_out( rst_out ),
    .rst_outb( rst_outb ),
    .outb( outb ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_txanlg_lvshift_core_w_drivers_7__w_sup(
    input  wire in,
    input  wire inb,
    input  wire rst_casc,
    input  wire rst_out,
    input  wire rst_outb,
    output wire out,
    inout  wire VDD,
    inout  wire VSS
);

wire midn;
wire midp;

skywater_txanlg_inv_chain_7__w_sup XBUFN (
    .in( midn ),
    .outb( out ),
    .VDD( VDD ),
    .VSS( VSS )
);

skywater_txanlg_lvshift_core_1__w_sup XCORE (
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


module skywater_txanlg_lvshift_7__w_sup(
    input  wire in,
    input  wire rst_casc,
    input  wire rst_out,
    input  wire rst_outb,
    output wire out,
    inout  wire VDD,
    inout  wire VDD_in,
    inout  wire VSS
);

wire in_buf;
wire inb_buf;

skywater_txanlg_inv_chain_6__w_sup XBUF (
    .in( in ),
    .out( in_buf ),
    .outb( inb_buf ),
    .VDD( VDD_in ),
    .VSS( VSS )
);

skywater_txanlg_lvshift_core_w_drivers_7__w_sup XLEV (
    .in( in_buf ),
    .inb( inb_buf ),
    .rst_casc( rst_casc ),
    .rst_out( rst_out ),
    .rst_outb( rst_outb ),
    .out( out ),
    .VDD( VDD ),
    .VSS( VSS )
);

endmodule


module skywater_txanlg__w_sup(
    input  wire din,
    input  wire [1:0] indrv_buf,
    input  wire [1:0] ipdrv_buf,
    input  wire itx_en_buf,
    input  wire por_vccl,
    input  wire porb_vccl,
    input  wire weak_pulldownen,
    input  wire weak_pullupenb,
    output wire txpadout,
    inout  wire VDDCore,
    inout  wire VDDIO,
    inout  wire VSS
);

wire din_io;
wire [1:0] nen_drv_io;
wire [1:0] nen_drvb_io;
wire pden_io;
wire [1:0] pen_drv_io;
wire puenb_io;
wire tristate_io;
wire tristateb_io;

skywater_txanlg_aib_driver_output_driver_1__w_sup XDRV (
    .din( din_io ),
    .n_enb_drv( nen_drvb_io[1:0] ),
    .p_en_drv( pen_drv_io[1:0] ),
    .tristate( tristate_io ),
    .tristateb( tristateb_io ),
    .weak_pden( pden_io ),
    .weak_puenb( puenb_io ),
    .txpadout( txpadout ),
    .VDD( VDDIO ),
    .VSS( VSS )
);

skywater_txanlg_lvshift_4__w_sup XLV_DIN (
    .in( din ),
    .rst_casc( porb_vccl ),
    .rst_out( por_vccl ),
    .rst_outb( VSS ),
    .out( din_io ),
    .VDD( VDDIO ),
    .VDD_in( VDDCore ),
    .VSS( VSS )
);

skywater_txanlg_lvshift_5__w_sup XLV_ITX_EN (
    .in( itx_en_buf ),
    .rst_casc( porb_vccl ),
    .rst_out( por_vccl ),
    .rst_outb( VSS ),
    .out( tristateb_io ),
    .outb( tristate_io ),
    .VDD( VDDIO ),
    .VDD_in( VDDCore ),
    .VSS( VSS )
);

skywater_txanlg_lvshift_6__w_sup XLV_NDRV_1 (
    .in( indrv_buf[1] ),
    .rst_casc( porb_vccl ),
    .rst_out( por_vccl ),
    .rst_outb( VSS ),
    .out( nen_drv_io[1] ),
    .outb( nen_drvb_io[1] ),
    .VDD( VDDIO ),
    .VDD_in( VDDCore ),
    .VSS( VSS )
);

skywater_txanlg_lvshift_6__w_sup XLV_NDRV_0 (
    .in( indrv_buf[0] ),
    .rst_casc( porb_vccl ),
    .rst_out( por_vccl ),
    .rst_outb( VSS ),
    .out( nen_drv_io[0] ),
    .outb( nen_drvb_io[0] ),
    .VDD( VDDIO ),
    .VDD_in( VDDCore ),
    .VSS( VSS )
);

skywater_txanlg_lvshift_7__w_sup XLV_PD (
    .in( weak_pulldownen ),
    .rst_casc( porb_vccl ),
    .rst_out( VSS ),
    .rst_outb( por_vccl ),
    .out( pden_io ),
    .VDD( VDDIO ),
    .VDD_in( VDDCore ),
    .VSS( VSS )
);

skywater_txanlg_lvshift_7__w_sup XLV_PDRV_1 (
    .in( ipdrv_buf[1] ),
    .rst_casc( porb_vccl ),
    .rst_out( por_vccl ),
    .rst_outb( VSS ),
    .out( pen_drv_io[1] ),
    .VDD( VDDIO ),
    .VDD_in( VDDCore ),
    .VSS( VSS )
);

skywater_txanlg_lvshift_7__w_sup XLV_PDRV_0 (
    .in( ipdrv_buf[0] ),
    .rst_casc( porb_vccl ),
    .rst_out( por_vccl ),
    .rst_outb( VSS ),
    .out( pen_drv_io[0] ),
    .VDD( VDDIO ),
    .VDD_in( VDDCore ),
    .VSS( VSS )
);

skywater_txanlg_lvshift_7__w_sup XLV_PU (
    .in( weak_pullupenb ),
    .rst_casc( porb_vccl ),
    .rst_out( VSS ),
    .rst_outb( por_vccl ),
    .out( puenb_io ),
    .VDD( VDDIO ),
    .VDD_in( VDDCore ),
    .VSS( VSS )
);

endmodule


module skywater_txanlg(
    input  wire din,
    input  wire [1:0] indrv_buf,
    input  wire [1:0] ipdrv_buf,
    input  wire itx_en_buf,
    input  wire por_vccl,
    input  wire porb_vccl,
    input  wire weak_pulldownen,
    input  wire weak_pullupenb,
    output wire txpadout
);

wire VDDCore_val;
wire VDDIO_val;
wire VSS_val;

assign VDDCore_val = 1'b1;
assign VDDIO_val = 1'b1;
assign VSS_val = 1'b0;

skywater_txanlg__w_sup XDUT (
    .din( din ),
    .indrv_buf( indrv_buf ),
    .ipdrv_buf( ipdrv_buf ),
    .itx_en_buf( itx_en_buf ),
    .por_vccl( por_vccl ),
    .porb_vccl( porb_vccl ),
    .weak_pulldownen( weak_pulldownen ),
    .weak_pullupenb( weak_pullupenb ),
    .txpadout( txpadout ),
    .VDDCore( VDDCore_val ),
    .VDDIO( VDDIO_val ),
    .VSS( VSS_val )
);

endmodule
