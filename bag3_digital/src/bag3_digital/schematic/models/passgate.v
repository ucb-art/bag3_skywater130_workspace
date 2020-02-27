{{ _header }}

parameter DELAY = {{ delay | default(0, true) }};
wire tmp;

assign #DELAY tmp = VSS ? 1'bx : (~VDD ? 1'b0 : s);

tranif1 XTRN1 (d, tmp, en );
tranif0 XTRN0 (d, tmp, enb);

endmodule
