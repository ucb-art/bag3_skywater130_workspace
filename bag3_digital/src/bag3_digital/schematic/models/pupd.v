{{ _header }}

parameter DELAY = {{ delay | default(0, true) }};

assign #DELAY out_pd = VSS ? 1'bx : (~VDD ? 1'b0 : (pden ? 1'b0 : 1'bz));
assign #DELAY out_pu = VSS ? 1'bx : (~VDD ? 1'b0 : (puenb ? 1'bz : 1'b1));

endmodule
