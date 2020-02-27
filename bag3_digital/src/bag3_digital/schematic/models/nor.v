{{ _header }}

parameter DELAY = {{ delay | default(0, true) }};

   assign #DELAY out = VSS ? 1'bx : (~VDD ? 1'b0 : ~|in );
   

endmodule
