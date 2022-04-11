{{ _header }}

parameter DELAY = {{ delay | default(0, true) }};
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
