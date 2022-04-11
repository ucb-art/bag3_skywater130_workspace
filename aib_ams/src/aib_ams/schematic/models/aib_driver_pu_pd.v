{{ _header }}

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

    assign{% if not _sch_params['strong'] %} (weak0, weak1){% endif %} out = out_temp;

endmodule
