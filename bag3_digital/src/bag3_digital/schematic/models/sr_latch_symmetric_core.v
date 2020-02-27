{{ _header }}

parameter DELAY = {{ delay | default(0, true) }};
logic temp_q, temp_qb;

{% if _sch_params['has_rstb'] %}
always_comb begin
    casez ({rsthb, r, sb, VDD, VSS})
        5'b00?10: temp_q = 1'b1;
        5'b01?10: temp_q = 1'bx;
        5'b10010: temp_q = 1'b1;
        5'b10110: temp_q = ~temp_qb;
        5'b11010: temp_q = 1'bx;
        5'b11110: temp_q = 1'b0;
        5'b???00: temp_q = 1'b0;
        default : temp_q = 1'bx;
    endcase
end

always_comb begin
    casez ({rstlb, s, rb, VDD, VSS})
        5'b00?10: temp_qb = 1'b1;
        5'b01?10: temp_qb = 1'bx;
        5'b10010: temp_qb = 1'b1;
        5'b10110: temp_qb = ~temp_q;
        5'b11010: temp_qb = 1'bx;
        5'b11110: temp_qb = 1'b0;
        5'b???00: temp_qb = 1'b0;
        default: temp_qb = 1'bx;
    endcase
end
{% else %}
always_comb begin
    casez ({r, sb, VDD, VSS})
        4'b0010: temp_q = 1'b1;
        4'b0110: temp_q = ~temp_qb;
        4'b1010: temp_q = 1'bx;
        4'b1110: temp_q = 1'b0;
        4'b??00: temp_q = 1'b0;
        default : temp_q = 1'bx;
    endcase
end

always_comb begin
    casez ({s, rb, VDD, VSS})
        4'b0010: temp_qb = 1'b1;
        4'b0110: temp_qb = ~temp_q;
        4'b1010: temp_qb = 1'bx;
        4'b1110: temp_qb = 1'b0;
        4'b??00: temp_qb = 1'b0;
        default: temp_qb = 1'bx;
    endcase
end
{% endif %}

assign #DELAY q = temp_q;
assign #DELAY qb = temp_qb;

endmodule
