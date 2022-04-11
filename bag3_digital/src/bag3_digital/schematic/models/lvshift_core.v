{{ _header }}

parameter DELAY = {{ delay | default(0, true) }};
logic outp_temp;
logic outn_temp;

{% if _sch_params['has_rst'] %}
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
{% else %}
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
{% endif %}

assign #DELAY outp = outp_temp;
assign #DELAY outn = outn_temp;

endmodule
