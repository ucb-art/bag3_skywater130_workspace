{{ _header }}

logic woutp;
logic woutn;

{% if _sch_params['has_rstb'] %}
always @(clk or rstb) begin
    if (~rstb || ~clk) {woutp, woutn} <= 2'b11;
{% else %}
always @(clk) begin
    if (~clk) {woutp, woutn} <= 2'b11;
{% endif %}
    else begin
        case ({inp, inn})
            2'b10: {woutp, woutn} <= 2'b10;
            2'b01: {woutp, woutn} <= 2'b01;
            2'b00: {woutp, woutn} <= 2'b11;
            2'b11: {woutp, woutn} <= 2'b10; // Added a bias to ensure that don't fall into
                                            // default case that produces x's simply because
                                            // of arbitrary event ordering resolution.
            default: {woutp, woutn} <= 2'bxx;
        endcase
    end
end

assign outp = VSS ? 1'bx : (~VDD ? 1'b0 : woutp);
assign outn = VSS ? 1'bx : (~VDD ? 1'b0 : woutn);

endmodule
