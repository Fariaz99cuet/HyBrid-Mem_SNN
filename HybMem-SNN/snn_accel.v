`timescale 1ns/1ps
// =====================================================================
//  HybMem-SNN : event-driven SNN accelerator  (196 -> 64 -> 10, MNIST 14x14)
//
//  * Weights live in two ROM/BRAM blocks, 8 weights (one "chunk") per read.
//  * Input pixels are rate-coded on the fly (pixel > random byte  ->  spike).
//  * Event-driven: a pixel / hidden neuron that does NOT spike costs 1 cycle and
//    fetches no weights.  A spike fetches its weight row (8 or 2 chunks) and adds it
//    to the membrane potentials of all target neurons.
//  * Neurons: v = v - (v >>> 4) + input,  spike if v >= TH, then v = v - TH.  No multipliers.
//  * ANGLE - hybrid weight memory: every weight bit can be declared "unreliable"
//    (noisy_mask bit = 1).  Reads of those bits flip randomly with probability 2^-ber_sel.
//    noisy_mask = 8'hFF -> all bits unreliable, 8'h0F -> only the 4 LSBs unreliable
//    (MSBs kept in a reliable memory).   ber_sel = 0 disables error injection.
//
//  Usage: write the 196 pixels through img_we/img_addr/img_data, pulse start,
//         wait for done, read pred / counts.
// =====================================================================
module snn_accel (
    input  clk,
    input  rst,                       // synchronous reset
    // image buffer write port
    input  img_we,
    input  [7:0] img_addr,            // 0..195
    input  [7:0] img_data,            // pixel 0..255
    // control
    input  start,                     // 1-cycle pulse
    input  [7:0] noisy_mask,          // which weight bits are stored in the unreliable memory
    input  [3:0] ber_sel,             // read bit-error probability = 2^-ber_sel  (0 = no errors)
    output reg busy,
    output reg done,                  // stays 1 until next start
    output [3:0]  pred,               // predicted digit
    output [79:0] counts,             // 10 x 8-bit output spike counts
    output reg [31:0] cycles,         // cycles used for the last image
    output reg [31:0] row_reads       // weight rows fetched (= events) for the last image
);
    localparam NPIX = 196;
    localparam NH   = 64;
    localparam VW   = 20;             // membrane width

`include "weights_init.vh"            // TSTEPS, TH, w1_mem, w2_mem (+ initial contents)

    // ---------------- states ----------------
    localparam [3:0] S_IDLE = 0, S_LEAK = 1, S_SCAN = 2, S_RD1 = 3, S_AC1 = 4, S_UPD1 = 5,
                     S_HSCAN = 6, S_RD2 = 7, S_AC2 = 8, S_UPD2 = 9, S_DONE = 10;

    reg [3:0]  st;
    reg [7:0]  pix;
    reg [2:0]  chunk;
    reg [5:0]  h;
    reg [7:0]  t;
    reg [31:0] ps;                    // input-spike PRNG (xorshift32)
    reg [63:0] hs;                    // hidden spikes of the current time step

    reg [7:0] img [0:NPIX-1];
    reg signed [VW-1:0] vh [0:NH-1];  // hidden membranes
    reg signed [VW-1:0] vo [0:15];    // output membranes (10 used, padded to 16)
    reg [7:0] cnt [0:15];

    // ---------------- xorshift32 ----------------
    function [31:0] xs32;
        input [31:0] x;
        reg [31:0] y;
        begin
            y = x ^ (x << 13);
            y = y ^ (y >> 17);
            y = y ^ (y << 5);
            xs32 = y;
        end
    endfunction

    // ---------------- weight memories (synchronous read -> block RAM) ----------------
    reg [63:0] w1_q, w2_q;
    wire [10:0] w1_addr = {pix, chunk};         // pix*8 + chunk
    wire [6:0]  w2_addr = {h, chunk[0]};        // h*2  + chunk
    always @(posedge clk) begin
        w1_q <= w1_mem[w1_addr];
        w2_q <= w2_mem[w2_addr];
    end

    always @(posedge clk) if (img_we) img[img_addr] <= img_data;

    // ---------------- unreliable-memory error injection ----------------
    reg [63:0] rnd;                             // free-running PRNG for bit errors
    reg [63:0] emask;
    integer j;
    function [63:0] rotl64;
        input [63:0] x;
        input integer s;
        begin
            rotl64 = (x << s) | (x >> (64 - s));
        end
    endfunction
    always @* begin
        emask = {64{1'b1}};
        for (j = 0; j < 15; j = j + 1)
            if (j < ber_sel) emask = emask & rotl64(rnd, 4*j + 1);   // AND of k random words -> p = 2^-k
        if (ber_sel == 0) emask = 64'd0;
    end

    wire [63:0] nerr;
    genvar g;
    generate
        for (g = 0; g < 8; g = g + 1) begin : NM
            assign nerr[g*8 +: 8] = emask[g*8 +: 8] & noisy_mask;
        end
    endgenerate
    wire [63:0] w1_rd = w1_q ^ nerr;            // what the datapath actually receives
    wire [63:0] w2_rd = w2_q ^ nerr;

    // ---------------- main FSM ----------------
    integer i, l;
    always @(posedge clk) begin
        rnd <= {xs32(rnd[63:32]), xs32(rnd[31:0])};
        if (busy) cycles <= cycles + 1;

        if (rst) begin
            st <= S_IDLE; busy <= 1'b0; done <= 1'b0;
            rnd <= {32'h9E3779B9, 32'h7F4A7C15};
            cycles <= 0; row_reads <= 0;
        end
        else begin
            case (st)
            S_IDLE: if (start) begin
                for (i = 0; i < NH; i = i + 1) vh[i] <= 0;
                for (i = 0; i < 16; i = i + 1) begin vo[i] <= 0; cnt[i] <= 0; end
                ps <= 32'd1;
                t <= 0; cycles <= 0; row_reads <= 0;
                busy <= 1'b1; done <= 1'b0;
                st <= S_LEAK;
            end

            // leak all neurons once per time step
            S_LEAK: begin
                for (i = 0; i < NH; i = i + 1) vh[i] <= vh[i] - (vh[i] >>> 4);
                for (i = 0; i < 16; i = i + 1) vo[i] <= vo[i] - (vo[i] >>> 4);
                pix <= 0;
                st <= S_SCAN;
            end

            // ---- layer 1: scan pixels, process only the ones that spike ----
            S_SCAN: begin
                ps <= xs32(ps);                                  // one random byte per pixel per step
                if (img[pix] > ps[7:0]) begin
                    row_reads <= row_reads + 1;
                    chunk <= 0;
                    st <= S_RD1;
                end
                else if (pix == NPIX-1) st <= S_UPD1;
                else pix <= pix + 1;
            end
            S_RD1: st <= S_AC1;                                  // w1_q is loaded at the end of this cycle
            S_AC1: begin
                for (l = 0; l < 8; l = l + 1)
                    vh[{chunk, 3'b000} + l] <= vh[{chunk, 3'b000} + l] + $signed(w1_rd[l*8 +: 8]);
                if (chunk == 7) begin
                    if (pix == NPIX-1) st <= S_UPD1;
                    else begin pix <= pix + 1; st <= S_SCAN; end
                end
                else begin chunk <= chunk + 1; st <= S_RD1; end
            end

            // hidden threshold / soft reset
            S_UPD1: begin
                for (i = 0; i < NH; i = i + 1) begin
                    if (vh[i] >= TH) begin hs[i] <= 1'b1; vh[i] <= vh[i] - TH; end
                    else hs[i] <= 1'b0;
                end
                h <= 0;
                st <= S_HSCAN;
            end

            // ---- layer 2: scan hidden neurons, process only the ones that spiked ----
            S_HSCAN: begin
                if (hs[h]) begin
                    row_reads <= row_reads + 1;
                    chunk <= 0;
                    st <= S_RD2;
                end
                else if (h == NH-1) st <= S_UPD2;
                else h <= h + 1;
            end
            S_RD2: st <= S_AC2;
            S_AC2: begin
                for (l = 0; l < 8; l = l + 1)
                    vo[{chunk[0], 3'b000} + l] <= vo[{chunk[0], 3'b000} + l] + $signed(w2_rd[l*8 +: 8]);
                if (chunk[0]) begin
                    if (h == NH-1) st <= S_UPD2;
                    else begin h <= h + 1; st <= S_HSCAN; end
                end
                else begin chunk <= 1; st <= S_RD2; end
            end

            // output threshold / soft reset / spike counters
            S_UPD2: begin
                for (i = 0; i < 16; i = i + 1) begin
                    if (vo[i] >= TH) begin
                        vo[i]  <= vo[i] - TH;
                        cnt[i] <= cnt[i] + 1'b1;
                    end
                end
                if (t == TSTEPS-1) st <= S_DONE;
                else begin t <= t + 1; st <= S_LEAK; end
            end

            S_DONE: begin
                busy <= 1'b0;
                done <= 1'b1;
                st   <= S_IDLE;
            end

            default: st <= S_IDLE;
            endcase
        end
    end

    // ---------------- result: argmax of the 10 output counters (first maximum wins) ----------------
    genvar c;
    generate
        for (c = 0; c < 10; c = c + 1) begin : CO
            assign counts[c*8 +: 8] = cnt[c];
        end
    endgenerate

    reg [3:0] best;
    reg [7:0] bc;
    integer a;
    always @* begin
        best = 4'd0;
        bc   = counts[7:0];
        for (a = 1; a < 10; a = a + 1)
            if (counts[a*8 +: 8] > bc) begin
                best = a;
                bc   = counts[a*8 +: 8];
            end
    end
    assign pred = best;

endmodule
