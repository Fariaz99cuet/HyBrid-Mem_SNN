`timescale 1ns/1ps
// Testbench for snn_accel.
//  Config 0 (no errors) must match the Python integer model exactly (all 10 output counts).
//  The other configs inject read errors into the weight memory and report accuracy.
module tb_snn_accel;
    parameter NTEST = 100;                 // must match NTEST in train_mnist_export.py

    reg clk = 0;
    always #5 clk = ~clk;                  // 100 MHz

    reg rst = 1, start = 0, img_we = 0;
    reg [7:0] img_addr = 0, img_data = 0;
    reg [7:0] noisy_mask = 0;
    reg [3:0] ber_sel = 0;
    wire busy, done;
    wire [3:0]  pred;
    wire [79:0] counts;
    wire [31:0] cycles, row_reads;

    snn_accel dut (.clk(clk), .rst(rst),
                   .img_we(img_we), .img_addr(img_addr), .img_data(img_data),
                   .start(start), .noisy_mask(noisy_mask), .ber_sel(ber_sel),
                   .busy(busy), .done(done), .pred(pred), .counts(counts),
                   .cycles(cycles), .row_reads(row_reads));

    reg [7:0] timg    [0:NTEST*196-1];
    reg [7:0] exp_cnt [0:NTEST*10-1];
    reg [3:0] lab     [0:NTEST-1];
    reg [31:0] exp_cyc [0:NTEST-1];        // cycle count predicted by the Python cycle model

    `include "test_data.vh"                // fills timg / exp_cnt / lab / exp_cyc / cfg_mask / cfg_ber

    integer n, p, k, cfg;
    integer correct, exact, cyc_ok, sum_cycles, sum_rows;
    reg     match;
    reg     clean_pass;

    initial begin
        if (NTEST_PY != NTEST) $display("WARNING: NTEST (%0d) != NTEST in Python (%0d)", NTEST, NTEST_PY);
        clean_pass = 0;
        repeat (4) @(posedge clk);
        @(negedge clk); rst = 0;
        repeat (2) @(negedge clk);

        $display("=====================================================================");
        $display(" config | noisy bits | bit-error prob | accuracy | cycles/img | rows/img");
        $display("=====================================================================");

        for (cfg = 0; cfg < NCFG; cfg = cfg + 1) begin
            noisy_mask = cfg_mask[cfg];
            ber_sel    = cfg_ber[cfg];
            correct = 0; exact = 0; cyc_ok = 0; sum_cycles = 0; sum_rows = 0;

            for (n = 0; n < NTEST; n = n + 1) begin
                // load the image
                for (p = 0; p < 196; p = p + 1) begin
                    @(negedge clk);
                    img_we = 1; img_addr = p; img_data = timg[n*196 + p];
                end
                @(negedge clk); img_we = 0;

                // run
                start = 1; @(negedge clk); start = 0;
                @(negedge clk); @(negedge clk);
                wait (done);
                @(negedge clk);

                if (pred === lab[n]) correct = correct + 1;
                match = 1;
                for (k = 0; k < 10; k = k + 1)
                    if (counts[k*8 +: 8] !== exp_cnt[n*10 + k]) match = 0;
                if (match) exact = exact + 1;
                if (cycles === exp_cyc[n]) cyc_ok = cyc_ok + 1;
                sum_cycles = sum_cycles + cycles;
                sum_rows   = sum_rows + row_reads;

                if (cfg == 0 && !match)
                    $display("  image %0d: output counts differ from Python model", n);
                if (cfg == 0 && cycles !== exp_cyc[n])
                    $display("  image %0d: cycles %0d, Python cycle model says %0d", n, cycles, exp_cyc[n]);
            end

            $display("   %0d     |    0x%02h    |     1/%0d%s    |  %3d/%0d   |   %6d   |  %5d",
                     cfg, noisy_mask, (ber_sel == 0) ? 0 : (1 << ber_sel),
                     (ber_sel == 0) ? "(off)" : "     ",
                     correct, NTEST, sum_cycles / NTEST, sum_rows / NTEST);

            if (cfg == 0) begin
                $display("   -> exact match of all 10 output counts with Python model: %0d / %0d", exact, NTEST);
                $display("   -> cycle count equals Python cycle model              : %0d / %0d", cyc_ok, NTEST);
                clean_pass = (exact == NTEST) && (cyc_ok == NTEST);
            end
        end

        $display("=====================================================================");
        $display("dense (non event-driven) design would fetch %0d weight rows per image", TB_TSTEPS * (196 + 64));
        if (clean_pass) $display("RESULT: PASS (hardware == Python model without memory errors)");
        else            $display("RESULT: FAIL");
        $finish;
    end

    // watchdog
    initial begin
        #2000000000;
        $display("TIMEOUT");
        $finish;
    end
endmodule
