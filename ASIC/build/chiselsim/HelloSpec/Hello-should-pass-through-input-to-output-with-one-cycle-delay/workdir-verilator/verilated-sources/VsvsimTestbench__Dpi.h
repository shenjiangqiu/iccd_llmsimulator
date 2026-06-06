// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Prototypes for DPI import and export functions.
//
// Verilator includes this file in all generated .cpp files that use DPI functions.
// Manually include this file where DPI .c import functions are declared to ensure
// the C functions match the expectations of the DPI imports.

#ifndef VERILATED_VSVSIMTESTBENCH__DPI_H_
#define VERILATED_VSVSIMTESTBENCH__DPI_H_  // guard

#include "svdpi.h"

#ifdef __cplusplus
extern "C" {
#endif


    // DPI EXPORTS
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:19:17
    extern void getBitWidthImpl_clock(int* value);
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:65:17
    extern void getBitWidthImpl_io_in(int* value);
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:53:17
    extern void getBitWidthImpl_io_out(int* value);
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:36:17
    extern void getBitWidthImpl_reset(int* value);
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:29:17
    extern void getBitsImpl_clock(svBitVecVal* value_clock);
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:75:17
    extern void getBitsImpl_io_in(svBitVecVal* value_io_in);
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:58:17
    extern void getBitsImpl_io_out(svBitVecVal* value_io_out);
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:46:17
    extern void getBitsImpl_reset(svBitVecVal* value_reset);
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:24:17
    extern void setBitsImpl_clock(const svBitVecVal* value_clock);
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:70:17
    extern void setBitsImpl_io_in(const svBitVecVal* value_io_in);
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:41:17
    extern void setBitsImpl_reset(const svBitVecVal* value_reset);
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:148:17
    extern void simulation_disableTrace(int* success);
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:133:17
    extern void simulation_enableTrace(int* success);
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:109:17
    extern void simulation_initializeTrace(const char* traceFilePath);

    // DPI IMPORTS
    // DPI import at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:14:40
    extern void initTestBenchScope();
    // DPI import at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:103:32
    extern void run_simulation(int timesteps, int* done);
    // DPI import at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:81:31
    extern int simulation_body();
    // DPI import at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:91:31
    extern int simulation_final();

#ifdef __cplusplus
}
#endif

#endif  // guard
