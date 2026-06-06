// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Implementation of DPI export functions
//
// Verilator compiles this file in when DPI functions are used.
// If you have multiple Verilated designs with the same DPI exported
// function names, you will get multiple definition link errors from here.
// This is an unfortunate result of the DPI specification.
// To solve this, either
//    1. Call VsvsimTestbench::{export_function} instead,
//       and do not even bother to compile this file
// or 2. Compile all __Dpi.cpp files in the same compiler run,
//       and #ifdefs already inserted here will sort everything out.

#include "VsvsimTestbench__Dpi.h"
#include "VsvsimTestbench.h"

#ifndef VL_DPIDECL_getBitWidthImpl_clock_
#define VL_DPIDECL_getBitWidthImpl_clock_
void getBitWidthImpl_clock(int* value) {
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:19:17
    return VsvsimTestbench::getBitWidthImpl_clock(value);
}
#endif

#ifndef VL_DPIDECL_getBitWidthImpl_io_in_
#define VL_DPIDECL_getBitWidthImpl_io_in_
void getBitWidthImpl_io_in(int* value) {
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:65:17
    return VsvsimTestbench::getBitWidthImpl_io_in(value);
}
#endif

#ifndef VL_DPIDECL_getBitWidthImpl_io_out_
#define VL_DPIDECL_getBitWidthImpl_io_out_
void getBitWidthImpl_io_out(int* value) {
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:53:17
    return VsvsimTestbench::getBitWidthImpl_io_out(value);
}
#endif

#ifndef VL_DPIDECL_getBitWidthImpl_reset_
#define VL_DPIDECL_getBitWidthImpl_reset_
void getBitWidthImpl_reset(int* value) {
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:36:17
    return VsvsimTestbench::getBitWidthImpl_reset(value);
}
#endif

#ifndef VL_DPIDECL_getBitsImpl_clock_
#define VL_DPIDECL_getBitsImpl_clock_
void getBitsImpl_clock(svBitVecVal* value_clock) {
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:29:17
    return VsvsimTestbench::getBitsImpl_clock(value_clock);
}
#endif

#ifndef VL_DPIDECL_getBitsImpl_io_in_
#define VL_DPIDECL_getBitsImpl_io_in_
void getBitsImpl_io_in(svBitVecVal* value_io_in) {
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:75:17
    return VsvsimTestbench::getBitsImpl_io_in(value_io_in);
}
#endif

#ifndef VL_DPIDECL_getBitsImpl_io_out_
#define VL_DPIDECL_getBitsImpl_io_out_
void getBitsImpl_io_out(svBitVecVal* value_io_out) {
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:58:17
    return VsvsimTestbench::getBitsImpl_io_out(value_io_out);
}
#endif

#ifndef VL_DPIDECL_getBitsImpl_reset_
#define VL_DPIDECL_getBitsImpl_reset_
void getBitsImpl_reset(svBitVecVal* value_reset) {
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:46:17
    return VsvsimTestbench::getBitsImpl_reset(value_reset);
}
#endif

#ifndef VL_DPIDECL_setBitsImpl_clock_
#define VL_DPIDECL_setBitsImpl_clock_
void setBitsImpl_clock(const svBitVecVal* value_clock) {
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:24:17
    return VsvsimTestbench::setBitsImpl_clock(value_clock);
}
#endif

#ifndef VL_DPIDECL_setBitsImpl_io_in_
#define VL_DPIDECL_setBitsImpl_io_in_
void setBitsImpl_io_in(const svBitVecVal* value_io_in) {
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:70:17
    return VsvsimTestbench::setBitsImpl_io_in(value_io_in);
}
#endif

#ifndef VL_DPIDECL_setBitsImpl_reset_
#define VL_DPIDECL_setBitsImpl_reset_
void setBitsImpl_reset(const svBitVecVal* value_reset) {
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:41:17
    return VsvsimTestbench::setBitsImpl_reset(value_reset);
}
#endif

#ifndef VL_DPIDECL_simulation_disableTrace_
#define VL_DPIDECL_simulation_disableTrace_
void simulation_disableTrace(int* success) {
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:148:17
    return VsvsimTestbench::simulation_disableTrace(success);
}
#endif

#ifndef VL_DPIDECL_simulation_enableTrace_
#define VL_DPIDECL_simulation_enableTrace_
void simulation_enableTrace(int* success) {
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:133:17
    return VsvsimTestbench::simulation_enableTrace(success);
}
#endif

#ifndef VL_DPIDECL_simulation_initializeTrace_
#define VL_DPIDECL_simulation_initializeTrace_
void simulation_initializeTrace(const char* traceFilePath) {
    // DPI export at /Users/sjq/git/LLMSimulator/ASIC/build/chiselsim/HelloSpec/Hello-should-pass-through-input-to-output-with-one-cycle-delay/workdir-verilator/../generated-sources/testbench.sv:109:17
    return VsvsimTestbench::simulation_initializeTrace(traceFilePath);
}
#endif

