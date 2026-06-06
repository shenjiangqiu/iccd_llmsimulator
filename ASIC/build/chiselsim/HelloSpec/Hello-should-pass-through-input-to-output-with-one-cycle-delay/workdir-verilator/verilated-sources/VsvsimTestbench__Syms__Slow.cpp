// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table implementation internals

#include "VsvsimTestbench__pch.h"

void VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__getBitWidthImpl_clock_TOP(VsvsimTestbench__Syms* __restrict vlSymsp, IData/*31:0*/ &value);
void VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__getBitWidthImpl_io_in_TOP(VsvsimTestbench__Syms* __restrict vlSymsp, IData/*31:0*/ &value);
void VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__getBitWidthImpl_io_out_TOP(VsvsimTestbench__Syms* __restrict vlSymsp, IData/*31:0*/ &value);
void VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__getBitWidthImpl_reset_TOP(VsvsimTestbench__Syms* __restrict vlSymsp, IData/*31:0*/ &value);
void VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__getBitsImpl_clock_TOP(VsvsimTestbench__Syms* __restrict vlSymsp, CData/*0:0*/ &value_clock);
void VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__getBitsImpl_io_in_TOP(VsvsimTestbench__Syms* __restrict vlSymsp, CData/*7:0*/ &value_io_in);
void VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__getBitsImpl_io_out_TOP(VsvsimTestbench__Syms* __restrict vlSymsp, CData/*7:0*/ &value_io_out);
void VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__getBitsImpl_reset_TOP(VsvsimTestbench__Syms* __restrict vlSymsp, CData/*0:0*/ &value_reset);
void VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__setBitsImpl_clock_TOP(VsvsimTestbench__Syms* __restrict vlSymsp, CData/*0:0*/ value_clock);
void VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__setBitsImpl_io_in_TOP(VsvsimTestbench__Syms* __restrict vlSymsp, CData/*7:0*/ value_io_in);
void VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__setBitsImpl_reset_TOP(VsvsimTestbench__Syms* __restrict vlSymsp, CData/*0:0*/ value_reset);
void VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__simulation_disableTrace_TOP(VsvsimTestbench__Syms* __restrict vlSymsp, IData/*31:0*/ &success);
void VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__simulation_enableTrace_TOP(VsvsimTestbench__Syms* __restrict vlSymsp, IData/*31:0*/ &success);
void VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__simulation_initializeTrace_TOP(VsvsimTestbench__Syms* __restrict vlSymsp, std::string traceFilePath);

VsvsimTestbench__Syms::VsvsimTestbench__Syms(VerilatedContext* contextp, const char* namep, VsvsimTestbench* modelp)
    : VerilatedSyms{contextp}
    // Setup internal state of the Syms class
    , __Vm_modelp{modelp}
    // Setup top module instance
    , TOP{this, namep}
{
    // Check resources
    Verilated::stackCheck(393);
    // Setup sub module instances
    // Configure time unit / time precision
    _vm_contextp__->timeunit(-9);
    _vm_contextp__->timeprecision(-10);
    // Setup each module's pointers to their submodules
    // Setup each module's pointer back to symbol table (for public functions)
    TOP.__Vconfigure(true);
    // Setup scopes
    __Vscopep_svsimTestbench = new VerilatedScope{this, "svsimTestbench", "svsimTestbench", "<null>", -9, VerilatedScope::SCOPE_OTHER};
    // Setup export functions - final: 0
    __Vscopep_svsimTestbench->exportInsert(0, "getBitWidthImpl_clock", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__getBitWidthImpl_clock_TOP));
    __Vscopep_svsimTestbench->exportInsert(0, "getBitWidthImpl_io_in", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__getBitWidthImpl_io_in_TOP));
    __Vscopep_svsimTestbench->exportInsert(0, "getBitWidthImpl_io_out", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__getBitWidthImpl_io_out_TOP));
    __Vscopep_svsimTestbench->exportInsert(0, "getBitWidthImpl_reset", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__getBitWidthImpl_reset_TOP));
    __Vscopep_svsimTestbench->exportInsert(0, "getBitsImpl_clock", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__getBitsImpl_clock_TOP));
    __Vscopep_svsimTestbench->exportInsert(0, "getBitsImpl_io_in", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__getBitsImpl_io_in_TOP));
    __Vscopep_svsimTestbench->exportInsert(0, "getBitsImpl_io_out", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__getBitsImpl_io_out_TOP));
    __Vscopep_svsimTestbench->exportInsert(0, "getBitsImpl_reset", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__getBitsImpl_reset_TOP));
    __Vscopep_svsimTestbench->exportInsert(0, "setBitsImpl_clock", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__setBitsImpl_clock_TOP));
    __Vscopep_svsimTestbench->exportInsert(0, "setBitsImpl_io_in", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__setBitsImpl_io_in_TOP));
    __Vscopep_svsimTestbench->exportInsert(0, "setBitsImpl_reset", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__setBitsImpl_reset_TOP));
    __Vscopep_svsimTestbench->exportInsert(0, "simulation_disableTrace", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__simulation_disableTrace_TOP));
    __Vscopep_svsimTestbench->exportInsert(0, "simulation_enableTrace", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__simulation_enableTrace_TOP));
    __Vscopep_svsimTestbench->exportInsert(0, "simulation_initializeTrace", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__simulation_initializeTrace_TOP));
    // Setup export functions - final: 1
    __Vscopep_svsimTestbench->exportInsert(1, "getBitWidthImpl_clock", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__getBitWidthImpl_clock_TOP));
    __Vscopep_svsimTestbench->exportInsert(1, "getBitWidthImpl_io_in", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__getBitWidthImpl_io_in_TOP));
    __Vscopep_svsimTestbench->exportInsert(1, "getBitWidthImpl_io_out", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__getBitWidthImpl_io_out_TOP));
    __Vscopep_svsimTestbench->exportInsert(1, "getBitWidthImpl_reset", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__getBitWidthImpl_reset_TOP));
    __Vscopep_svsimTestbench->exportInsert(1, "getBitsImpl_clock", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__getBitsImpl_clock_TOP));
    __Vscopep_svsimTestbench->exportInsert(1, "getBitsImpl_io_in", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__getBitsImpl_io_in_TOP));
    __Vscopep_svsimTestbench->exportInsert(1, "getBitsImpl_io_out", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__getBitsImpl_io_out_TOP));
    __Vscopep_svsimTestbench->exportInsert(1, "getBitsImpl_reset", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__getBitsImpl_reset_TOP));
    __Vscopep_svsimTestbench->exportInsert(1, "setBitsImpl_clock", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__setBitsImpl_clock_TOP));
    __Vscopep_svsimTestbench->exportInsert(1, "setBitsImpl_io_in", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__setBitsImpl_io_in_TOP));
    __Vscopep_svsimTestbench->exportInsert(1, "setBitsImpl_reset", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__setBitsImpl_reset_TOP));
    __Vscopep_svsimTestbench->exportInsert(1, "simulation_disableTrace", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__simulation_disableTrace_TOP));
    __Vscopep_svsimTestbench->exportInsert(1, "simulation_enableTrace", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__simulation_enableTrace_TOP));
    __Vscopep_svsimTestbench->exportInsert(1, "simulation_initializeTrace", (void*)(&VsvsimTestbench___024root____Vdpiexp_svsimTestbench__DOT__simulation_initializeTrace_TOP));
}

VsvsimTestbench__Syms::~VsvsimTestbench__Syms() {
    // Tear down scopes
    VL_DO_CLEAR(delete __Vscopep_svsimTestbench, __Vscopep_svsimTestbench = nullptr);
    // Tear down sub module instances
}
