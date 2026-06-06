// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VsvsimTestbench.h for the primary calling header

#include "VsvsimTestbench__pch.h"

void VsvsimTestbench___024root___ctor_var_reset(VsvsimTestbench___024root* vlSelf);

VsvsimTestbench___024root::VsvsimTestbench___024root(VsvsimTestbench__Syms* symsp, const char* namep)
 {
    vlSymsp = symsp;
    vlNamep = strdup(namep);
    // Reset structure values
    VsvsimTestbench___024root___ctor_var_reset(this);
}

void VsvsimTestbench___024root::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

VsvsimTestbench___024root::~VsvsimTestbench___024root() {
    VL_DO_DANGLING(std::free(const_cast<char*>(vlNamep)), vlNamep);
}
