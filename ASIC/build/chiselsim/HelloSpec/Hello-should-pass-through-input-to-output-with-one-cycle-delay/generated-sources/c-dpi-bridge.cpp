#include <stdint.h>

#ifdef SVSIM_ENABLE_VERILATOR_SUPPORT
#include "verilated-sources/VsvsimTestbench__Dpi.h"
#endif
#ifdef SVSIM_ENABLE_VCS_SUPPORT
#include "vc_hdrs.h"
#endif

extern "C" {
 svScope setScopeToTestBench();
void getBitWidth_clock(int* result) {
           svScope prev = setScopeToTestBench();
           getBitWidthImpl_clock(result);
           svSetScope(prev);
        }
void getBits_clock(svBitVecVal* result) {
           svScope prev = setScopeToTestBench();
           getBitsImpl_clock(result);
           svSetScope(prev);
        }
void setBits_clock(const svBitVecVal* data) {
           svScope prev = setScopeToTestBench();
           setBitsImpl_clock(data);
           svSetScope(prev);
        }
void getBitWidth_reset(int* result) {
           svScope prev = setScopeToTestBench();
           getBitWidthImpl_reset(result);
           svSetScope(prev);
        }
void getBits_reset(svBitVecVal* result) {
           svScope prev = setScopeToTestBench();
           getBitsImpl_reset(result);
           svSetScope(prev);
        }
void setBits_reset(const svBitVecVal* data) {
           svScope prev = setScopeToTestBench();
           setBitsImpl_reset(data);
           svSetScope(prev);
        }
void getBitWidth_io_out(int* result) {
           svScope prev = setScopeToTestBench();
           getBitWidthImpl_io_out(result);
           svSetScope(prev);
        }
void getBits_io_out(svBitVecVal* result) {
           svScope prev = setScopeToTestBench();
           getBitsImpl_io_out(result);
           svSetScope(prev);
        }
void getBitWidth_io_in(int* result) {
           svScope prev = setScopeToTestBench();
           getBitWidthImpl_io_in(result);
           svSetScope(prev);
        }
void getBits_io_in(svBitVecVal* result) {
           svScope prev = setScopeToTestBench();
           getBitsImpl_io_in(result);
           svSetScope(prev);
        }
void setBits_io_in(const svBitVecVal* data) {
           svScope prev = setScopeToTestBench();
           setBitsImpl_io_in(data);
           svSetScope(prev);
        }

int port_getter(int id, int *bitWidth, void (**getter)(uint8_t*)) {
  switch (id) {
    case 0: // clock
      getBitWidth_clock(bitWidth);
      *getter = (void(*)(uint8_t*))getBits_clock;
      return 0;
    case 1: // reset
      getBitWidth_reset(bitWidth);
      *getter = (void(*)(uint8_t*))getBits_reset;
      return 0;
    case 2: // io_out
      getBitWidth_io_out(bitWidth);
      *getter = (void(*)(uint8_t*))getBits_io_out;
      return 0;
    case 3: // io_in
      getBitWidth_io_in(bitWidth);
      *getter = (void(*)(uint8_t*))getBits_io_in;
      return 0;
    default:
      return -1;
  }
}

int port_setter(int id, int *bitWidth, void (**setter)(const uint8_t*)) {
  switch (id) {
    case 0: // clock
      getBitWidth_clock(bitWidth);
      *setter = (void(*)(const uint8_t*))setBits_clock;
      return 0;
    case 1: // reset
      getBitWidth_reset(bitWidth);
      *setter = (void(*)(const uint8_t*))setBits_reset;
      return 0;
    case 3: // io_in
      getBitWidth_io_in(bitWidth);
      *setter = (void(*)(const uint8_t*))setBits_io_in;
      return 0;
    default:
      return -1;
  }
}

} // extern "C"

