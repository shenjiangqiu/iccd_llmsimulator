package asic

import chisel3._

class Hello extends Module {
  val io = IO(new Bundle {
    val in  = Input(UInt(8.W))
    val out = Output(UInt(8.W))
  })

  val reg = RegInit(0.U(8.W))
  reg := io.in
  io.out := reg
}
