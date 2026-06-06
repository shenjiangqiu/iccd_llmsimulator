package asic

import chisel3._
import _root_.circt.stage.ChiselStage

class VecAdd2 extends Module {
  val nLanes = 32
  val inWidth = 2
  val outWidth = inWidth + 1

  val io = IO(new Bundle {
    val vecA   = Input(Vec(nLanes, UInt(inWidth.W)))
    val vecB   = Input(Vec(nLanes, UInt(inWidth.W)))
    val vecSum = Output(Vec(nLanes, UInt(outWidth.W)))
  })

  for (i <- 0 until nLanes) {
    io.vecSum(i) := io.vecA(i) + io.vecB(i)
  }
}

object GenVecAdd2 extends App {
  ChiselStage.emitSystemVerilogFile(
    new VecAdd2,
    Array("--target-dir", "gen-verilog"),
    firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info", "-default-layer-specialization=enable")
  )
}
