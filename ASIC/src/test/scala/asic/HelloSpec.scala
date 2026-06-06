package asic

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.must.Matchers

class HelloSpec extends AnyFreeSpec with Matchers with ChiselSim {

  "Hello should pass through input to output with one cycle delay" in {
    simulate(new Hello) { dut =>
      dut.io.in.poke(42.U)
      dut.clock.step()
      dut.io.out.expect(42.U)
    }
  }
}
