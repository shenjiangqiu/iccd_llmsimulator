PROJECT_ROOT := $(shell pwd)
DOCKER_IMAGE := openroad/orfs
DOCKER_RUN  := docker run --platform linux/amd64 --rm -v $(PROJECT_ROOT):/work -w /work $(DOCKER_IMAGE)

PDK_ROOT    := /OpenROAD-flow-scripts/flow/platforms/nangate45

.PHONY: all verilog synth area clean

all: verilog synth area

verilog:
	sbt "runMain asic.GenVecAdd2"

synth: verilog
	mkdir -p build
	$(DOCKER_RUN) bash -c "export PDK_LIB=$(PDK_ROOT)/lib/NangateOpenCellLibrary_typical.lib && /usr/local/bin/yosys flow/synth.ys"

area: synth
	$(DOCKER_RUN) /OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad -no_init -exit flow/openroad_area.tcl

clean:
	rm -rf gen-verilog build
