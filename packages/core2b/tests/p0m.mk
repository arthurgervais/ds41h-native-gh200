# p0m.mk -- torch-free build of the mixed-step P0 builders and their golden replay (c1-overlap, 15 Nov 2026).
#   make -f tests/p0m.mk            (from fpr/core; CXX=g++-13 on the Clariden login node)
#   ./build_p0m/replay_mixed <out>/<variant>/<workload> [...]     (tests/record_mixed_cpu.py --out <out> --variant <v>)
#   make -f tests/p0m.mk nw         the beta0b NativeWorker over the same goldens with fakes at the device boundary:
#   ./build_p0m/nw_mixed_test build_p0m/libnwfake.so [--drop X] [--mutant N] <out>/<variant>/<workload> [...]
CXX ?= c++
CXXFLAGS ?= -O2 -std=c++17 -ffp-contract=off -Wall -Wextra -Wno-unused-parameter
B := build_p0m
SRC := src/p0/mixed_build.cpp src/p0/inputs_build.cpp src/core2b/mixed_exec.cpp src/core2b/mixed_issue_nd.cpp src/core2b/span_program.cpp src/tables/tables.cpp src/kv/kv_manager.cpp
HDR := src/p0/mixed_build.h src/p0/inputs_build.h src/core2b/mixed_exec.h src/core2b/mixed_issue_nd.h src/core2b/span_program.h include/common.h include/tables.h include/kv.h include/scheduler.h tests/json_min.h tests/mixed_golden.h
NWSRC := src/core2b/native_worker.cc src/core2b/native_mixed.cc src/core2b/mixed_owner.cpp src/core2b/cabi_libs.cpp src/core2b/driver_bridge.cpp src/p0/moe_decide.cpp \
  src/core2b/owner_p1.cpp src/p1/gt_build.cpp src/p1/pf_build.cpp src/p1/steps_build.cpp src/p1/cp_build.cpp
NWHDR := src/core2b/native_worker.h src/core2b/native_mixed.h src/core2b/mixed_owner.h src/core2b/cabi_libs.h src/core2b/driver_bridge.h src/core2b/nd_api.h src/core2b/engine_handover.h src/p0/moe_decide.h tests/nw_p1_fake.h src/p1/p1_cmirror.h src/p1/gt_build.h src/p1/pf_build.h src/p1/steps_build.h src/p1/cp_build.h
all: $(B)/replay_mixed
nw: $(B)/nw_mixed_test $(B)/libnwfake.so
$(B)/replay_mixed: tests/replay_mixed.cpp $(SRC) $(HDR)
	@mkdir -p $(B)
	$(CXX) $(CXXFLAGS) -Iinclude -o $@ $< $(SRC)
$(B)/nw_mixed_test: tests/nw_mixed_test.cpp $(SRC) $(NWSRC) $(HDR) $(NWHDR)
	@mkdir -p $(B)
	$(CXX) $(CXXFLAGS) -Iinclude -Isrc/core2b -o $@ $< $(SRC) $(NWSRC) -ldl
$(B)/libnwfake.so: tests/nw_fake_dev.cpp tests/nw_p1_fake_dev.cpp src/p1/p1_cmirror.h
	@mkdir -p $(B)
	$(CXX) $(CXXFLAGS) -fPIC -shared -o $@ $< tests/nw_p1_fake_dev.cpp
clean:
	rm -rf $(B)
