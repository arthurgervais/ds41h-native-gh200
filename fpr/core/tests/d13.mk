# d13.mk -- torch-free build of the D13 / HSD-prepare port and its replays (C1 tail session, 14 Nov 2026).
#   make -f tests/d13.mk            (from fpr/core; CXX=g++-13 on the Clariden login node)
#   ./build_d13/replay_d13 <trace dir>   ./build_d13/replay_hsd <trace dir>
CXX ?= c++
CXXFLAGS ?= -O2 -std=c++17 -ffp-contract=off -Wall -Wextra -Wno-unused-parameter
B := build_d13
SRC := src/d13/d13_host.cpp src/d13/hsd_prep.cpp src/tables/tables.cpp   # (tables.cpp: c1-sync's TableBuilder)
HDR := include/d13.h include/common.h include/tables.h include/kv.h src/d13/d13_host.h src/d13/hsd_prep.h tests/gtrace_json.h
all: $(B)/replay_d13 $(B)/replay_hsd
$(B)/%: tests/%.cpp $(SRC) $(HDR)
	@mkdir -p $(B)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SRC)
clean:
	rm -rf $(B)
