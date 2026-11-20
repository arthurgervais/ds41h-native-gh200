# kv_tables.mk -- torch-free build of the KV manager / tables ports and their golden-trace replays (c1-sync, 14 Nov 2026).
#   make -f tests/kv_tables.mk            (from fpr/core; CXX=g++-13 on the Clariden login node)
#   ./build_kv/replay_kv <dir>/kv.jsonl         ./build_kv/replay_tables <dir>/tables.jsonl
# Record the traces: python3 tests/record_kv_tables.py --out DIR [test modules | drv:<workload> ...]
CXX ?= c++
CXXFLAGS ?= -O2 -std=c++17 -ffp-contract=off -Wall -Wextra -Wno-unused-parameter
B := build_kv
KV_SRC := src/kv/kv_manager.cpp
TB_SRC := src/tables/tables.cpp
HDR := include/common.h include/geometry.h include/kv.h include/tables.h include/scheduler.h src/kv/kv_manager.h \
       tests/replay_common.h tests/replay_kv_lines.h tests/gtrace_json.h
all: $(B)/replay_kv $(B)/replay_tables
$(B)/replay_kv: tests/replay_kv.cpp $(KV_SRC) $(HDR)
	@mkdir -p $(B)
	$(CXX) $(CXXFLAGS) -Iinclude -o $@ $< $(KV_SRC)
$(B)/replay_tables: tests/replay_tables.cpp $(TB_SRC) $(KV_SRC) $(HDR)
	@mkdir -p $(B)
	$(CXX) $(CXXFLAGS) -Iinclude -o $@ $< $(TB_SRC) $(KV_SRC)
clean:
	rm -rf $(B)
