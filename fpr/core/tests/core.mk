# fpr/core/tests/core.mk -- c1-overlap's components (src/core, src/grammar, src/v1wire) and their CPU tests, torch-free
# (c1-overlap, 14 Nov 2026). Build rule of the stage-2 core: -O2 -std=c++17 -ffp-contract=off.
#   make -f core.mk -C fpr/core/tests OUT=$DS41H_HOME/tmp/fprcore/build XGR=$DS41H_HOME/tmp/xgr-0.2.8 \
#        XGRLIB=$DS41H_HOME/tmp/fprcore/xgr/libxgrammar.a CXX=$DS41H_HOME/bin/g++
# xgrammar: the installed wheel's version (0.2.8, commit 97787376f) built from source as a static library
# ($DS41H_HOME/tmp/fprcore/build_xgr.sh: cpp/*.cc without the Python bindings, the upstream flags without -Werror).
CXX ?= g++
OUT ?= build
XGR ?= xgrammar
XGRLIB ?= $(OUT)/libxgrammar.a
SRC := ..
INC ?= $(SRC)/include      # the component headers (the build snapshot's, see README_core.md)
CXXFLAGS := -O2 -std=c++17 -ffp-contract=off -fPIC -Wall -Wextra -Wno-unused-parameter -pthread
XINC := -I$(XGR)/include -isystem $(XGR)/3rdparty/picojson -isystem $(XGR)/3rdparty/dlpack/include -DXGRAMMAR_ENABLE_CPPTRACE=0

WIRE := $(OUT)/v1wire.o $(OUT)/pyfloat.o
GRAM := $(OUT)/grammar.o
CORE := $(OUT)/lifecycle.o $(OUT)/core.o

PY ?= python3
PYINC ?= $(shell $(PY) -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PYBIND ?= $(shell $(PY) -c "import pybind11; print(pybind11.get_include())")
PYEXT ?= $(shell $(PY) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

all: $(OUT)/test_wire $(OUT)/test_grammar $(OUT)/replay_core $(OUT)/libfprcore.a

# stage 2a: the pybind module (make -f core.mk pymod PY=$DS41H_HOME/venv/bin/python)
pymod: $(OUT)/fpr_core_py$(PYEXT)
# (+ the core's block hasher and c1-sync's tables.cpp: CoreX builds batches over KV views for the row tokens)
$(OUT)/fpr_core_py$(PYEXT): $(SRC)/src/core/py_core.cc $(OUT)/lifecycle.o $(SRC)/src/core/blockhash.cc $(SRC)/src/tables/tables.cpp $(WIRE) $(GRAM)
	$(CXX) $(CXXFLAGS) -shared -I$(INC) -I$(SRC)/src/core -isystem $(PYINC) -isystem $(PYBIND) $(XINC) $< $(SRC)/src/core/blockhash.cc $(SRC)/src/tables/tables.cpp $(OUT)/lifecycle.o $(GRAM) $(WIRE) $(XGRLIB) -o $@.new && mv $@.new $@   # (a running server keeps its mapped copy)

$(OUT):
	mkdir -p $(OUT)

$(OUT)/v1wire.o: $(SRC)/src/v1wire/v1wire.cc $(SRC)/src/v1wire/v1wire.h $(SRC)/src/v1wire/msgpack.h | $(OUT)
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(OUT)/pyfloat.o: $(SRC)/src/v1wire/pyfloat.cc | $(OUT)
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(OUT)/grammar.o: $(SRC)/src/grammar/grammar.cc $(SRC)/src/grammar/grammar.h | $(OUT)
	$(CXX) $(CXXFLAGS) $(XINC) -c $< -o $@
$(OUT)/lifecycle.o: $(SRC)/src/core/lifecycle.cc $(SRC)/src/core/lifecycle.h $(SRC)/src/grammar/grammar.h | $(OUT)
	$(CXX) $(CXXFLAGS) -I$(INC) -c $< -o $@
$(OUT)/core.o: $(SRC)/src/core/core.cc $(SRC)/include/core.h $(SRC)/src/core/lifecycle.h | $(OUT)
	$(CXX) $(CXXFLAGS) -I$(INC) -c $< -o $@

$(OUT)/libfprcore.a: $(WIRE) $(GRAM) $(CORE)
	rm -f $@ && ar rcs $@ $^

$(OUT)/test_wire: test_wire.cpp json.h $(WIRE)
	$(CXX) $(CXXFLAGS) $< $(WIRE) -o $@
$(OUT)/test_grammar: test_grammar.cpp json.h $(WIRE) $(GRAM)
	$(CXX) $(CXXFLAGS) $(XINC) $< $(GRAM) $(WIRE) $(XGRLIB) -o $@
$(OUT)/replay_core: replay_core.cpp json.h $(WIRE) $(GRAM) $(CORE)
	$(CXX) $(CXXFLAGS) -I$(INC) $(XINC) $< $(CORE) $(GRAM) $(WIRE) $(XGRLIB) -o $@

.PHONY: all pymod
