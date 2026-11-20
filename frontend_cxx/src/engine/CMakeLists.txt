# fe_engine: EngineCore client + output processor (owner E). See DESIGN.md and tools/SEAM.md.
set(FE_ENGINE_SRCS mp.cpp json_bridge.cpp sampling.cpp protocol.cpp seam.cpp client.cpp detok.cpp
    output_processor.cpp metrics.cpp engine.cpp)
set(_srcs "")
foreach(s ${FE_ENGINE_SRCS})
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${s})
    list(APPEND _srcs ${s})
  endif()
endforeach()
add_library(fe_engine STATIC ${_srcs})
target_include_directories(fe_engine PUBLIC ${CMAKE_SOURCE_DIR}/src ${TP}/include)
target_compile_definitions(fe_engine PUBLIC ZMQ_STATIC)
target_compile_options(fe_engine PRIVATE -Wall -Wextra -Wno-unused-parameter)
find_package(Threads REQUIRED)
target_link_libraries(fe_engine PUBLIC ${TP}/lib64/libzmq.a Threads::Threads)
if(TARGET fe_common)
  target_link_libraries(fe_engine PUBLIC fe_common)
  target_compile_definitions(fe_engine PUBLIC FE_HAVE_COMMON=1)
endif()
# fe_tokenizer (owner B) provides fe::DecodeStream; link it when present (see detok.h).
if(TARGET fe_tokenizer)
  target_link_libraries(fe_engine PUBLIC fe_tokenizer)
  target_compile_definitions(fe_engine PUBLIC FE_HAVE_TOKENIZER=1)
endif()

# ---- tests (ctest -L engine)
set(FE_ENGINE_TESTS
  test_mp
  test_request_golden
  test_output_golden
  test_client_seam
  test_async_engine
)
foreach(t ${FE_ENGINE_TESTS})
  if(EXISTS ${CMAKE_SOURCE_DIR}/tests/engine/${t}.cpp)
    add_executable(engine_${t} ${CMAKE_SOURCE_DIR}/tests/engine/${t}.cpp)
    target_link_libraries(engine_${t} PRIVATE fe_engine)
    target_compile_definitions(engine_${t} PRIVATE FE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  endif()
endforeach()
if(TARGET engine_test_mp)
  add_test(NAME engine_mp COMMAND engine_test_mp)
endif()
if(TARGET engine_test_request_golden)
  add_test(NAME engine_request_golden COMMAND engine_test_request_golden)
endif()
if(TARGET engine_test_output_golden)
  add_test(NAME engine_output_golden COMMAND engine_test_output_golden)
  set_tests_properties(engine_output_golden PROPERTIES SKIP_RETURN_CODE 77)
endif()
if(TARGET engine_test_client_seam)
  # Needs the venv python (vLLM's msgspec structs in tools/mock_engine.py); ~40 s for the vLLM import.
  set(FE_VENV_PY "$DS41H_HOME/venv/bin/python" CACHE FILEPATH "python with vLLM")
  add_test(NAME engine_client_seam
           COMMAND engine_test_client_seam --python ${FE_VENV_PY} --mock ${CMAKE_SOURCE_DIR}/tools/mock_engine.py)
  set_tests_properties(engine_client_seam PROPERTIES TIMEOUT 300 LABELS "engine;slow")
endif()
if(TARGET engine_test_async_engine)
  add_test(NAME engine_async_engine
           COMMAND engine_test_async_engine --python ${FE_VENV_PY} --mock ${CMAKE_SOURCE_DIR}/tools/mock_engine.py)
  set_tests_properties(engine_async_engine PROPERTIES TIMEOUT 300 LABELS "engine;slow")
endif()
if(TARGET engine_test_output_golden AND EXISTS ${CMAKE_SOURCE_DIR}/tools/golden/engine_outputs_private.mpk)
  add_test(NAME engine_output_golden_private
           COMMAND engine_test_output_golden ${CMAKE_SOURCE_DIR}/tools/golden/engine_outputs_private.mpk)
  set_tests_properties(engine_output_golden_private PROPERTIES LABELS "engine;private" SKIP_RETURN_CODE 77)
endif()
foreach(t engine_mp engine_request_golden engine_output_golden)
  if(TEST ${t})
    set_tests_properties(${t} PROPERTIES LABELS engine)
  endif()
endforeach()
