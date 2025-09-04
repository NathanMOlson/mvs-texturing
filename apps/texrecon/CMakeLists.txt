file (GLOB HEADERS "*.h")

set(BIN texrecon)

add_executable(${BIN} texrecon.cpp arguments.cpp)
set_property(TARGET ${BIN} PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
include_directories(${TBB_INCLUDE_DIRS})
target_link_libraries(${BIN} tex ${TBB_LIBRARIES} ${MVE_MVE_LIBRARY} ${MVE_UTIL_LIBRARY})
install(TARGETS ${BIN} RUNTIME DESTINATION bin)


include_directories(SYSTEM "/usr/include/opencv4")
add_executable(texrecon2 texrecon2.cpp arguments.cpp)
set_property(TARGET texrecon2 PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
include_directories(${TBB_INCLUDE_DIRS})
target_link_libraries(texrecon2 tex ${TBB_LIBRARIES} ${MVE_MVE_LIBRARY} ${MVE_UTIL_LIBRARY})
install(TARGETS texrecon2 RUNTIME DESTINATION bin)
