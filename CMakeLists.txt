list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
cmake_minimum_required(VERSION 3.2)

project("ppcg" C CXX)
set(PPCG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PPCG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
configure_file(${PPCG_SOURCE_DIR}/gitversion.h.cmake ${PPCG_BINARY_DIR}/include/gitversion.h)

add_subdirectory(isl)
add_subdirectory(pet)

set(ppcg_sources
	cpu.c
	cuda.c
	opencl.c
	cuda_common.c
	gpu.c
	gpu_array_tile.c
	gpu_group.c
	gpu_print.c
	gpu_tree.c
	schedule.c
	ppcg_options.c
	ppcg.c
	print.c
	version.c
)

add_executable(ppcg ${ppcg_sources})
target_include_directories(ppcg PRIVATE "${PPCG_BINARY_DIR}/include")
target_link_libraries(ppcg pet isl)
