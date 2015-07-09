list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
cmake_minimum_required(VERSION 3.2)

project("ppcg" C CXX)
set(PPCG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(PPCG_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

set(POLYBENCH_DIR "" CACHE PATH "Directory where polybench resides")

if (TARGET prl_opencl)
  get_target_property(PRL_INCLUDE_DIR prl_opencl INTERFACE_INCLUDE_DIRECTORIES)
  #TODO: static libs are put into ARCHIVE_OUTPUT_DIRECTORY
  get_target_property(PRL_LIBRARY_PATH prl_opencl LIBRARY_OUTPUT_DIRECTORY)
else ()
endif ()


## Platform introspection
# Determine version of ppcg
if (EXISTS "${PPCG_SOURCE_DIR}/GIT_HEAD_ID")
  # The source comes from a 'make dist' archive
  file(READ "${PPCG_SOURCE_DIR}/GIT_HEAD_ID" GIT_HEAD_ID)
  string(STRIP "${GIT_HEAD_ID}" GIT_HEAD_ID)
elseif (EXISTS "${PPCG_SOURCE_DIR}/gitversion.h")
  # The source directory is preconfigured
  file(READ "${PPCG_SOURCE_DIR}/gitversion.h" GITVERSION_H)
  string(REGEX REPLACE ".*\\\"([^\\\"]*)\\\".*" "\\1" GIT_HEAD_ID "${GITVERSION_H}")
elseif (EXISTS "${PPCG_SOURCE_DIR}/.git" AND "${PPCG_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
  # This is the GIT repository itself
  # GetGitRevisionDescription uses CMAKE_CURRENT_SOURCE_DIR to look for the .git directory, so it must refer to PET's source directory
  include(GetGitRevisionDescription)
  git_describe(GIT_HEAD_ID)
else ()
  # Unknown revision
  set(GIT_HEAD_ID "UNKNOWN")
endif ()
message(STATUS "PPCG version: ${GIT_HEAD_ID}")


configure_file("${PPCG_SOURCE_DIR}/gitversion.h.cmake" "${PPCG_BINARY_DIR}/include/gitversion.h")


set(EXEEXT)
set(GIT_HEAD_VERSION GIT_HEAD_ID)
set(CC cc)
set(srcdir "${PPCG_SOURCE_DIR}")
set(bindir "${PPCG_BINARY_DIR}")
set(HAVE_OPENCL yes)
set(HAVE_OPENMP yes)
configure_file("${PPCG_SOURCE_DIR}/polybench_test.sh.in" "${PPCG_BINARY_DIR}/polybench_test.sh" @ONLY)
file(COPY "${PPCG_BINARY_DIR}/polybench_test.sh"
  DESTINATION "${PPCG_BINARY_DIR}/bin"
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

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




enable_testing()
add_test(NAME polybench COMMAND "${PPCG_BINARY_DIR}/bin/polybench_test.sh")
