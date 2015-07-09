# Check if OpenCL is available (headers and library)
AC_DEFUN([AX_CHECK_OPENCL], [
	AC_SUBST(HAVE_OPENCL)
	HAVE_OPENCL=no
	AC_CHECK_HEADER([CL/opencl.h], [
		AC_CHECK_LIB([OpenCL], [clGetPlatformIDs], [
			HAVE_OPENCL=yes
		])
	])
])
