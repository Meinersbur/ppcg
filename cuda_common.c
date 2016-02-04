/*
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 */

#include <ctype.h>
#include <limits.h>
#include <string.h>

#include "cuda_common.h"
#include "ppcg.h"

/* Open the host .cu file and the kernel .hu and .cu files for writing.
 * Add the necessary includes.
 */
void cuda_open_files(struct cuda_info *info, const char *input, const char *output)
{
    char name[PATH_MAX];
    int len;

	if (output) {
		const char *ext;

		ext = strrchr(output, '.');
		len = ext ? ext - output : strlen(output);
		memcpy(name, output, len);

		info->host_c = fopen(output, "w");
	} else {
		len = ppcg_extract_base_name(name, input);

		strcpy(name + len, "_host.c");
		info->host_c = fopen(name, "w");
	}

    strcpy(name + len, "_kernel.cu");
    info->kernel_cu = fopen(name, "w");

    strcpy(name + len, "_kernel.h");
    info->kernel_h = fopen(name, "w");

	fprintf(info->host_c, "\n");
	fprintf(info->host_c, "#include \"%s\"\n\n", name);

    fprintf(info->kernel_cu, "#include \"%s\"\n\n", name);
	fprintf(info->kernel_cu, "#include <cuda.h>\n");
    fprintf(info->kernel_cu, "#include <assert.h>\n");
    fprintf(info->kernel_cu, "#include <stdio.h>\n");

	//TODO: Header guard
    fprintf(info->kernel_h, "#ifdef __cplusplus\n");
    fprintf(info->kernel_h, "extern \"C\" {\n");
    fprintf(info->kernel_h, "#endif\n");
}

/* Close all output files.
 */
void cuda_close_files(struct cuda_info *info)
{
    fprintf(info->kernel_h, "\n");
    fprintf(info->kernel_h, "#ifdef __cplusplus\n");
    fprintf(info->kernel_h, "} /* extern \"C\"*/\n");
    fprintf(info->kernel_h, "#endif\n");

    fclose(info->kernel_cu);
    fclose(info->kernel_h);
    fclose(info->host_c);
}
