#ifndef _CUDA_COMMON_H_
#define _CUDA_COMMON_H_

#include <stdio.h>

struct cuda_info {
	FILE *host_c;
	FILE *kernel_cu;
	FILE *kernel_h;
};

struct ppcg_options;

void cuda_open_files(struct cuda_info *info, struct ppcg_options *options, const char *input, const char *output);
void cuda_close_files(struct cuda_info *info, struct ppcg_options *options);

#endif
