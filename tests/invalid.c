#include <stdlib.h>

int main()
{
#pragma scop
	int A[100];
	for (int i = 0; i < 100; ++i)
		A[i-1] = 0;
#pragma endscop
	return EXIT_SUCCESS;
}
