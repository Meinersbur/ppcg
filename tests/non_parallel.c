#include <stdlib.h>

int main()
{
	int a[1000], b[1000], s;

	for (int i = 0; i < 1000; ++i)
		a[i] = 1 + i;
#pragma scop
	b[0] = 1;
	s = b[0];
	for (int i = 1; i < 1000; ++i)
		b[i] = s + i;
#pragma endscop
	for (int i = 0; i < 1000; ++i)
		if (b[i] != a[i])
			return EXIT_FAILURE;

	return EXIT_SUCCESS;
}
