#include <stdlib.h>

struct S {
	int v;
};

int main()
{
	int a[1000], s;
	struct S b[1000];

	for (int i = 0; i < 1000; ++i)
		a[i] = 1 + i;
#pragma scop
	b[0].v = 1;
	s = b[0].v;
	for (int i = 1; i < 1000; ++i)
		b[i].v = s + i;
#pragma endscop
	for (int i = 0; i < 1000; ++i)
		if (b[i].v != a[i])
			return EXIT_FAILURE;

	return EXIT_SUCCESS;
}
