#include <stdlib.h>

/* Check that strides in the statement instance set
 * are handled correctly.
 */
int main()
{
	int a[1000], b[1000];

	for (int i = 0; i < 1000; ++i)
		a[i] = i;
#pragma scop
	for (int i = 0; i < 3*1000; i += 3)
		b[i/3] = a[i/3];
#pragma endscop
	for (int i = 0; i < 1000; ++i)
		if (b[i] != a[i])
			return EXIT_FAILURE;

	return EXIT_SUCCESS;
}
