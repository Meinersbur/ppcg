#define KUNZH_TEST 1

#if defined KUNZH_TEST

#define PATH_MAX 1024

#if defined PATH_MAX
# define LT_PATHMAX PATH_MAX
#elif defined MAXPATHLEN
# define LT_PATHMAX MAXPATHLEN
#else
# define LT_PATHMAX 1024
#endif

#endif