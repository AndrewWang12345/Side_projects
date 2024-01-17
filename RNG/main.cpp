#include <stdio.h>
#include <limits.h>

char randoms(float *randf, float min, float max)
{
    int retries= 10;
    unsigned long long rand64;

    while(retries--) {
        if ( __builtin_ia32_rdrand64_step(&rand64) ) {
            *randf= (float)rand64/ULONG_MAX*(max - min) + min;
            return 1;
        }
    }
    return 0;
}

int main()
{
    float randf;

    if ( randoms(&randf, -100.001, 100.001) ) printf("%f\n", randf);
    else printf("Failed to get a random value\n");
    return 0;
}
