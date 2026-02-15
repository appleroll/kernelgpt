#include "kernelgpt/math_utils.h"
#include "math.h"

// Linear Congruential Generator
static unsigned long int next = 1;
void mysrand(unsigned int seed) { next = seed; }
int myrand(void) {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next / 65536) % 32768;
}

double random_uniform(void) { return (double)myrand() / 32768.0; }

double random_gauss(double mu, double sigma) {
    // Box-Muller transform
    double u1 = random_uniform();
    double u2 = random_uniform();
    if(u1 < 1e-6) u1 = 1e-6;
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return z0 * sigma + mu;
}

double cos(double x) {
    // Taylor series for cos(x) approx
    double term = 1;
    double sum = 1;
    double x2 = x * x;
    for (int i = 1; i <= 10; i++) {
        term *= -x2 / ((2*i) * (2*i-1));
        sum += term;
    }
    return sum;
}
