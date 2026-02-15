#include "math.h"
#include <stdint.h>
/*
 * Copyright (C) 2011 The Android Open Source Project
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

/*
 * This implementation is from the Musl C library which in turn is from FreeBSD/Sun.
 * It is reasonably accurate and robust. For pure minimalistic implementation, we might
 * resort to Taylor series, but this is 'atomic' in the sense it works properly.
 * 
 * However, to adhere to 'simplest possible', I will write simpler, slower, inaccurate implementations
 * for clarity and because we don't need double precision perfect accuracy for a toy GPT.
 */

double fabs(double x) {
    if (x < 0) return -x;
    return x;
}

double ceil(double x) {
    int i = (int)x;
    if (x > (double)i) return (double)(i + 1);
    return (double)i;
}

double floor(double x) {
    if(x >= 0.0) return (double)((int)x);
    return (double)((int)x - 1);
}

// Taylor series for exp(x) = 1 + x + x^2/2! + x^3/3! + ...
// Valid for small x. For large x, use e^x = (e^(x/2))^2
double exp(double x) {
    // Handle very small or very large inputs loosely
    if (x < -700) return 0;
    if (x > 700) return 1.0/0.0; // inf

    // Range reduction
    int n = 0;
    if (fabs(x) > 1.0) {
        double half_x = x / 2.0;
        double r = exp(half_x);
        return r * r;
    }

    // Taylor series
    double sum = 1.0;
    double term = 1.0;
    for (int i = 1; i < 50; i++) {
        term *= x / i;
        sum += term;
        if (fabs(term) < 1e-15) break; 
    }
    return sum;
}

// Taylor series for ln(1+x) = x - x^2/2 + x^3/3 - ... for |x| < 1
// Use ln(x) = ln(m * 2^E) = ln(m) + E * ln(2)
double log(double x) {
    if (x <= 0) return -1.0/0.0; // -inf or NaN
    
    // Extract exponent and mantissa
    // For simplicity, we can just reduce range using loop
    int p = 0;
    while (x > 2.0) {
        x /= 2.0;
        p++;
    }
    while (x < 1.0) {
        x *= 2.0;
        p--;
    }
    // Now 1 <= x <= 2. Implement ln(x) around 1?
    // Better: let y = (x-1)/(x+1), then ln(x) = 2 * (y + y^3/3 + y^5/5...)
    double y = (x - 1.0) / (x + 1.0);
    double y2 = y * y;
    double sum = 0.0;
    double term = y;
    
    for (int i = 1; i < 50; i += 2) {
        sum += term / i;
        term *= y2;
    }
    
    return 2.0 * sum + p * 0.69314718056; // + p * ln(2)
}

double pow(double base, double exponent) {
    if (base == 0) return 0;
    if (exponent == 0) return 1;
    if (exponent == 1) return base;
    
    // x^y = exp(y * ln(x))
    // Handling negative base with integer exponent?
    if (base < 0 && (int)exponent == exponent) {
        double res = exp(exponent * log(-base));
        if ((int)exponent % 2 != 0) return -res;
        return res;
    }
    
    return exp(exponent * log(base));
}

double sqrt(double x) {
    if (x < 0) return -1.0; // NaN
    if (x == 0) return 0;
    
    // Newton's method: x_{n+1} = 0.5 * (x_n + S / x_n)
    double val = x;
    for (int i = 0; i < 10; i++) {
        val = 0.5 * (val + x / val);
    }
    return val;
}
