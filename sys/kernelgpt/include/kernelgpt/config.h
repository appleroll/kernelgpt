#ifndef KERNELGPT_CONFIG_H
#define KERNELGPT_CONFIG_H

typedef struct {
    double temperature;
    int max_steps;
    double learning_rate;
    double beta1;
    double beta2;
    double eps;
} GPTConfig;

extern GPTConfig gpt_config;

#endif
