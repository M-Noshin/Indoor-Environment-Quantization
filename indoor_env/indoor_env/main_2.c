/*******************************************************************************
* Copyright (C) 2019-2024 Maxim Integrated Products, Inc.
*******************************************************************************/
// indoor_env — timing + energy with reliable µs reporting

#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "mxc.h"
#include "tmr.h"
#include "cnn.h"
#include "sampledata.h"
#include "sampleoutput.h"

#define NUM_RUNS     1000
#define PRINT_EVERY  100

volatile uint32_t cnn_time;                 /* filled by CNN ISR */
static uint32_t   t_us[NUM_RUNS];           /* per‑run latencies */
static int32_t    ml_data[CNN_NUM_OUTPUTS];
static q15_t      ml_softmax[CNN_NUM_OUTPUTS];

/* ---------- helpers ---------- */
static inline uint32_t ticks_to_us(uint32_t ticks)
{
    /* APB clock = SystemCoreClock, prescale = 1 */
    return (uint32_t)((ticks * 1000000ULL) / SystemCoreClock);
}

void fail(void)
{
    printf("\n*** FAIL ***\n\n");
    while (1);
}

static const uint32_t input_0[] = SAMPLE_INPUT_0;
static inline void load_input(void)
{
    memcpy32((uint32_t *)0x51800000, input_0, 202);
}

/* Known‑answer test check */
static const uint32_t sample_output[] = SAMPLE_OUTPUT;
int check_output(void)
{
    int i; uint32_t mask,len; volatile uint32_t *addr; const uint32_t *ptr=sample_output;
    while((addr=(volatile uint32_t*)*ptr++)!=0){mask=*ptr++;len=*ptr++;for(i=0;i<len;i++)if((*addr++&mask)!=*ptr++)return CNN_FAIL;}return CNN_OK;
}

static inline void softmax_layer(void)
{
    cnn_unload((uint32_t*)ml_data);
    softmax_q17p14_q15((const q31_t*)ml_data, CNN_NUM_OUTPUTS, ml_softmax);
}

/* ------------------------------------------------------------------------- */
int main(void)
{
    int i;
    double sum_sq = 0.0;
    uint64_t sum_inf_us = 0;           /* accumulate inference time */

    /* ---- clocks & timer ---- */
    MXC_ICC_Enable(MXC_ICC0);
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    MXC_GCR->ipll_ctrl |= MXC_F_GCR_IPLL_CTRL_EN;
    SystemCoreClockUpdate();           /* now SystemCoreClock is valid */

    mxc_tmr_cfg_t cfg = {
        .pres    = MXC_TMR_PRES_1,
        .mode    = MXC_TMR_MODE_CONTINUOUS,
        .bitMode = MXC_TMR_BIT_MODE_32,
        .clock   = MXC_TMR_APB_CLK,
        .cmp_cnt = 0
    };
    MXC_TMR_Init(MXC_TMR0, &cfg, true);

    printf("Waiting...\n");
    MXC_Delay(SEC(2));

    /* ---- CNN setup ---- */
    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_IPLL,
               MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV4);
    cnn_init();
    cnn_load_weights();
    cnn_load_bias();
    cnn_configure();

    printf("\n*** CNN Inference Test indoor_env ***\n");
    printf("*** %d runs (run 1 & every %d) ***\n", NUM_RUNS, PRINT_EVERY);

    uint32_t ticks, us;

    /* ---- Weight loading (100×) ---- */
    printf("\nMeasuring weight‑loading (100 runs)...\n");
    SYS_START;
    MXC_TMR_SetCount(MXC_TMR0, 0); MXC_TMR_Start(MXC_TMR0);
    for (i = 0; i < 100; i++) cnn_load_weights();
    ticks = MXC_TMR_GetCount(MXC_TMR0); MXC_TMR_Stop(MXC_TMR0);
    SYS_COMPLETE;
    us = ticks_to_us(ticks);
    printf("Weight‑loading time : %u µs\n", us);

    /* ---- Input loading (100×) ---- */
    printf("\nMeasuring input‑loading (100 runs)...\n");
    SYS_START;
    MXC_TMR_SetCount(MXC_TMR0, 0); MXC_TMR_Start(MXC_TMR0);
    for (i = 0; i < 100; i++) load_input();
    ticks = MXC_TMR_GetCount(MXC_TMR0); MXC_TMR_Stop(MXC_TMR0);
    SYS_COMPLETE;
    us = ticks_to_us(ticks);
    printf("Input‑loading time  : %u µs\n", us);

    /* ---- Inference loop (1000×) ---- */
    printf("\nMeasuring inference (%d runs)...\n", NUM_RUNS);
    SYS_START;   /* energy window around entire loop */
    for (i = 0; i < NUM_RUNS; i++) {
        load_input();

        /* fast CNN clock */
        MXC_GCR->pclkdiv = (MXC_GCR->pclkdiv &
                           ~(MXC_F_GCR_PCLKDIV_CNNCLKDIV | MXC_F_GCR_PCLKDIV_CNNCLKSEL)) |
                           MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1 |
                           MXC_S_GCR_PCLKDIV_CNNCLKSEL_IPLL;

        cnn_time = 0;
        cnn_start();
        while (!cnn_time)
            MXC_LP_EnterSleepMode();

        t_us[i] = cnn_time;               /* per‑run latency */
        sum_inf_us += cnn_time;           /* accumulate */
        sum_sq     += (double)cnn_time * cnn_time;

        /* slow clock back for UART */
        MXC_GCR->pclkdiv = (MXC_GCR->pclkdiv &
                           ~(MXC_F_GCR_PCLKDIV_CNNCLKDIV | MXC_F_GCR_PCLKDIV_CNNCLKSEL)) |
                           MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV4 |
                           MXC_S_GCR_PCLKDIV_CNNCLKSEL_IPLL;
    }
    SYS_COMPLETE;

    printf("Inference total time: %llu µs (avg %.3f µs/run)\n",
           (unsigned long long)sum_inf_us,
           (double)sum_inf_us / NUM_RUNS);

    /* ---- Verify & softmax ---- */
    if (check_output() != CNN_OK) fail();
    softmax_layer();
    printf("\n*** PASS ***\n\n");

    /* ---- Per‑run stats ---- */
    double mean = (double)sum_inf_us / NUM_RUNS;
    double std  = sqrt(sum_sq/NUM_RUNS - mean*mean);

    for (i = 0; i < NUM_RUNS; i++)
        if (i == 0 || (i + 1) % PRINT_EVERY == 0)
            printf("Run %4d latency: %.3f µs\n", i + 1, (double)t_us[i]);

    printf("\nLatency over %d runs:\n  Mean: %.3f µs\n  Std‑dev: %.3f µs\n  Last: %.3f µs\n",
           NUM_RUNS, mean, std, (double)t_us[NUM_RUNS - 1]);

    /* ---- Classification ---- */
    printf("\nClassification results (last run):\n");
    for (i = 0; i < CNN_NUM_OUTPUTS; i++) {
        int digs = (1000 * ml_softmax[i] + 0x4000) >> 15;
        printf("[%7d] -> Class %d: %d.%d%%\n", ml_data[i], i, digs / 10, digs % 10);
    }

    cnn_disable();
    MXC_GCR->ipll_ctrl &= ~MXC_F_GCR_IPLL_CTRL_EN;
    printf("\n*** DONE ***\n");
    return 0;
}
