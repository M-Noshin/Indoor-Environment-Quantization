/*******************************************************************************
 * indoor_env_1d_e2e — MAX78002 e2e latency / energy bench
 *
 * Two measurement modes (see project.mk):
 *   1) ADI-style PMON blocks (compare with indoor_env_1d_*_q8824 synthesis demos)
 *   2) Per-loop e2e stages: T_prep, T_INP+INF, T_act (paper path)
 *
 * PMON GPIO (external EVKIT monitor):
 *   P1.6 SYS_*  — idle baseline, T_prep, T_act, optional prep-load block
 *   P1.7 CNN_*  — weight load, input load, fused load+inf (ADI), T_INP+INF
 ******************************************************************************/

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "mxc.h"
#include "cnn.h"
#include "preprocess.h"
#include "preprocess_config.h"
#include "sampledata.h"
#include "sampleoutput.h"

#ifndef E2E_NUM_RUNS
#define E2E_NUM_RUNS 100
#endif

#ifndef E2E_RUN_KAT
#define E2E_RUN_KAT 0
#endif

#ifndef E2E_RUN_E2E_KAT
#define E2E_RUN_E2E_KAT 0
#endif

#ifndef E2E_PMON_ADI_ONLY
#define E2E_PMON_ADI_ONLY 0
#endif

/* ADI synthesis-demo PMON characterization (idle / weight / input / fused). */
#ifndef E2E_PMON_ADI_BLOCKS
#define E2E_PMON_ADI_BLOCKS 1
#endif

#ifndef E2E_PMON_ADI_REPS
#define E2E_PMON_ADI_REPS 100
#endif

/* Ticks for MXC_TMR_Delay between PMON blocks (ADI uses 500000). */
#ifndef E2E_PMON_SETTLE_TICKS
#define E2E_PMON_SETTLE_TICKS 500000
#endif

#ifndef E2E_SYS_PMON_ARM_DELAY_MS
#define E2E_SYS_PMON_ARM_DELAY_MS 5000
#endif

/* Optional 100x preprocess_ctf SYS block. Off by default to keep ADI PMON sequence exact. */
#ifndef E2E_PMON_PREP_BLOCK
#define E2E_PMON_PREP_BLOCK 0
#endif

/* Dedicated EVKIT System Power Mode runs. Select one and put PMON in System Power Mode. */
#define E2E_SYS_PMON_NONE 0
#define E2E_SYS_PMON_PREP 1
#define E2E_SYS_PMON_ACT 2
#define E2E_SYS_PMON_FULL 3
#define E2E_SYS_PMON_INP 4
#define E2E_SYS_PMON_1S 5
#define E2E_SYS_PMON_CNNCLR_1S 6
#define E2E_SYS_PMON_PREP_SUSTAINED 7
#define E2E_SYS_PMON_PREP_IDLE 8

#ifndef E2E_SYS_PMON_MODE
#define E2E_SYS_PMON_MODE E2E_SYS_PMON_NONE
#endif

#ifndef E2E_SYS_PMON_FIXED_MS
#define E2E_SYS_PMON_FIXED_MS 1000
#endif

#ifndef E2E_SYS_PMON_IDLE_MS
#define E2E_SYS_PMON_IDLE_MS 60000
#endif

#ifndef E2E_SYS_PMON_REPS
#if E2E_SYS_PMON_MODE == E2E_SYS_PMON_PREP
#define E2E_SYS_PMON_REPS 10000
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_INP
#define E2E_SYS_PMON_REPS 100000
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_ACT
#define E2E_SYS_PMON_REPS 100000
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_FULL
#define E2E_SYS_PMON_REPS 2000
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_1S
#define E2E_SYS_PMON_REPS 1
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_CNNCLR_1S
#define E2E_SYS_PMON_REPS 1
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_PREP_SUSTAINED
#define E2E_SYS_PMON_REPS 1200000
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_PREP_IDLE
#define E2E_SYS_PMON_REPS 1
#else
#define E2E_SYS_PMON_REPS E2E_PMON_ADI_REPS
#endif
#endif

#if (E2E_SYS_PMON_MODE == E2E_SYS_PMON_NONE) || (E2E_SYS_PMON_MODE == E2E_SYS_PMON_INP) \
  || (E2E_SYS_PMON_MODE == E2E_SYS_PMON_ACT) || (E2E_SYS_PMON_MODE == E2E_SYS_PMON_FULL)
#define E2E_NEEDS_CNN 1
#else
#define E2E_NEEDS_CNN 0
#endif

/*
 * Wait for CNN IRQ: sleep (alpha >= 81) vs __NOP busy-wait (alpha <= 71).
 * Smaller-alpha synthesis trees hang if the core sleeps (see indoor_env_1d_51_q8824).
 * Override with -DE2E_CNN_WAIT_NOP=0|1 in project.mk.
 */
#ifndef E2E_CNN_WAIT_ALPHA_THRESHOLD
#define E2E_CNN_WAIT_ALPHA_THRESHOLD 71
#endif
#ifndef E2E_CNN_WAIT_NOP
#if (CTF_ALPHA <= E2E_CNN_WAIT_ALPHA_THRESHOLD)
#define E2E_CNN_WAIT_NOP 1
#else
#define E2E_CNN_WAIT_NOP 0
#endif
#endif

mxc_gpio_cfg_t gpio_trig1, gpio_trig2;
volatile uint32_t cnn_time;

static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

#if E2E_RUN_KAT || E2E_RUN_E2E_KAT
static const uint32_t sample_output[] = SAMPLE_OUTPUT;
#endif

#if E2E_PMON_ADI_BLOCKS && (E2E_SYS_PMON_MODE == E2E_SYS_PMON_NONE)
static const uint32_t sample_input_ch0[] = SAMPLE_INPUT_0;
static const uint32_t sample_input_ch8[] = SAMPLE_INPUT_8;
#endif

typedef struct {
  uint32_t last_us;
  uint32_t min_us;
  uint32_t max_us;
  uint64_t sum_us;
} stage_timing_t;

static stage_timing_t t_prep;
static stage_timing_t t_inp_inf;
static stage_timing_t t_act;

static void timing_reset(stage_timing_t *t)
{
  t->last_us = 0;
  t->min_us = UINT32_MAX;
  t->max_us = 0;
  t->sum_us = 0;
}

static void timing_update(stage_timing_t *t, uint32_t us)
{
  t->last_us = us;
  if (us < t->min_us) {
    t->min_us = us;
  }
  if (us > t->max_us) {
    t->max_us = us;
  }
  t->sum_us += us;
}

static void print_us_3(uint64_t us_x1000)
{
  unsigned whole = (unsigned)(us_x1000 / 1000U);
  unsigned frac = (unsigned)(us_x1000 % 1000U);

  printf("%u.%u%u%u", whole, frac / 100U, (frac / 10U) % 10U, frac % 10U);
}

static void timing_print(const char *label, int num_runs, const stage_timing_t *t)
{
  uint64_t mean_us_x1000 = (num_runs > 0)
                               ? ((t->sum_us * 1000U + (uint64_t)(num_runs / 2)) /
                                  (uint64_t)num_runs)
                               : 0U;

  printf("%s over %d runs: min=", label, num_runs);
  print_us_3((uint64_t)t->min_us * 1000U);
  printf(" us, max=");
  print_us_3((uint64_t)t->max_us * 1000U);
  printf(" us, mean=");
  print_us_3(mean_us_x1000);
  printf(" us, last=");
  print_us_3((uint64_t)t->last_us * 1000U);
  printf(" us\n");
}

#if E2E_RUN_KAT || E2E_RUN_E2E_KAT
static int check_output(void)
{
  int i;
  uint32_t mask, len;
  volatile uint32_t *addr;
  const uint32_t *ptr = sample_output;

  while ((addr = (volatile uint32_t *)*ptr++) != 0) {
    mask = *ptr++;
    len = *ptr++;
    for (i = 0; i < len; i++) {
      uint32_t read = *addr++ & mask;
      uint32_t expected = *ptr++;
      if (read != expected) {
        printf("Data mismatch (%d/%d) at 0x%08x: expected 0x%08x, read 0x%08x\n", i + 1, len,
               (unsigned)(addr - 1), (unsigned)expected, (unsigned)read);
        return CNN_FAIL;
      }
  }
  }
  return CNN_OK;
}
#endif

static void softmax_layer(void)
{
  cnn_unload((uint32_t *)ml_data);
  softmax_shift_q17p14_q15((q31_t *)ml_data, CNN_NUM_OUTPUTS, 4, ml_softmax);
}

#if E2E_SYS_PMON_MODE == E2E_SYS_PMON_NONE
static void gpio_pmon_init(void)
{
  gpio_trig1.port = MXC_GPIO1;
  gpio_trig1.mask = MXC_GPIO_PIN_6;
  gpio_trig1.pad = MXC_GPIO_PAD_NONE;
  gpio_trig1.func = MXC_GPIO_FUNC_OUT;
  MXC_GPIO_Config(&gpio_trig1);

  gpio_trig2.port = MXC_GPIO1;
  gpio_trig2.mask = MXC_GPIO_PIN_7;
  gpio_trig2.pad = MXC_GPIO_PAD_NONE;
  gpio_trig2.func = MXC_GPIO_FUNC_OUT;
  MXC_GPIO_Config(&gpio_trig2);

  SYS_COMPLETE;
  CNN_COMPLETE;
}
#endif

static void pmon_settle_delay(void)
{
  MXC_TMR_Delay(MXC_TMR0, E2E_PMON_SETTLE_TICKS);
}

#if E2E_SYS_PMON_MODE != E2E_SYS_PMON_NONE
static void gpio_sys_pmon_init(void)
{
  gpio_trig1.port = MXC_GPIO1;
  gpio_trig1.mask = MXC_GPIO_PIN_6;
  gpio_trig1.pad = MXC_GPIO_PAD_NONE;
  gpio_trig1.func = MXC_GPIO_FUNC_OUT;
  gpio_trig1.vssel = MXC_GPIO_VSSEL_VDDIO;
  gpio_trig1.drvstr = MXC_GPIO_DRVSTR_0;
  MXC_GPIO_Config(&gpio_trig1);

  gpio_trig2.port = MXC_GPIO1;
  gpio_trig2.mask = MXC_GPIO_PIN_7;
  gpio_trig2.pad = MXC_GPIO_PAD_NONE;
  gpio_trig2.func = MXC_GPIO_FUNC_OUT;
  gpio_trig2.vssel = MXC_GPIO_VSSEL_VDDIO;
  gpio_trig2.drvstr = MXC_GPIO_DRVSTR_0;
  MXC_GPIO_Config(&gpio_trig2);

  SYS_COMPLETE;
  CNN_COMPLETE;
}

static void pmon_countdown_delay(uint32_t delay_ms)
{
  uint32_t remaining_ms = delay_ms;

  while (remaining_ms >= 1000U) {
    MXC_Delay(SEC(1));
    remaining_ms -= 1000U;
    printf(".");
  }
  if (remaining_ms > 0U) {
    MXC_Delay(MSEC(remaining_ms));
  }
  printf("\n");
}

#if E2E_SYS_PMON_MODE == E2E_SYS_PMON_CNNCLR_1S
static void pmon_cnn_timer_clear_pulse(void)
{
  printf("[PMON CLR] Emitting CNN-mode idle + active pulse to clear PMON TMR1 flag...\n");
  SYS_START;
  MXC_Delay(MSEC(100));
  SYS_COMPLETE;
  MXC_Delay(MSEC(100));

  CNN_START;
  MXC_Delay(MSEC(100));
  CNN_COMPLETE;
  MXC_Delay(MSEC(100));
}
#endif
#endif

/* ADI load_input(): pre-baked int8 sampledata.h -> CNN SRAM (for PMON parity). */
#if E2E_PMON_ADI_BLOCKS && (E2E_SYS_PMON_MODE == E2E_SYS_PMON_NONE)
static void load_input_sampledata(void)
{
  memcpy32((uint32_t *)CNN_INPUT_ADDR_CH0, sample_input_ch0, CNN_INPUT_WORDS_PER_CH);
  memcpy32((uint32_t *)CNN_INPUT_ADDR_CH1, sample_input_ch8, CNN_INPUT_WORDS_PER_CH);
}
#endif

static void wait_for_cnn(void)
{
#if E2E_CNN_WAIT_NOP
  while (cnn_time == 0) {
    __NOP();
  }
#else
  while (cnn_time == 0) {
    MXC_LP_EnterSleepMode();
  }
#endif
}

static void cnn_clock_inference(void)
{
  MXC_GCR->pclkdiv = (MXC_GCR->pclkdiv & ~(MXC_F_GCR_PCLKDIV_CNNCLKDIV | MXC_F_GCR_PCLKDIV_CNNCLKSEL))
                     | MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1 | MXC_S_GCR_PCLKDIV_CNNCLKSEL_IPLL;
}

static void cnn_clock_idle(void)
{
  MXC_GCR->pclkdiv = (MXC_GCR->pclkdiv & ~(MXC_F_GCR_PCLKDIV_CNNCLKDIV | MXC_F_GCR_PCLKDIV_CNNCLKSEL))
                     | MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV4 | MXC_S_GCR_PCLKDIV_CNNCLKSEL_IPLL;
}

static void cnn_run_inference_once(void)
{
  cnn_time = 0;
  cnn_clock_inference();
  cnn_start();
  wait_for_cnn();
  cnn_clock_idle();
}

#if E2E_NEEDS_CNN
static void cnn_prepare_model(void)
{
  cnn_init();
  cnn_load_weights();
  cnn_load_bias();
  cnn_configure();
}
#endif

#if E2E_RUN_KAT || E2E_RUN_E2E_KAT
static void fail(void)
{
  printf("\n*** FAIL ***\n\n");
  while (1) {}
}
#endif

#if E2E_PMON_ADI_BLOCKS && (E2E_SYS_PMON_MODE == E2E_SYS_PMON_NONE)
static void pmon_print_energy_guide(void)
{
  printf("\n--- ADI-style PMON energy (read EVKIT monitor) ---\n");
  printf("  P1.6 SYS high:  (1) idle 1 s baseline\n");
  printf("  P1.7 CNN high:  (2) weight load x%d  (3) input load (sampledata) x%d\n", E2E_PMON_ADI_REPS,
         E2E_PMON_ADI_REPS);
  printf("  P1.7 CNN high:  (4) fused load_input+inference x%d  (matches synthesis demo)\n",
         E2E_PMON_ADI_REPS);
#if E2E_PMON_PREP_BLOCK
  printf("  P1.6 SYS high:  (optional) e2e preprocess_ctf x%d\n", E2E_PMON_ADI_REPS);
#endif
  printf("See monitor display for energy in each window.\n");
}

/* Same order as indoor_env_1d_91_q8824 / indoor_env_1d_51_q8824 main.c */
static void pmon_adi_characterization(void)
{
  int i;

  pmon_print_energy_guide();

#if E2E_PMON_PREP_BLOCK
  printf("\n[PMON 1/5] Measuring system base (idle) power — P1.6, 1 s...\n");
#else
  printf("\n[PMON 1/4] Measuring system base (idle) power — P1.6, 1 s...\n");
#endif
  SYS_START;
  MXC_Delay(SEC(1));
  SYS_COMPLETE;

  cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_IPLL, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV4);
  cnn_init();

#if E2E_PMON_PREP_BLOCK
  printf("[PMON 2/5] Measuring weight loading — P1.7, %d x cnn_load_weights()...\n", E2E_PMON_ADI_REPS);
#else
  printf("[PMON 2/4] Measuring weight loading — P1.7, %d x cnn_load_weights()...\n", E2E_PMON_ADI_REPS);
#endif
  CNN_START;
  for (i = 0; i < E2E_PMON_ADI_REPS; i++) {
    cnn_load_weights();
  }
  CNN_COMPLETE;
  pmon_settle_delay();

#if E2E_PMON_PREP_BLOCK
  printf("[PMON 3/5] Measuring input loading (sampledata memcpy) — P1.7, %d x load_input()...\n",
         E2E_PMON_ADI_REPS);
#else
  printf("[PMON 3/4] Measuring input loading (sampledata memcpy) — P1.7, %d x load_input()...\n",
         E2E_PMON_ADI_REPS);
#endif
  CNN_START;
  for (i = 0; i < E2E_PMON_ADI_REPS; i++) {
    load_input_sampledata();
  }
  CNN_COMPLETE;
  pmon_settle_delay();

  cnn_load_bias();
  cnn_configure();

#if E2E_PMON_PREP_BLOCK
  printf("[PMON 4/5] Measuring input load + inference (ADI fused) — P1.7, %d iterations...\n",
         E2E_PMON_ADI_REPS);
#else
  printf("[PMON 4/4] Measuring input load + inference (ADI fused) — P1.7, %d iterations...\n",
         E2E_PMON_ADI_REPS);
#endif
  CNN_START;
  for (i = 0; i < E2E_PMON_ADI_REPS; i++) {
    load_input_sampledata();
    cnn_run_inference_once();
  }
  CNN_COMPLETE;
  pmon_settle_delay();

#if E2E_RUN_KAT
  if (check_output() != CNN_OK) {
    fail();
  }
  printf("\nKnown-answer check: PASS (ADI fused sampledata loop vs sampleoutput.h)\n");
#endif

#if E2E_PMON_PREP_BLOCK
  pmon_settle_delay();
  printf("[PMON 5/5] Measuring e2e preprocess (raw CTF -> int8) — P1.6, %d x preprocess_ctf()...\n",
         E2E_PMON_ADI_REPS);
  SYS_START;
  for (i = 0; i < E2E_PMON_ADI_REPS; i++) {
    fill_ctf_raw_from_recording(0);
    preprocess_ctf((const float (*)[2])ctf_raw);
  }
  SYS_COMPLETE;
#endif

#ifdef CNN_INFERENCE_TIMER
  printf("ADI fused block: last inference time (cnn_time) = %u us\n", cnn_time);
#endif
}
#endif /* E2E_PMON_ADI_BLOCKS && E2E_SYS_PMON_MODE == E2E_SYS_PMON_NONE */

#if E2E_SYS_PMON_MODE != E2E_SYS_PMON_NONE
static void pmon_system_power_mode_run(void)
{
  int i;
  uint32_t window_us;

  printf("\n--- EVKIT System Power Mode run ---\n");
  printf("Set PMON to System Power Mode before this window.\n");
#if (E2E_SYS_PMON_MODE == E2E_SYS_PMON_1S) || (E2E_SYS_PMON_MODE == E2E_SYS_PMON_CNNCLR_1S)
  printf("Measurement window: P1.6 SYS high, fixed %d ms calibration pulse.\n",
         E2E_SYS_PMON_FIXED_MS);
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_PREP_IDLE
  printf("Measurement window: P1.6 SYS high, fixed %d ms M4F idle baseline.\n",
         E2E_SYS_PMON_IDLE_MS);
#else
  printf("Measurement window: P1.6 SYS high, %d iterations.\n", E2E_SYS_PMON_REPS);
#endif
  if (E2E_SYS_PMON_ARM_DELAY_MS > 0) {
    printf("Arming delay: %d ms. Put PMON in System Power Mode now (PMON serial: send 's').\n",
           E2E_SYS_PMON_ARM_DELAY_MS);
  } else {
    printf("No arming delay. PMON must already be in System Power Mode (PMON serial: 's').\n");
  }

  fill_ctf_raw_from_recording(0);
  gpio_sys_pmon_init();

#if E2E_SYS_PMON_MODE == E2E_SYS_PMON_CNNCLR_1S
  printf("[PMON CLR] Step 1: send 't' to PMON serial for CNN Power Mode now.\n");
  pmon_countdown_delay(E2E_SYS_PMON_ARM_DELAY_MS);
  pmon_cnn_timer_clear_pulse();
  printf("[PMON CLR] Step 2: send 's' to PMON serial for System Power Mode now.\n");
  pmon_countdown_delay(E2E_SYS_PMON_ARM_DELAY_MS);
#else
  if (E2E_SYS_PMON_ARM_DELAY_MS > 0) {
    pmon_countdown_delay(E2E_SYS_PMON_ARM_DELAY_MS);
  }
#endif

#if E2E_SYS_PMON_MODE == E2E_SYS_PMON_PREP
  printf("CNN accelerator remains disabled for prep-only isolation.\n");
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_PREP_SUSTAINED
  printf("CNN accelerator remains disabled for sustained prep mW measurement.\n");
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_PREP_IDLE
  printf("CNN accelerator remains disabled for sustained prep-idle baseline mW measurement.\n");
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_INP
  printf("Packing one input before the input-copy measurement window...\n");
  preprocess_ctf_pack((const float (*)[2])ctf_raw);
#endif

#if E2E_SYS_PMON_MODE == E2E_SYS_PMON_ACT
  printf("Priming CNN output for activation-only measurement...\n");
  preprocess_ctf((const float (*)[2])ctf_raw);
  cnn_run_inference_once();
#endif

  printf("[SYS PMON] Measuring ");
#if E2E_SYS_PMON_MODE == E2E_SYS_PMON_PREP
  printf("preprocess_ctf_pack");
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_PREP_SUSTAINED
  printf("sustained preprocess_ctf_pack");
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_PREP_IDLE
  printf("sustained prep-idle baseline");
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_INP
  printf("preprocess_ctf_load_input");
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_ACT
  printf("activation/softmax");
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_FULL
  printf("full e2e path");
#elif (E2E_SYS_PMON_MODE == E2E_SYS_PMON_1S) || (E2E_SYS_PMON_MODE == E2E_SYS_PMON_CNNCLR_1S)
  printf("fixed 1 s calibration pulse");
#endif
#if (E2E_SYS_PMON_MODE == E2E_SYS_PMON_1S) || (E2E_SYS_PMON_MODE == E2E_SYS_PMON_CNNCLR_1S)
  printf(" — P1.6, %d ms...\n", E2E_SYS_PMON_FIXED_MS);
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_PREP_IDLE
  printf(" — P1.6, %d ms...\n", E2E_SYS_PMON_IDLE_MS);
#else
  printf(" — P1.6, %d iterations...\n", E2E_SYS_PMON_REPS);
#endif

  /* EVKIT PMON v2.0 System Power captures the P1.6 SYS window. */
  MXC_TMR_SW_Start(MXC_TMR0);
  SYS_START;
  for (i = 0; i < E2E_SYS_PMON_REPS; i++) {
#if E2E_SYS_PMON_MODE == E2E_SYS_PMON_PREP
    preprocess_ctf_pack((const float (*)[2])ctf_raw);
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_PREP_SUSTAINED
    preprocess_ctf_pack((const float (*)[2])ctf_raw);
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_PREP_IDLE
    MXC_Delay(MSEC(E2E_SYS_PMON_IDLE_MS));
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_INP
    preprocess_ctf_load_input();
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_ACT
    softmax_layer();
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_FULL
    preprocess_ctf_pack((const float (*)[2])ctf_raw);
    preprocess_ctf_load_input();
    cnn_run_inference_once();
    softmax_layer();
#elif (E2E_SYS_PMON_MODE == E2E_SYS_PMON_1S) || (E2E_SYS_PMON_MODE == E2E_SYS_PMON_CNNCLR_1S)
    MXC_Delay(MSEC(E2E_SYS_PMON_FIXED_MS));
#endif
  }
  SYS_COMPLETE;
  window_us = MXC_TMR_SW_Stop(MXC_TMR0);
  pmon_settle_delay();

  printf("System Power Mode run complete; use PMON screen/serial result for this one stage.\n");
#if (E2E_SYS_PMON_MODE == E2E_SYS_PMON_1S) || (E2E_SYS_PMON_MODE == E2E_SYS_PMON_CNNCLR_1S)
  printf("Firmware SYS PMON calibration window = %u us total.\n", window_us);
  printf("PMON T should be close to %d ms on the display.\n", E2E_SYS_PMON_FIXED_MS);
  printf("If PMON T is ~45.7 s, reset the PMON MCU and rerun before any CNN Power capture.\n");
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_PREP_IDLE
  printf("Firmware SYS PMON idle baseline window = %u us total.\n", window_us);
  printf("For sustained prep energy, ignore PMON E/T and record only PMON mW from this idle run.\n");
  printf("Use E_prep_uJ = (P_prep_mW - P_idle_mW) * T_prep_us / 1000.\n");
#else
  printf("Firmware SYS PMON window = %u us total, %u us/iteration over %d reps.\n",
         window_us, (unsigned)(window_us / (uint32_t)E2E_SYS_PMON_REPS), E2E_SYS_PMON_REPS);
#if (E2E_SYS_PMON_MODE == E2E_SYS_PMON_PREP_SUSTAINED) || (E2E_SYS_PMON_MODE == E2E_SYS_PMON_PREP_IDLE)
  printf("For sustained prep energy, ignore PMON E/T and record only PMON mW from this run.\n");
  printf("Use E_prep_uJ = (P_prep_mW - P_idle_mW) * T_prep_us / 1000.\n");
#else
  printf("If PMON T matches this firmware window, divide PMON E and T by %d.\n", E2E_SYS_PMON_REPS);
  printf("If PMON T is the fixed ~45.7 s interval, use PMON average power x UART stage latency instead.\n");
#endif
#endif
}
#endif

static void run_one_classification(int run_idx)
{
  uint32_t us;
  uint32_t inp_inf_us;
  (void)run_idx;

  fill_ctf_raw_from_recording(0);

  SYS_START;
  MXC_TMR_SW_Start(MXC_TMR0);
  preprocess_ctf_pack((const float (*)[2])ctf_raw);
  us = MXC_TMR_SW_Stop(MXC_TMR0);
  SYS_COMPLETE;
  timing_update(&t_prep, us);

  CNN_START;
  MXC_TMR_SW_Start(MXC_TMR0);
  preprocess_ctf_load_input();
  cnn_run_inference_once();
  inp_inf_us = MXC_TMR_SW_Stop(MXC_TMR0);
  CNN_COMPLETE;
  timing_update(&t_inp_inf, inp_inf_us);

  SYS_START;
  MXC_TMR_SW_Start(MXC_TMR0);
  softmax_layer();
  us = MXC_TMR_SW_Stop(MXC_TMR0);
  SYS_COMPLETE;
  timing_update(&t_act, us);
}

int main(void)
{
  int i;
  int digs, tens;

  MXC_ICC_Enable(MXC_ICC0);

  MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
#if E2E_NEEDS_CNN
  MXC_GCR->ipll_ctrl |= MXC_F_GCR_IPLL_CTRL_EN;
#endif
  SystemCoreClockUpdate();

  printf("\n*** indoor_env_1d_e2e (source: %s, alpha=%d) ***\n", E2E_SYNTH_SOURCE, CTF_ALPHA);
#if E2E_CNN_WAIT_NOP
  printf("CNN wait: __NOP busy-wait (alpha<=%d; sleep can hang on smaller models)\n",
         E2E_CNN_WAIT_ALPHA_THRESHOLD);
#else
  printf("CNN wait: WFI/sleep until IRQ (typical for alpha>%d)\n", E2E_CNN_WAIT_ALPHA_THRESHOLD);
#endif
  printf("Waiting for debugger...\n");
  MXC_Delay(SEC(2));

#if E2E_SYS_PMON_MODE == E2E_SYS_PMON_NONE
  gpio_pmon_init();
#endif

  cnn_disable();
  MXC_SYS_ClockSourceEnable(MXC_SYS_CLOCK_IPO);

#if !E2E_RUN_KAT
  printf("\nKnown-answer check skipped (E2E_RUN_KAT=0)\n");
#endif

#if E2E_SYS_PMON_MODE == E2E_SYS_PMON_PREP
  printf("\nSystem PMON prep-only mode: CNN block not enabled.\n");
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_1S
  printf("\nSystem PMON fixed-1s calibration mode: CNN block not enabled.\n");
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_CNNCLR_1S
  printf("\nSystem PMON CNN-clear + fixed-1s calibration mode: CNN block not enabled.\n");
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_PREP_SUSTAINED
  printf("\nSystem PMON sustained prep-mW mode: CNN block not enabled.\n");
#elif E2E_SYS_PMON_MODE == E2E_SYS_PMON_PREP_IDLE
  printf("\nSystem PMON sustained prep-idle baseline mode: CNN block not enabled.\n");
#elif E2E_SYS_PMON_MODE != E2E_SYS_PMON_NONE
  cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_IPLL, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV4);
#elif E2E_PMON_ADI_BLOCKS
  pmon_adi_characterization();
#else
  cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_IPLL, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV4);
#endif

#if E2E_PMON_ADI_ONLY
  printf("\nADI-only PMON mode complete; stopping before e2e loop.\n");
  printf("Use PMON serial records for idle, kernels, input, and fused input+inference.\n");
  cnn_disable();
  MXC_GCR->ipll_ctrl &= ~MXC_F_GCR_IPLL_CTRL_EN;
  while (1) {}
#endif

#if E2E_NEEDS_CNN
  cnn_prepare_model();
#endif

#if E2E_SYS_PMON_MODE != E2E_SYS_PMON_NONE
  pmon_system_power_mode_run();
#if E2E_NEEDS_CNN
  cnn_disable();
  MXC_GCR->ipll_ctrl &= ~MXC_F_GCR_IPLL_CTRL_EN;
#endif
  while (1) {}
#endif

  printf("\n=== E2e per-loop benchmark (%d runs) ===\n", E2E_NUM_RUNS);
  printf("PMON per iteration: P1.6 = T_prep + T_act;  P1.7 = fused T_INP+INF\n");

  timing_reset(&t_prep);
  timing_reset(&t_inp_inf);
  timing_reset(&t_act);

  for (i = 0; i < E2E_NUM_RUNS; i++) {
    run_one_classification(i);
  }

  printf("\n--- Per-stage latency (e2e path) ---\n");
  timing_print("T_prep", E2E_NUM_RUNS, &t_prep);
  timing_print("T_INP+INF", E2E_NUM_RUNS, &t_inp_inf);
  timing_print("T_act", E2E_NUM_RUNS, &t_act);
  printf("T_e2e (sum of last-run stages): ");
  print_us_3((uint64_t)(t_prep.last_us + t_inp_inf.last_us + t_act.last_us) * 1000U);
  printf(" us\n");

  printf("\n--- Per-stage energy (e2e path, external PMON) ---\n");
  printf("Per loop: integrate P1.6 during prep and act; P1.7 during fused input load + inference.\n");

#if E2E_RUN_E2E_KAT
  if (check_output() != CNN_OK) {
    fail();
  }
  printf("\nE2E known-answer check: PASS (last e2e run vs sampleoutput.h)\n");
#else
  printf("\nE2E known-answer check skipped (E2E_RUN_E2E_KAT=0; sampleoutput.h is for sampledata.h)\n");
#endif

  printf("\nClassification (last e2e run):\n");
  for (i = 0; i < CNN_NUM_OUTPUTS; i++) {
    digs = (1000 * ml_softmax[i] + 0x4000) >> 15;
    tens = digs % 10;
    digs = digs / 10;
    printf("  Class %d: %d.%d%%\n", i, digs, tens);
  }

  printf("\n*** DONE ***\n\n");

  cnn_disable();
  MXC_GCR->ipll_ctrl &= ~MXC_F_GCR_IPLL_CTRL_EN;

  return 0;
}
