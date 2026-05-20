/*******************************************************************************
 * indoor_env_1d_e2e — MAX78002 e2e latency bench (simulated VNA buffer in SRAM)
 *
 * CNN/weights from ai8xize (indoor_env_1d_91_q8824, alpha=91, INT 8-8-2-4).
 * Post-synthesis: ctf_raw in SRAM, preprocess_ctf (T_prep), timed prep+inf+act.
 *
 * PMON (external power monitor on EVKIT):
 *   P1.6 / gpio_trig1 (SYS_*):  high during T_prep and T_act (M4F)
 *   P1.7 / gpio_trig2 (CNN_*):  high during T_INP+INF (accelerator)
 ******************************************************************************/

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "mxc.h"
#include "cnn.h"
#include "preprocess.h"
#include "preprocess_config.h"
#include "sampleoutput.h"

#ifndef E2E_NUM_RUNS
#define E2E_NUM_RUNS 100
#endif

#ifndef E2E_RUN_KAT
#define E2E_RUN_KAT 1
#endif

mxc_gpio_cfg_t gpio_trig1, gpio_trig2;
volatile uint32_t cnn_time;

static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

static const uint32_t sample_output[] = SAMPLE_OUTPUT;

typedef struct {
  uint32_t last_us;
  uint32_t min_us;
  uint32_t max_us;
  uint64_t sum_us;
} stage_timing_t;

static stage_timing_t t_prep;
static stage_timing_t t_inf;
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

static void timing_print(const char *label, int num_runs, const stage_timing_t *t)
{
  unsigned mean_us = (num_runs > 0) ? (unsigned)(t->sum_us / (uint64_t)num_runs) : 0U;
  printf("%s over %d runs: min=%u us, max=%u us, mean=%u us, last=%u us\n", label, num_runs,
         t->min_us, t->max_us, mean_us, t->last_us);
}

static void fail(void)
{
  printf("\n*** FAIL ***\n\n");
  while (1) {}
}

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
      if ((*addr++ & mask) != *ptr++) {
        return CNN_FAIL;
      }
    }
  }
  return CNN_OK;
}

static void softmax_layer(void)
{
  cnn_unload((uint32_t *)ml_data);
  softmax_shift_q17p14_q15((q31_t *)ml_data, CNN_NUM_OUTPUTS, 4, ml_softmax);
}

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

static void run_one_classification(int run_idx)
{
  uint32_t us;
  (void)run_idx;

  /* Untimed: emulate "VNA sweep finished; data in SRAM" */
  fill_ctf_raw_from_recording(0);

  /* T_prep — M4F (PMON: P1.6 SYS) */
  SYS_START;
  MXC_TMR_SW_Start(MXC_TMR0);
  preprocess_ctf((const float (*)[2])ctf_raw);
  us = MXC_TMR_SW_Stop(MXC_TMR0);
  SYS_COMPLETE;
  timing_update(&t_prep, us);

  /* T_INP+INF — CNN accelerator (PMON: P1.7 CNN) */
  cnn_time = 0;
  cnn_clock_inference();
  CNN_START;
#ifndef CNN_INFERENCE_TIMER
  MXC_TMR_SW_Start(MXC_TMR0);
#endif
  cnn_start();
  while (cnn_time == 0) {
    MXC_LP_EnterSleepMode();
  }
#ifndef CNN_INFERENCE_TIMER
  us = MXC_TMR_SW_Stop(MXC_TMR0);
#else
  us = cnn_time;
#endif
  CNN_COMPLETE;
  cnn_clock_idle();
  timing_update(&t_inf, us);

  /* T_act — M4F softmax (PMON: P1.6 SYS) */
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
  MXC_GCR->ipll_ctrl |= MXC_F_GCR_IPLL_CTRL_EN;
  SystemCoreClockUpdate();

  printf("\n*** indoor_env_1d_e2e (source: %s, alpha=%d) ***\n", E2E_SYNTH_SOURCE, CTF_ALPHA);
  printf("Waiting for debugger...\n");
  MXC_Delay(SEC(2));

  gpio_pmon_init();
  printf("PMON: P1.6 (SYS) = T_prep + T_act;  P1.7 (CNN) = T_INP+INF\n");
  printf("      Use external monitor on these pins for per-stage energy.\n");

  cnn_disable();
  MXC_SYS_ClockSourceEnable(MXC_SYS_CLOCK_IPO);
  cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_IPLL, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV4);
  cnn_init();

  cnn_load_weights();
  cnn_load_bias();
  cnn_configure();

  printf("Running %d classifications (Flash recording -> SRAM ctf_raw, then timed stages)...\n",
         E2E_NUM_RUNS);

  timing_reset(&t_prep);
  timing_reset(&t_inf);
  timing_reset(&t_act);

  for (i = 0; i < E2E_NUM_RUNS; i++) {
    run_one_classification(i);
  }

  printf("\n--- Per-stage latency (TMR0 / CNN inference timer) ---\n");
  timing_print("T_prep", E2E_NUM_RUNS, &t_prep);
  timing_print("T_INP+INF", E2E_NUM_RUNS, &t_inf);
  timing_print("T_act", E2E_NUM_RUNS, &t_act);
  printf("T_e2e (sum of last-run stages): %u us\n",
         t_prep.last_us + t_inf.last_us + t_act.last_us);

  printf("\n--- Per-stage energy (external PMON) ---\n");
  printf("Integrate current on P1.6 during each SYS high window (prep, then act per loop).\n");
  printf("Integrate current on P1.7 during each CNN high window (inference per loop).\n");

#if E2E_RUN_KAT
  if (check_output() != CNN_OK) {
    fail();
  }
  printf("\nKnown-answer check: PASS (last run vs sampleoutput.h)\n");
#else
  printf("\nKnown-answer check skipped (E2E_RUN_KAT=0)\n");
#endif

  printf("\nClassification (last run):\n");
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
