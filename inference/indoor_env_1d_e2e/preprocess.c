#include "preprocess.h"

#include <string.h>

#include "mxc.h"
#include "cnn.h"
#include "preprocess_config.h"
#include "ctf_recordings.h"

/* Match training/datasets/indoor_environment_1D.py GlobalMinMaxNormalize */
#define GLOBAL_MIN (-0.011066f)
#define GLOBAL_MAX (0.011379f)
#define Q_SCALE (256.0f / (GLOBAL_MAX - GLOBAL_MIN))
#define Q_OFFSET ((-GLOBAL_MIN * Q_SCALE) - 128.0f)

float ctf_raw[CTF_ALPHA][2];

static uint32_t packed_ch0_words[CNN_INPUT_WORDS_PER_CH];
static uint32_t packed_ch1_words[CNN_INPUT_WORDS_PER_CH];

void fill_ctf_raw_from_recording(int idx)
{
  (void)idx;
  if (CTF_NUM_RECORDINGS < 1) {
    return;
  }
  memcpy(ctf_raw, ctf_recording_0, sizeof(ctf_raw));
}

static inline int clamp_int8(int32_t v)
{
  if (v < -128) {
    return -128;
  }
  if (v > 127) {
    return 127;
  }
  return (int)v;
}

static inline int32_t round_away_from_zero_f32(float x)
{
  return (int32_t)(x + ((x >= 0.0f) ? 0.5f : -0.5f));
}

void preprocess_ctf_pack(const float raw[CTF_ALPHA][2])
{
  memset(packed_ch0_words, 0, sizeof(packed_ch0_words));
  memset(packed_ch1_words, 0, sizeof(packed_ch1_words));

  for (int k = 0; k < CTF_ALPHA; k++) {
    unsigned shift = (unsigned)(k & 3) * 8U;
    uint32_t word = (uint32_t)k >> 2;

    int32_t q0 = round_away_from_zero_f32(raw[k][0] * Q_SCALE + Q_OFFSET);
    int32_t q1 = round_away_from_zero_f32(raw[k][1] * Q_SCALE + Q_OFFSET);

    packed_ch0_words[word] |= (uint32_t)(uint8_t)clamp_int8(q0) << shift;
    packed_ch1_words[word] |= (uint32_t)(uint8_t)clamp_int8(q1) << shift;
  }
}

void preprocess_ctf_load_input(void)
{
  memcpy32((uint32_t *)CNN_INPUT_ADDR_CH0, packed_ch0_words, CNN_INPUT_WORDS_PER_CH);
  memcpy32((uint32_t *)CNN_INPUT_ADDR_CH1, packed_ch1_words, CNN_INPUT_WORDS_PER_CH);
}

void preprocess_ctf(const float raw[CTF_ALPHA][2])
{
  preprocess_ctf_pack(raw);
  preprocess_ctf_load_input();
}
