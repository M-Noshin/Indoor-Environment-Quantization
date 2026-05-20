#include "preprocess.h"

#include <math.h>
#include <string.h>

#include "cnn.h"
#include "ctf_recordings.h"

/* Match training/datasets/indoor_environment_1D.py GlobalMinMaxNormalize */
static const float global_min = -0.011066f;
static const float global_max = 0.011379f;

float ctf_raw[CTF_ALPHA][2];

void fill_ctf_raw_from_recording(int idx)
{
  (void)idx;
  if (CTF_NUM_RECORDINGS < 1) {
    return;
  }
  memcpy(ctf_raw, ctf_recording_0, sizeof(ctf_raw));
}

static int clamp_int8(int32_t v)
{
  if (v < -128) {
    return -128;
  }
  if (v > 127) {
    return 127;
  }
  return (int)v;
}

static void pack_channel_to_cnn(uint32_t dst_addr, const int8_t *ch, int n)
{
  uint32_t words[CNN_INPUT_WORDS_PER_CH];

  memset(words, 0, sizeof(words));
  memcpy(words, ch, (size_t)n);
  memcpy32((uint32_t *)dst_addr, words, CNN_INPUT_WORDS_PER_CH);
}

void preprocess_ctf(const float raw[CTF_ALPHA][2])
{
  const float global_range = global_max - global_min;
  int8_t ch0[CTF_ALPHA];
  int8_t ch1[CTF_ALPHA];

  for (int k = 0; k < CTF_ALPHA; k++) {
    float vals[2] = { raw[k][0], raw[k][1] };
    for (int c = 0; c < 2; c++) {
      float x_norm = (vals[c] - global_min) / global_range;
      int32_t q = (int32_t)lroundf((x_norm - 0.5f) * 256.0f);
      int8_t q8 = (int8_t)clamp_int8(q);
      if (c == 0) {
        ch0[k] = q8;
      } else {
        ch1[k] = q8;
      }
    }
  }

  pack_channel_to_cnn(CNN_INPUT_ADDR_CH0, ch0, CTF_ALPHA);
  pack_channel_to_cnn(CNN_INPUT_ADDR_CH1, ch1, CTF_ALPHA);
}
