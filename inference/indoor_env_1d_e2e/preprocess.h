#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <stdint.h>

#include "preprocess_config.h"

/* Runtime buffer: simulates VNA sweep in MCU SRAM */
extern float ctf_raw[CTF_ALPHA][2];

/* Untimed: Flash const recording -> ctf_raw */
void fill_ctf_raw_from_recording(int idx);

/* Compatibility wrapper: raw floats -> packed int8 -> CNN accelerator input memory */
void preprocess_ctf(const float raw[CTF_ALPHA][2]);

/* Split e2e timing path: raw floats -> packed int8 words in SRAM */
void preprocess_ctf_pack(const float raw[CTF_ALPHA][2]);

/* Split e2e timing path: packed int8 words -> CNN accelerator input memory */
void preprocess_ctf_load_input(void);

#endif // PREPROCESS_H
