#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <stdint.h>

#define CTF_ALPHA 91

/* Runtime buffer: simulates VNA sweep in MCU SRAM */
extern float ctf_raw[CTF_ALPHA][2];

/* Untimed: Flash const recording -> ctf_raw */
void fill_ctf_raw_from_recording(int idx);

/* Timed T_prep: raw floats -> int8 in CNN accelerator input memory */
void preprocess_ctf(const float raw[CTF_ALPHA][2]);

#endif // PREPROCESS_H
