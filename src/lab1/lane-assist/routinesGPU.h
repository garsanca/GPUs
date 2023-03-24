#ifndef ROUTINESGPU_H
#define ROUTINESGPU_H

#include <stdint.h>

	
void lane_assist_GPU(uint8_t *im, int height, int width,
	int *x1, int *y1, int *x2, int *y2, int *nlines);

#endif

