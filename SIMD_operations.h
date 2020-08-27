//Copyright <2020> <Tianhong SHEN>

//Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

//The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include <pmmintrin.h>
#include <xmmintrin.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <setjmp.h>
#include <unistd.h>

#define __forceinline __attribute__((always_inline))

void print128i_4(__m128i var)
{
    uint32_t val[4];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %d %d %d %d \n", 
           val[0], val[1], val[2], val[3]);
}

inline __m128i _mm_not_si128 (__m128i a)
{
	return _mm_xor_si128(a, _mm_set1_epi32(0xffffffff));
}

void print128i_b(__m128i var)
{
	uint32_t val[4];
	memcpy(val, &var, sizeof(val));
	for (int i = 0; i < 4; i++)
	{
	  if (val[i] < 0x80000000) // positive integer
	  {
	    printf(" 0 ");
	  	for (int j = 1; j < 32; j++)
	  	{
	  		if (val[i] >= pow(2, 31-j)) {val[i] -= pow(2, 31-j); printf(" 1 ");}
	  		else printf(" 0 ");
	  	}
	  	printf("\n");
	  }
	  else // negative integer
	  {
	  	int p = -1*val[i];
	  	int b[32];
	  	b[0] = 0;
	  	for (int j = 1; j < 32; j++)
	  	{
	  		if (p >= pow(2, 31-j)) {p -= pow(2, 31-j); b[j] = 1;}
	  		else b[j] = 0;
	  	}
	  	for (int k = 0; k < 32; k++) {b[k] = 1 - b[k];}
	  	int carry[32] = {0};
	  	carry[31] = 1;
	  	int result[32];
	  	for (int l = 31; l > 0; l--)
	  	{
	  		if (carry[l] + b[l] == 2) {result[l] = 0; carry[l-1] = 1;}
	  		else {result[l] = carry[l] + b[l];}
	  	}
	  	if (carry[0] + b[0] == 2) {result[0] = 0;}
	  	else {result[0] = carry[0] + b[0];}
	  	for (int m = 0; m < 32; m++) {printf(" %d ", result[m]);}
	  	printf("\n");
	  }
	}
	//printf("\n");
}


int bits2int(int a1, int a2, int a3, int a4, int a5, int a6, int a7, int a8, int a9, int a10, int a11, int a12, int a13, int a14, int a15, int a16, int a17, int a18, int a19, int a20, int a21, int a22, int a23, int a24, int a25, int a26, int a27, int a28, int a29, int a30, int a31, int a32)
{
  int result;
  if (a1 == 0) {result = a2*pow(2,30) + a3*pow(2,29) + a4*pow(2,28) + a5*pow(2,27) + a6*pow(2,26) + a7*pow(2,25) + a8*pow(2,24) + a9*pow(2,23) + a10*pow(2,22) + a11*pow(2,21) + a12*pow(2,20) + a13*pow(2,19) + a14*pow(2,18) + a15*pow(2,17) + a16*pow(2,16) + a17*pow(2,15) + a18*pow(2,14) + a19*pow(2,13) + a20*pow(2,12) + a21*pow(2,11) + a22*pow(2,10) + a23*pow(2,9) + a24*pow(2,8) + a25*pow(2,7) + a26*pow(2,6) + a27*pow(2,5) + a28*pow(2,4) + a29*pow(2,3) + a30*pow(2,2) + a31*pow(2,1) + a32*pow(2,0);}
  else {
  	a1 = 1 - a1; a2 = 1 - a2; a3 = 1 - a3; a4 = 1 - a4;
  	a5 = 1 - a5; a6 = 1 - a6; a7 = 1 - a7; a8 = 1 - a8;
  	a9 = 1 - a9; a10 = 1 - a10; a11 = 1 - a11; a12 = 1 - a12;
  	a13 = 1 - a13; a14 = 1 - a14; a15 = 1 - a15; a16 = 1 - a16;
  	a17 = 1 - a17; a18 = 1 - a18; a19 = 1 - a19; a20 = 1 - a20;
  	a21 = 1 - a21; a22 = 1 - a22; a23 = 1 - a23; a24 = 1 - a24;
  	a25 = 1 - a25; a26 = 1 - a26; a27 = 1 - a27; a28 = 1 - a28;
  	a29 = 1 - a29; a30 = 1 - a30; a31 = 1 - a31; a32 = 1 - a32;
  	result = -1 * (1 + a2*pow(2,30) + a3*pow(2,29) + a4*pow(2,28) + a5*pow(2,27) + a6*pow(2,26) + a7*pow(2,25) + a8*pow(2,24) + a9*pow(2,23) + a10*pow(2,22) + a11*pow(2,21) + a12*pow(2,20) + a13*pow(2,19) + a14*pow(2,18) + a15*pow(2,17) + a16*pow(2,16) + a17*pow(2,15) + a18*pow(2,14) + a19*pow(2,13) + a20*pow(2,12) + a21*pow(2,11) + a22*pow(2,10) + a23*pow(2,9) + a24*pow(2,8) + a25*pow(2,7) + a26*pow(2,6) + a27*pow(2,5) + a28*pow(2,4) + a29*pow(2,3) + a30*pow(2,2) + a31*pow(2,1) + a32*pow(2,0));
  }
  return result;
}









