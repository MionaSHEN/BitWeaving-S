//Copyright <2020> <Tianhong SHEN>

//Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

//The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define __STDC_LIMIT_MACROS    1

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <iostream>

#include "SIMD_operations.h"

#include <emmintrin.h>
#include <pmmintrin.h>
#include <xmmintrin.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <setjmp.h>
#include <unistd.h>
// Compiler name
#define MACTOSTR(x)    #x
#define MACROVALUESTR(x)    MACTOSTR(x)
#if defined(__ICL)    // Intel C++
#  if defined(__VERSION__)
#    define COMPILER_NAME    "Intel C++ " __VERSION__
#  elif defined(__INTEL_COMPILER_BUILD_DATE)
#    define COMPILER_NAME    "Intel C++ (" MACROVALUESTR(__INTEL_COMPILER_BUILD_DATE) ")"
#  else
#    define COMPILER_NAME    "Intel C++"
#  endif    // #  if defined(__VERSION__)
#elif defined(_MSC_VER)    // Microsoft VC++
#  if defined(_MSC_FULL_VER)
#    define COMPILER_NAME    "Microsoft VC++ (" MACROVALUESTR(_MSC_FULL_VER) ")"
#  elif defined(_MSC_VER)
#    define COMPILER_NAME    "Microsoft VC++ (" MACROVALUESTR(_MSC_VER) ")"
#  else
#    define COMPILER_NAME    "Microsoft VC++"
#  endif    // #  if defined(_MSC_FULL_VER)
#elif defined(__GNUC__)    // GCC
#  if defined(__CYGWIN__)
#    define COMPILER_NAME    "GCC(Cygmin) " __VERSION__
#  elif defined(__MINGW32__)
#    define COMPILER_NAME    "GCC(MinGW) " __VERSION__
#  else
#    define COMPILER_NAME    "GCC " __VERSION__
#  endif    // #  if defined(_MSC_FULL_VER)
#else
#  define COMPILER_NAME    "Unknown Compiler"
#endif    // #if defined(__ICL)    // Intel C++

#define __forceinline __attribute__((always_inline))

using namespace std;

//int C_length = 128; //number of data in the database
int C_length = 2000000; //number of data in the database
int B = 32;  // length of each data
int C[32][8000000];

int main (int argc, char **argv)
{
  srand( (unsigned)time( NULL ) );
  for (int i = 0; i < C_length; i++)
  {
  	int data = rand()%((int)pow(2.0, B)); // generate the data by random
  	//if (i%32 == 0 && i > 0) printf("\n");
  	if (B == 8) printf("%3d ", data);
  	if (B == 4) printf("%2d ", data);
  	for (int bit = 0; bit < B; bit++)
  	{  
  	  if (data >= pow(2, B-bit-1)) {data -= pow(2, B-bit-1); C[bit][i] = 1;}
  	  else {C[bit][i] = 0;}
  	}
  	//for (int bit = 0; bit < B; bit++) {printf("%d ", C[bit][i]);} printf("\n");
  }
  //printf("\n");
  double total_time = 0;
  int loop_num = 50;
  for (int loop = 0; loop < loop_num; loop++)
  {
	  printf("*****Loop #%d***************************************************************************************\n", loop);
	  int c1 = rand()%((int)pow(2.0, B));
	  int c2 = rand()%((int)pow(2.0, B));
	  //printf("%d, %d\n", c1, c2);
	  int temp;
	  if (c1 == c2) {c2 = (c2+1)%((int)pow(2.0, B)); }
	  if (c1 > c2)
	  {
	  	temp = c1;
	  	c1 = c2;
	  	c2 = temp;
	  } 
	  printf("C1 and C2: %d  %d\n", c1, c2);
	  clock_t startTime, endTime;
	  
	  startTime = clock();
	  vector<int> c1_bits;
	  for (int bit = 0; bit < B; bit++)
  	  {  
  	    if (c1 >= pow(2, B-bit-1)) {c1 -= pow(2, B-bit-1); c1_bits.push_back(1);}
  	    else {c1_bits.push_back(0);}
  	  }
  	  //for (int bit = 0; bit < B; bit++) {printf("%d ", c1_bits[bit]);}
  	  //printf("\n");
  	  vector<int> c2_bits;
	  for (int bit = 0; bit < B; bit++)
  	  {  
  	    if (c2 >= pow(2, B-bit-1)) {c2 -= pow(2, B-bit-1); c2_bits.push_back(1);}
  	    else {c2_bits.push_back(0);}
  	  }
  	  //for (int bit = 0; bit < B; bit++) {printf("%d ", c2_bits[bit]);}
  	  //printf("\n");
	  
	  vector<__m128i> c1_128;
	  for (int bit = 0; bit < B; bit++) {c1_128.push_back(_mm_set_epi32(c1_bits[bit]*0xffffffff,c1_bits[bit]*0xffffffff,c1_bits[bit]*0xffffffff,c1_bits[bit]*0xffffffff));}
	  	  
	  vector<__m128i> c2_128;
      for (int bit = 0; bit < B; bit++) {c2_128.push_back(_mm_set_epi32(c2_bits[bit]*0xffffffff,c2_bits[bit]*0xffffffff,c2_bits[bit]*0xffffffff,c2_bits[bit]*0xffffffff));}
	  
	  int section_num = C_length / 32; 
	  int period_num = C_length / 128; //4 sections executed in parallel
	  for (int period = 0; period < period_num; period++)
	  {
		// section #0, #1, #2, #3 
		__m128i m_lt_128 = _mm_set_epi32(0,0,0,0);
		__m128i m_gt_128 = _mm_set_epi32(0,0,0,0);
		__m128i m_eq1_128 = _mm_set_epi32(0xffffffff,0xffffffff,0xffffffff,0xffffffff);
		__m128i m_eq2_128 = _mm_set_epi32(0xffffffff,0xffffffff,0xffffffff,0xffffffff);
		//print128i_4(m_lt_128); print128i_4(m_gt_128); print128i_4(m_eq1_128); print128i_4(m_eq2_128);
		
		for (int i = 0; i < B; i++)
		{
	  	  int s_vi_0 = bits2int(C[i][period*128], C[i][period*128+1], C[i][period*128+2], C[i][period*128+3],C[i][period*128+4], C[i][period*128+5], C[i][period*128+6], C[i][period*128+7],C[i][period*128+8], C[i][period*128+9], C[i][period*128+10], C[i][period*128+11],C[i][period*128+12], C[i][period*128+13], C[i][period*128+14], C[i][period*128+15],C[i][period*128+16], C[i][period*128+17], C[i][period*128+18], C[i][period*128+19],C[i][period*128+20], C[i][period*128+21], C[i][period*128+22], C[i][period*128+23],C[i][period*128+24], C[i][period*128+25], C[i][period*128+26], C[i][period*128+27],C[i][period*128+28], C[i][period*128+29], C[i][period*128+30], C[i][period*128+31]);
	  	  //printf("%d\n", s_vi_0);
	  	  int s_vi_1 = bits2int(C[i][period*128+32], C[i][period*128+1+32], C[i][period*128+2+32], C[i][period*128+3+32],C[i][period*128+4+32], C[i][period*128+5+32], C[i][period*128+6+32], C[i][period*128+7+32],C[i][period*128+8+32], C[i][period*128+9+32], C[i][period*128+10+32], C[i][period*128+11+32],C[i][period*128+12+32], C[i][period*128+13+32], C[i][period*128+14+32], C[i][period*128+15+32],C[i][period*128+16+32], C[i][period*128+17+32], C[i][period*128+18+32], C[i][period*128+19+32],C[i][period*128+20+32], C[i][period*128+21+32], C[i][period*128+22+32], C[i][period*128+23+32],C[i][period*128+24+32], C[i][period*128+25+32], C[i][period*128+26+32], C[i][period*128+27+32],C[i][period*128+28+32], C[i][period*128+29+32], C[i][period*128+30+32], C[i][period*128+31+32]);
	  	  int s_vi_2 = bits2int(C[i][period*128+64], C[i][period*128+1+64], C[i][period*128+2+64], C[i][period*128+3+64],C[i][period*128+4+64], C[i][period*128+5+64], C[i][period*128+6+64], C[i][period*128+7+64],C[i][period*128+8+64], C[i][period*128+9+64], C[i][period*128+10+64], C[i][period*128+11+64],C[i][period*128+12+64], C[i][period*128+13+64], C[i][period*128+14+64], C[i][period*128+15+64],C[i][period*128+16+64], C[i][period*128+17+64], C[i][period*128+18+64], C[i][period*128+19+64],C[i][period*128+20+64], C[i][period*128+21+64], C[i][period*128+22+64], C[i][period*128+23+64],C[i][period*128+24+64], C[i][period*128+25+64], C[i][period*128+26+64], C[i][period*128+27+64],C[i][period*128+28+64], C[i][period*128+29+64], C[i][period*128+30+64], C[i][period*128+31+64]);
	  	  int s_vi_3 = bits2int(C[i][period*128+96], C[i][period*128+1+96], C[i][period*128+2+96], C[i][period*128+3+96],C[i][period*128+4+96], C[i][period*128+5+96], C[i][period*128+6+96], C[i][period*128+7+96],C[i][period*128+8+96], C[i][period*128+9+96], C[i][period*128+10+96], C[i][period*128+11+96],C[i][period*128+12+96], C[i][period*128+13+96], C[i][period*128+14+96], C[i][period*128+15+96],C[i][period*128+16+96], C[i][period*128+17+96], C[i][period*128+18+96], C[i][period*128+19+96],C[i][period*128+20+96], C[i][period*128+21+96], C[i][period*128+22+96], C[i][period*128+23+96],C[i][period*128+24+96], C[i][period*128+25+96], C[i][period*128+26+96], C[i][period*128+27+96],C[i][period*128+28+96], C[i][period*128+29+96], C[i][period*128+30+96], C[i][period*128+31+96]);
	  	  __m128i s_vi_128 = _mm_set_epi32(s_vi_3, s_vi_2, s_vi_1, s_vi_0);  //remind the order
	  	  m_gt_128 = _mm_or_si128(m_gt_128, _mm_and_si128(m_eq1_128, _mm_and_si128(_mm_not_si128(c1_128[i]), s_vi_128)));
	  	  m_lt_128 = _mm_or_si128(m_lt_128, _mm_and_si128(m_eq2_128, _mm_and_si128(c2_128[i], _mm_not_si128(s_vi_128))));
		  m_eq1_128 = _mm_and_si128(m_eq1_128, _mm_not_si128(_mm_xor_si128(s_vi_128, c1_128[i])));
		  m_eq2_128 = _mm_and_si128(m_eq2_128, _mm_not_si128(_mm_xor_si128(s_vi_128, c2_128[i])));
		}
		__m128i period_result = _mm_and_si128(m_gt_128, m_lt_128);
		//print128i_b(period_result);
	  }
	  endTime = clock();
	  double time = (double)(endTime - startTime) / CLOCKS_PER_SEC;
	  cout << "Time: " << time << endl;
	  total_time += time;
	  //cout << total_time << endl;
	  
  }
  printf("*****Summary: **************************************************************************************\n");
  printf("%d Rows of data, each data represented by %d bits.\n", C_length, B);
  cout << "Repeated for " << loop_num << " times. " << endl;
  cout << "Average Time: " << double(total_time / loop_num) << " s" << endl;
  
  return EXIT_SUCCESS;
}













