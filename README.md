# BitWeaving-S  
## Overview
A simple implementation of BitWeaving, a technique that exploits the parallelism available at the bit level in modern processors. BitWeaving operates on multiple bits of data in a single cycle, processing bits from different columns in each cycle. For more details, please refer to [the BitWeaving Paper](https://dl.acm.org/doi/10.1145/2463676.2465322).    
## Usage
Just compile and run.
```
$ make  
$ ./BitWeaving
```
## Detail of the codes
### Basic information
In this program VBP (Algorithm 2 in [the BitWeaving Paper](https://dl.acm.org/doi/10.1145/2463676.2465322)) is implemented. You can set the number of rows of data in the database, as well as the length of the data (number of bits used to represent a row of data). The length of each word is fixed to 32.
### About SIMD
SIMD (Single Instruction, Multiple Data) are used to speed up the program. A 128-bit register is devided into 4 32-bit parts (an int in C/C++). So 4 sections in the VBP algorithm can run in parallel.
## Reference
Yinan Li and Jignesh M. Patel. 2013. BitWeaving: fast scans for main memory data processing. In Proceedings of the 2013 ACM SIGMOD International Conference on Management of Data (SIGMOD '13). Association for Computing Machinery, New York, NY, USA, 289–300. DOI:https://doi.org/10.1145/2463676.2465322  
