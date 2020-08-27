bitweaving:bitweaving.cpp SIMD_operations.h
	g++ -mavx2 bitweaving.cpp SIMD_operations.h -o bitweaving -g 
clean:
	rm simd
