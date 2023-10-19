package blas64

import "C"

// #include "cblas.h"
import "C"

// Copy Copies a vector to another vector (double-precision).
func Copy(length int, src []float64, strideSrc int, dst []float64, strideDst int) {
	// https://developer.apple.com/documentation/accelerate/1513214-cblas_dcopy
	C.cblas_dcopy(
		(C.int)(int32(length)),    // N - Number of elements in the vectors.
		(*C.double)(&src[0]),      // X - Source vector X.
		(C.int)(int32(strideSrc)), // incX - Stride within X. For example, if incX is 7, every 7th element is used.
		(*C.double)(&dst[0]),      // Y - Destination vector Y.
		(C.int)(int32(strideDst)), // incY - Stride within Y. For example, if incY is 7, every 7th element is used.
	)
}

// Axpy Computes a constant times a vector plus a vector (double-precision).
func Axpy(length int, alpha float64, src []float64, strideSrc int, dst []float64, strideDst int) {
	// https://developer.apple.com/documentation/accelerate/1513298-cblas_daxpy
	C.cblas_daxpy(
		(C.int)(int32(length)),    // N - Number of elements in the vectors.
		(C.double)(alpha),         // alpha - Scaling factor for the values in X.
		(*C.double)(&src[0]),      // X - Input vector X.
		(C.int)(int32(strideSrc)), // incX - Stride within X. For example, if incX is 7, every 7th element is used
		(*C.double)(&dst[0]),      // Y - Input vector Y.
		(C.int)(int32(strideDst)), // incY - Stride within Y. For example, if incY is 7, every 7th element is used.
	)
}

// Scal Multiplies each element of a vector by a constant (double-precision).
func Scal(length int, alpha float64, src []float64, strideSrc int) {
	// https://developer.apple.com/documentation/accelerate/1513084-cblas_dscal
	C.cblas_dscal(
		(C.int)(int32(length)),    // N - Number of elements in the vectors.
		(C.double)(alpha),         // alpha - The constant scaling factor to multiply by.
		(*C.double)(&src[0]),      // X - Input vector X.
		(C.int)(int32(strideSrc)), // incX - Stride within X. For example, if incX is 7, every 7th element is used
	)
}

// Gemm Multiplies two matrices (double-precision).
func Gemm(
	order, aTrans, bTrans uint32,
	aW, aH, bW int,
	alpha float64,
	a []float64, aDim int,
	b []float64, bDim int,
	beta float64,
	c []float64, cDim int,
) {
	// https://developer.apple.com/documentation/accelerate/1513282-cblas_dgemm
	C.cblas_dgemm(
		order, // Specifies row-major (C) or column-major (Fortran) data ordering.

		aTrans, // Specifies whether to transpose matrix A.
		bTrans, // Specifies whether to transpose matrix B.

		(C.int)(int32(aH)), // Number of rows in matrices A and C.
		(C.int)(int32(bW)), // Number of columns in matrices B and C.
		(C.int)(int32(aW)), // Number of columns in matrix A; number of rows in matrix B

		(C.double)(alpha),    // alpha - Scaling factor for the product of matrices A and B.
		(*C.double)(&a[0]),   // Matrix A.
		(C.int)(int32(aDim)), // The size of the first dimension of matrix A; if you are passing a matrix A[m][n], the value should be m.

		(*C.double)(&b[0]),   // Matrix B.
		(C.int)(int32(bDim)), // The size of the first dimension of matrix B; if you are passing a matrix B[m][n], the value should be m.

		(C.double)(beta),     // beta - Scaling factor for matrix C.
		(*C.double)(&c[0]),   // Matrix C.
		(C.int)(int32(cDim)), // The size of the first dimension of matrix C; if you are passing a matrix C[m][n], the value should be m.
	)
}
