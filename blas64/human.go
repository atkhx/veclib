package blas64

func MatrixMultiplyAB(aW int, a, b, c []float64, alpha, beta float64) {
	aH := len(a) / aW
	bW := len(b) / aW

	Gemm(CBLASOrderRowMajor, CBLASNoTrans, CBLASNoTrans,
		aW, aH, bW,
		alpha,

		a, aW,
		b, bW,

		beta,
		c, bW,
	)
}

func MatrixMultiplyAonTransposedB(aW int, a, b, c []float64, alpha, beta float64) {
	aH := len(a) / aW
	bW := len(b) / aW

	Gemm(CBLASOrderRowMajor, CBLASNoTrans, CBLASTrans,
		aW, aH, bW,
		alpha,

		a, aW,
		b, aW, // change bW to aW

		beta,
		c, bW,
	)
}

func MatrixMultiplyTransposedAonB(aW int, a, b, c []float64, alpha, beta float64) {
	// aW - ширина уже транспонированной матрицы

	aH := len(a) / aW
	bW := len(b) / aW

	Gemm(CBLASOrderRowMajor, CBLASTrans, CBLASNoTrans,
		aW, aH, bW,
		alpha,

		a, aH, // change aW to aH
		b, bW,

		beta,
		c, bW,
	)
}
