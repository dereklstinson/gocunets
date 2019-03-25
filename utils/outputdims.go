package utils

func findoutputdim(x, w, s, p, d int32) int32 {
	return 1 + (x+2*p-(((w-1)*d)+1))/s
}

type convprop struct {
	input, weight, output int32

	s, p, d int32
}
type tensor struct {
	dims []int32
}
