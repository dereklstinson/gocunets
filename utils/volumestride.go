package utils

//FindStridesInt32 returns the strides of the dims given for an array.
//example if for an array of NCHW it will  return [C*H*W,H*W,W,1] those can be used for traversing a 4d array
func FindStridesInt32(dims []int32) (strides []int32) {
	mult := int32(1)
	strides = make([]int32, len(dims))
	for i := len(dims) - 1; i >= 0; i-- {
		strides[i] = mult
		mult *= dims[i]
	}

	return strides

}

//FinStridesInt returns the strides of the dims given for an array.
//example if for an array of NCHW it will  return [C*H*W,H*W,W,1] those can be used for traversing a 4d array
func FinStridesInt(dims []int) (strides []int) {
	mult := 1
	strides = make([]int, len(dims))
	for i := len(dims) - 1; i >= 0; i-- {
		strides[i] = mult
		mult *= dims[i]
	}

	return strides
}

//FindVolumeInt32 returns the total volume in int32
func FindVolumeInt32(dims []int32) int32 {
	mult := int32(1)

	for i := len(dims) - 1; i >= 0; i-- {

		mult *= dims[i]
	}

	return mult
}

//FindVolumeInt returns the total volume in int
func FindVolumeInt(dims []int) int {
	mult := 1

	for i := len(dims) - 1; i >= 0; i-- {

		mult *= dims[i]
	}

	return mult
}

//AbsoluteValue returns the absolutevalue in float32
func AbsoluteValue(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
