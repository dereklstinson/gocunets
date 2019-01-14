package utils

// ToInt32Slice takes a go slice and convets it to a []int32. No overflow checking
func ToInt32Slice(input interface{}) []int32 {
	switch array := input.(type) {
	case []int32:
		return array
	case []float32:
		x := make([]int32, len(array))
		for i := range array {
			x[i] = int32(array[i])
		}
		return x
	case []int8:
		x := make([]int32, len(array))
		for i := range array {
			x[i] = int32(array[i])
		}
		return x
	case []int64:
		x := make([]int32, len(array))
		for i := range array {
			x[i] = int32(array[i])
		}
		return x
	case []uint32:
		x := make([]int32, len(array))
		for i := range array {
			x[i] = int32(array[i])
		}
		return x
	case []uint64:
		x := make([]int32, len(array))
		for i := range array {
			x[i] = int32(array[i])
		}
		return x
	case []int:
		x := make([]int32, len(array))
		for i := range array {
			x[i] = int32(array[i])
		}
		return x
	case []uint:
		x := make([]int32, len(array))
		for i := range array {
			x[i] = int32(array[i])
		}
		return x
	case []int16:
		x := make([]int32, len(array))
		for i := range array {
			x[i] = int32(array[i])
		}
		return x
	case []uint16:
		x := make([]int32, len(array))
		for i := range array {
			x[i] = int32(array[i])
		}
		return x
	case []float64:
		x := make([]int32, len(array))
		for i := range array {
			x[i] = int32(array[i])
		}
		return x
	case []uint8:
		x := make([]int32, len(array))
		for i := range array {
			x[i] = int32(array[i])
		}
		return x
	}
	return nil
}

// ToUint8Slice takes a go slice and convets it to a []uint8. No overflow checking
func ToUint8Slice(input interface{}) []uint8 {
	switch array := input.(type) {
	case []uint8:
		return array
	case []float32:
		x := make([]uint8, len(array))
		for i := range array {
			x[i] = uint8(array[i])
		}
		return x
	case []int32:
		x := make([]uint8, len(array))
		for i := range array {
			x[i] = uint8(array[i])
		}
		return x
	case []int64:
		x := make([]uint8, len(array))
		for i := range array {
			x[i] = uint8(array[i])
		}
		return x
	case []uint32:
		x := make([]uint8, len(array))
		for i := range array {
			x[i] = uint8(array[i])
		}
		return x
	case []uint64:
		x := make([]uint8, len(array))
		for i := range array {
			x[i] = uint8(array[i])
		}
		return x
	case []int:
		x := make([]uint8, len(array))
		for i := range array {
			x[i] = uint8(array[i])
		}
		return x
	case []uint:
		x := make([]uint8, len(array))
		for i := range array {
			x[i] = uint8(array[i])
		}
		return x
	case []int16:
		x := make([]uint8, len(array))
		for i := range array {
			x[i] = uint8(array[i])
		}
		return x
	case []uint16:
		x := make([]uint8, len(array))
		for i := range array {
			x[i] = uint8(array[i])
		}
		return x
	case []float64:
		x := make([]uint8, len(array))
		for i := range array {
			x[i] = uint8(array[i])
		}
		return x
	case []int8:
		x := make([]uint8, len(array))
		for i := range array {
			x[i] = uint8(array[i])
		}
		return x
	}
	return nil
}

// ToInt8Slice takes a go slice and convets it to a []int8. No overflow checking
func ToInt8Slice(input interface{}) []int8 {
	switch array := input.(type) {
	case []int8:
		return array
	case []float32:
		x := make([]int8, len(array))
		for i := range array {
			x[i] = int8(array[i])
		}
		return x
	case []int32:
		x := make([]int8, len(array))
		for i := range array {
			x[i] = int8(array[i])
		}
		return x
	case []int64:
		x := make([]int8, len(array))
		for i := range array {
			x[i] = int8(array[i])
		}
		return x
	case []uint32:
		x := make([]int8, len(array))
		for i := range array {
			x[i] = int8(array[i])
		}
		return x
	case []uint64:
		x := make([]int8, len(array))
		for i := range array {
			x[i] = int8(array[i])
		}
		return x
	case []int:
		x := make([]int8, len(array))
		for i := range array {
			x[i] = int8(array[i])
		}
		return x
	case []uint:
		x := make([]int8, len(array))
		for i := range array {
			x[i] = int8(array[i])
		}
		return x
	case []int16:
		x := make([]int8, len(array))
		for i := range array {
			x[i] = int8(array[i])
		}
		return x
	case []uint16:
		x := make([]int8, len(array))
		for i := range array {
			x[i] = int8(array[i])
		}
		return x
	case []float64:
		x := make([]int8, len(array))
		for i := range array {
			x[i] = int8(array[i])
		}
		return x
	case []uint8:
		x := make([]int8, len(array))
		for i := range array {
			x[i] = int8(array[i])
		}
		return x
	}
	return nil
}

//ToFLoat64Slice takes a go slice and converts it to a []float64.
func ToFLoat64Slice(input interface{}) []float64 {
	switch array := input.(type) {
	case []float64:
		return array
	case []float32:
		x := make([]float64, len(array))
		for i := range array {
			x[i] = float64(array[i])
		}
		return x
	case []int32:
		x := make([]float64, len(array))
		for i := range array {
			x[i] = float64(array[i])
		}
		return x
	case []int64:
		x := make([]float64, len(array))
		for i := range array {
			x[i] = float64(array[i])
		}
		return x
	case []uint32:
		x := make([]float64, len(array))
		for i := range array {
			x[i] = float64(array[i])
		}
		return x
	case []uint64:
		x := make([]float64, len(array))
		for i := range array {
			x[i] = float64(array[i])
		}
		return x
	case []int:
		x := make([]float64, len(array))
		for i := range array {
			x[i] = float64(array[i])
		}
		return x
	case []uint:
		x := make([]float64, len(array))
		for i := range array {
			x[i] = float64(array[i])
		}
		return x
	case []int16:
		x := make([]float64, len(array))
		for i := range array {
			x[i] = float64(array[i])
		}
		return x
	case []uint16:
		x := make([]float64, len(array))
		for i := range array {
			x[i] = float64(array[i])
		}
		return x
	case []int8:
		x := make([]float64, len(array))
		for i := range array {
			x[i] = float64(array[i])
		}
		return x
	case []uint8:
		x := make([]float64, len(array))
		for i := range array {
			x[i] = float64(array[i])
		}
		return x
	}
	return nil
}

//ToFloat32Slice takes a go typed slice and converts it to a []float32 slice
func ToFloat32Slice(input interface{}) []float32 {
	switch array := input.(type) {
	case []float32:
		return array
	case []float64:
		x := make([]float32, len(array))
		for i := range array {
			x[i] = float32(array[i])
		}
		return x
	case []int32:
		x := make([]float32, len(array))
		for i := range array {
			x[i] = float32(array[i])
		}
		return x
	case []int64:
		x := make([]float32, len(array))
		for i := range array {
			x[i] = float32(array[i])
		}
		return x
	case []uint32:
		x := make([]float32, len(array))
		for i := range array {
			x[i] = float32(array[i])
		}
		return x
	case []uint64:
		x := make([]float32, len(array))
		for i := range array {
			x[i] = float32(array[i])
		}
		return x
	case []int:
		x := make([]float32, len(array))
		for i := range array {
			x[i] = float32(array[i])
		}
		return x
	case []uint:
		x := make([]float32, len(array))
		for i := range array {
			x[i] = float32(array[i])
		}
		return x
	case []int16:
		x := make([]float32, len(array))
		for i := range array {
			x[i] = float32(array[i])
		}
		return x
	case []uint16:
		x := make([]float32, len(array))
		for i := range array {
			x[i] = float32(array[i])
		}
		return x
	case []int8:
		x := make([]float32, len(array))
		for i := range array {
			x[i] = float32(array[i])
		}
		return x
	case []uint8:
		x := make([]float32, len(array))
		for i := range array {
			x[i] = float32(array[i])
		}
		return x
	}
	return nil
}
