package utils

import (
	"github.com/dereklstinson/cutil"
	"github.com/dereklstinson/half"
)

//DivideAll will divide all the values in data by value
func DivideAll(data []float32, value float32) {
	for i := range data {
		data[i] /= value
	}
}

//MultiplyAll adds value to all the elements in data
func MultiplyAll(data []float32, value float32) {
	for i := range data {
		data[i] *= value
	}
}

//AddAll adds value to all the elements in data
func AddAll(data []float32, value float32) {
	for i := range data {
		data[i] += value
	}
}

//SetAllEx sets all the elements in sliceofvaltype to val.  sliceofvaltype needs to be a slice of val.  It will panic if not.
//Also unsupported slices will not work. It will take all 1d slices of go types.  It will also take slices of half.Float16.
//And slices of cutil.CScalar.
func SetAllEx(sliceofvaltype interface{}, val interface{}) {
	switch array := sliceofvaltype.(type) {
	case []int8:
		v, ok := val.(int8)
		if ok {
			for i := range array {
				array[i] = v
			}
		}
	case []float32:
		v, ok := val.(float32)
		if ok {
			for i := range array {
				array[i] = v
			}
		}

	case []int32:
		v, ok := val.(int32)
		if ok {
			for i := range array {
				array[i] = v
			}
		}
	case []int64:
		v, ok := val.(int64)
		if ok {
			for i := range array {
				array[i] = v
			}
		}
	case []uint32:
		v, ok := val.(uint32)
		if ok {
			for i := range array {
				array[i] = v
			}
		}
	case []uint64:
		v, ok := val.(uint64)
		if ok {
			for i := range array {
				array[i] = v
			}
		}
	case []int:
		v, ok := val.(int)
		if ok {
			for i := range array {
				array[i] = v
			}
		}
	case []uint:
		v, ok := val.(uint)
		if ok {
			for i := range array {
				array[i] = v
			}
		}
	case []int16:
		v, ok := val.(int16)
		if ok {
			for i := range array {
				array[i] = v
			}
		}
	case []uint16:
		v, ok := val.(uint16)
		if ok {
			for i := range array {
				array[i] = v
			}
		}
	case []float64:
		v, ok := val.(float64)
		if ok {
			for i := range array {
				array[i] = v
			}
		}
	case []uint8:
		v, ok := val.(uint8)
		if ok {
			for i := range array {
				array[i] = v
			}
		}
	case []half.Float16:
		v, ok := val.(half.Float16)
		if ok {
			for i := range array {
				array[i] = v
			}
		}
	case []cutil.CScalar:
		v, ok := val.(cutil.CScalar)
		if ok {
			for i := range array {
				array[i] = v
			}
		}
	default:
		panic("SetAllEx: Unsupported slice.")

	}
	panic("SetAllEx: val not some type of elements of sliceofvaltype")
}

//SetAll sets the data elements to value
func SetAll(data []float32, value float32) {
	for i := range data {
		data[i] = value
	}
}
