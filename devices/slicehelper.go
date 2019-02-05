package devices

import "github.com/dereklstinson/half"

//Len returns the length of the slice passed
func Len(slice interface{}) uint {
	switch x := slice.(type) {
	case []uint8:
		return uint(len(x))
	case []int8:
		return uint(len(x))
	case []uint16:
		return uint(len(x))
	case []int16:
		return uint(len(x))
	case []uint32:
		return uint(len(x))
	case []int32:
		return uint(len(x))
	case []uint64:
		return uint(len(x))
	case []int64:
		return uint(len(x))
	case []uint:
		return uint(len(x))
	case []int:
		return uint(len(x))
	case []Float16:
		return uint(len(x))
	case []half.Float16:
		return uint(len(x))
	case []float32:
		return uint(len(x))
	case []float64:
		return uint(len(x))
	case uint8:
		return 1
	case int8:
		return 1
	case uint16:
		return 1
	case int16:
		return 1
	case uint32:
		return 1
	case int32:
		return 1
	case uint64:
		return 1
	case int64:
		return 1
	case uint:
		return 1
	case int:
		return 1
	case Float16:
		return 1
	case half.Float16:
		return 1
	case float32:
		return 1
	case float64:
		return 1

	}
	return 0
}
