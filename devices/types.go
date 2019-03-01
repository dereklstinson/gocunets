/*
Package devices is kind of a pipe dream.  Where I and others can can add different libraries to the go platform.  I will probably make this its own package one day.

*/
package devices

import (
	"unsafe"

	"github.com/dereklstinson/half"
)

/*
//SizeT is used commonly to specify the number of bytes a chunck of data contains
type SizeT uint

//Uint returns the uint value of SizeT
func (s SizeT) Uint() uint {
	return uint(s)
}

*/

//MakeFloat16Slice makes slice of float16 from a slice of float32
func MakeFloat16Slice(vals []float32) []half.Float16 {
	x := make([]half.Float16, len(vals))
	for i := range vals {
		x[i] = half.NewFloat16(vals[i])
	}
	return x
}

//Slice interfaces with device memory similarly to go slices
type Slice interface {
	Set(value interface{}, offset uint)
	Get(value interface{}, offset uint)
	Length() uint
	Type() Type
}

//Type are used to determine allocation size and offset array stuff so it is easier to handle memory copying
//Between devices
type Type uint

//These are flags to determine data type for device slices
const (
	Uint8 = Type(iota)
	Int8
	Uint16
	Int16
	Uint32
	Int32
	Uint
	Int
	Uint64
	Int64
	Float16H
	Float32
	Float64
	UNKNOWN
)

//Typer returns a Type equivilent to its own type
type Typer interface {
	Type() Type
}

//SizeOf returns the SizeT which is the size in bytes of the type
func (t Type) SizeOf() uint {
	switch t {
	case Uint8:
		return uint(1)
	case Int8:
		return uint(1)
	case Uint16:
		return uint(2)
	case Int16:
		return uint(2)
	case Uint32:
		return uint(4)
	case Int32:
		return uint(4)
	case Uint:
		var x uint
		return uint(unsafe.Sizeof(x))
	case Int:
		var x int
		return uint(unsafe.Sizeof(x))
	case Uint64:
		return uint(8)
	case Int64:
		return uint(8)
	case Float16H:
		return uint(2)
	case Float32:
		return uint(4)
	case Float64:
		return uint(8)

	}
	return 0
}
