package gomem

import (
	"errors"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

//TensorND is a struct that holds all sorts of different slices that are not visible to the user.
type TensorND struct {
	dataF32  []float32
	dataF64  []float64
	dataI32  []int32
	dataI8   []int8
	dataU8   []uint8
	dims     []int32
	slide    []int32
	datatype DataType
	xl       bool
	frmt     TensorFormat
}

//DataType is a type that is used for flags.DataTypeFlag will pass DataType through methods
type DataType int

//DataTypeFlag is used to pass DataType values through methods
type DataTypeFlag struct {
}

//Float32 makes the datatype Float32
func (d DataTypeFlag) Float32() DataType {
	return DataType(1)
}

//Float64 makes a datatype Float64
func (d DataTypeFlag) Float64() DataType {
	return DataType(2)
}

//Int32 makes a datatype Int32
func (d DataTypeFlag) Int32() DataType {
	return DataType(3)
}

//UInt8 makes a datatype UInt8
func (d DataTypeFlag) UInt8() DataType {
	return DataType(6)
}

//Int8 makes a datatype Int8
func (d DataTypeFlag) Int8() DataType {
	return DataType(7)
}

//TensorFormat is used as a flag for the different types of tensors
type TensorFormat int

//TensorFormatFlag passes TensorFormat through methods
type TensorFormatFlag struct {
}

//NCHW makes the tensor NCHW
func (t TensorFormatFlag) NCHW() TensorFormat {
	return TensorFormat(1)
}

//NHWC makes the tensor NHWC
func (t TensorFormatFlag) NHWC() TensorFormat {
	return TensorFormat(2)
}

//NewTensorND Creates a tensor based on the dims frmt and dtype passed.  Slide is for future reference if not nil it will return an error
func NewTensorND(dims, slide []int32, frmt TensorFormat, dtype DataType) (*TensorND, error) {
	if len(dims) > 8 {
		return nil, errors.New("Too many Dims")
	}
	mult := int32(1)

	for i := 0; i < len(dims); i++ {

		mult *= dims[i]
	}
	if slide == nil {
		mult = int32(1)
		slide := make([]int32, len(dims))
		for i := len(dims) - 1; i >= 0; i-- {
			slide[i] = mult
			mult *= dims[i]
		}

	} else {
		return nil, errors.New("slide must be nil for the time being")
		/*
			mult := int32(1)
			slide2 := make([]int32, len(dims))
			for i := len(dims) - 1; i >= 0; i-- {
				slide2[i] = mult
				mult *= dims[i]
			}
			for i := range slide {
				if slide[i] < slide2[i] {
					return nil, errors.New("Slide Must be larger than fully packed")
				}
			}
		*/
	}

	var dflg DataTypeFlag
	switch dtype {
	case dflg.Float32():
		data := make([]float32, mult)
		return &TensorND{
			dataF32: data,
			dims:    dims,
			frmt:    frmt,
			slide:   slide,
		}, nil
	case dflg.Float64():
		data := make([]float64, mult)
		return &TensorND{
			dataF64: data,
			dims:    dims,
			frmt:    frmt,
			slide:   slide,
		}, nil
	case dflg.Int32():
		data := make([]int32, mult)
		return &TensorND{
			dataI32: data,
			dims:    dims,
			frmt:    frmt,
			slide:   slide,
		}, nil
	case dflg.Int8():
		data := make([]int8, mult)
		return &TensorND{
			dataI8: data,
			dims:   dims,
			frmt:   frmt,
			slide:  slide,
		}, nil
	case dflg.UInt8():
		data := make([]uint8, mult)
		return &TensorND{
			dataU8: data,
			dims:   dims,
			frmt:   frmt,
			slide:  slide,
		}, nil
	}
	return nil, errors.New("Unsupported Datatype")
}

//Dims returns the dims
func (h *TensorND) Dims() []int32 {
	return h.dims
}

//Slide will return the slide
func (h *TensorND) Slide() []int32 {
	return h.slide
}

//Frmt returns the format
func (h *TensorND) Frmt() TensorFormat {
	return h.frmt
}

//DataType returns the Format
func (h *TensorND) DataType() DataType {
	return h.datatype
}

//CudnnGoPointer returns a gocudnn.GoPointer to use with cuda
func (h *TensorND) CudnnGoPointer() (*gocudnn.GoPointer, error) {
	var dflg DataTypeFlag
	switch h.datatype {
	case dflg.Float32():
		return gocudnn.MakeGoPointer(h.dataF32)
	case dflg.Float64():
		return gocudnn.MakeGoPointer(h.dataF64)
	case dflg.Int32():
		return gocudnn.MakeGoPointer(h.dataI32)
	case dflg.Int8():
		return gocudnn.MakeGoPointer(h.dataI8)
	case dflg.UInt8():
		return gocudnn.MakeGoPointer(h.dataU8)
	}
	return nil, errors.New("Unsupported Datatype")
}
