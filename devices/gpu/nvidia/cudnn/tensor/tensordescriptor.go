package tensor

import (
	"github.com/dereklstinson/GoCuNets/utils"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

type tensordescriptor struct {
	index     int
	tD        *gocudnn.TensorD
	tDstrided *gocudnn.TensorD
	fD        *gocudnn.FilterD
	dims      []int32
	strides   []int32
}

func maketensordescriptor(frmt gocudnn.TensorFormat, dtype gocudnn.DataType, dims []int32) (*tensordescriptor, error) {
	var flg gocudnn.TensorFormat
	tens, err := gocudnn.CreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	filts, err := gocudnn.CreateFilterDescriptor()
	if err != nil {
		return nil, err
	}
	tenstrided, err := gocudnn.CreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	err = tens.Set(frmt, dtype, dims, nil)
	if err != nil {
		return nil, err
	}
	err = filts.Set(dtype, frmt, dims)
	if err != nil {
		return nil, err
	}
	err = tenstrided.Set(flg.Strided(), dtype, dims, utils.FindStridesInt32(dims))

	if err != nil {
		return nil, err
	}
	return &tensordescriptor{
		tD:        tens,
		tDstrided: tenstrided,
		fD:        filts,
		dims:      dims,
		strides:   utils.FindStridesInt32(dims),
	}, nil

}
