package tensor

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/utils"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

type tensordescriptor struct {
	index     int
	tD        *gocudnn.TensorD
	tDstrided *gocudnn.TensorD
	fD        *gocudnn.FilterD

	dims    []int32
	strides []int32
}

func maketensordescriptor(frmt cudnn.TensorFormat, dtype cudnn.DataType, dims []int32) (*tensordescriptor, error) {

	var thelper gocudnn.Tensor
	var fhelper gocudnn.Filter
	if len(dims) < 4 {
		return nil, errors.New("Dims less than 4. Create A 4 dim Tensor and set dims not needed to 1")
	}

	if len(dims) > 4 {
		tens, err := thelper.NewTensorNdDescriptorEx(frmt.Cu(), dtype.Cu(), dims)
		if err != nil {
			return nil, err
		}
		filts, err := fhelper.NewFilterNdDescriptor(dtype.Cu(), frmt.Cu(), dims)
		if err != nil {
			return nil, err
		}
		tensstrided, err := thelper.NewTensorNdDescriptor(dtype.Cu(), dims, utils.FindStridesInt32(dims))
		if err != nil {
			return nil, err
		}
		return &tensordescriptor{
			tD:        tens,
			tDstrided: tensstrided,
			fD:        filts,
			dims:      dims,
			strides:   utils.FindStridesInt32(dims),
		}, nil
	}

	tens, err := thelper.NewTensor4dDescriptor(dtype.Cu(), frmt.Cu(), dims)
	if err != nil {
		return nil, err
	}
	tensstrided, err := thelper.NewTensor4dDescriptorEx(dtype.Cu(), dims, utils.FindStridesInt32(dims))
	if err != nil {
		return nil, err
	}
	filts, err := fhelper.NewFilter4dDescriptor(dtype.Cu(), frmt.Cu(), dims)
	if err != nil {
		return nil, err
	}

	return &tensordescriptor{
		tD:        tens,
		tDstrided: tensstrided,
		fD:        filts,
		dims:      dims,
		strides:   utils.FindStridesInt32(dims),
	}, nil

}
