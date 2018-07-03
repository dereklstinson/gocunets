package convolution

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Convolution is a struct
type Convolution struct {
	descriptor *gocudnn.ConvolutionD
	helper     gocudnn.Convolution
	fwdalgo    gocudnn.ConvFwdAlgo
	bwddata    gocudnn.ConvBwdDataAlgo
	bwdfilt    gocudnn.ConvBwdFiltAlgo
	dims       int
	group      int
}

func Flags() gocudnn.ConvolutionFlags {
	return gocudnn.ConvolutionFlags{}
}

func Create(mode gocudnn.ConvolutionMode, data gocudnn.DataType, pad, stride, dilation []int32) (*Convolution, error) {
	helper := gocudnn.Convolution{}
	if len(pad) == 2 {
		desc, err := helper.NewConvolution2dDescriptor(mode, data, pad, stride, dilation)
		if err != nil {
			return nil, err
		}
		return &Convolution{
			descriptor: desc,
			dims:       len(pad),
		}, nil
	} else {
		desc, err := helper.NewConvolutionNdDescriptor(mode, data, pad, stride, dilation)
		if err != nil {
			return nil, err
		}
		return &Convolution{
			descriptor: desc,
			dims:       len(pad),
		}, nil
	}

}
func (c *Convolution) Group(group int32) error {
	return c.descriptor.SetGroupCount(group)
}

func MakeGroup(groupnumber int32, group []*Convolution) error {
	var err error
	for i := 0; i < len(group); i++ {
		err = group[i].descriptor.SetGroupCount(groupnumber)
		if err != nil {
			return err
		}
	}
	return nil
}

func (c *Convolution) SetMathType(math gocudnn.MathType) error {
	return c.descriptor.SetMathType(math)
}

func (c *Convolution) AlgoLists(handle *gocudnn.Handle, x, w, y *tensor.Tensor) ([]gocudnn.ConvFwdAlgoPerformance, error) {
	maxfwd, err := c.helper.Funcs.Fwd.GetConvolutionForwardAlgorithmMaxCount(handle)
	if err != nil {
		return nil, err
	}
	fwdlist, err := c.helper.Funcs.Fwd.FindConvolutionForwardAlgorithm(handle, x.TD(), w.FD(), c.descriptor, y.TD(), maxfwd)
	if err != nil {
		return nil, err
	}
	return fwdlist, nil
}

func (c *Convolution) OutputDim(input *tensor.Tensor, filter *tensor.Tensor) ([]int32, error) {
	_, _, dims, err := input.Properties()
	if err != nil {
		return nil, err
	}
	_, _, fdims, err := filter.Properties()
	if err != nil {
		return nil, err
	}
	if len(dims) != len(fdims) {
		return nil, errors.New("length of dims not same")
	}
	if len(dims) == 4 {
		return c.descriptor.GetConvolution2dForwardOutputDim(input.TD(), filter.FD())
	}
	return c.descriptor.GetConvolutionNdForwardOutputDim(input.TD(), filter.FD())

}
func (c *Convolution) WorkSizeFwd(handle *gocudnn.Handle, x, w, y tensor.Tensor) (gocudnn.SizeT, error) {
	return c.helper.Funcs.Fwd.GetConvolutionForwardWorkspaceSize(handle, x.TD(), w.FD(), c.descriptor, y.TD(), c.fwdalgo)
}
func (c *Convolution) FwdProp(handle *gocudnn.Handle)
func (c *Convolution) Destroy() error {
	err := c.descriptor.DestroyDescriptor()
	if err != nil {
		return err
	}
	c.group = 0
	c.dims = 0
	return nil
}
