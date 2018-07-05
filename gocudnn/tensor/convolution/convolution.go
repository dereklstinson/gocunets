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

//Flags returns the flags that are used for convolution
func Flags() gocudnn.ConvolutionFlags {
	return gocudnn.ConvolutionFlags{}
}

//Create Creates a convolution struct
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
	}
	desc, err := helper.NewConvolutionNdDescriptor(mode, data, pad, stride, dilation)
	if err != nil {
		return nil, err
	}
	return &Convolution{
		descriptor: desc,
		dims:       len(pad),
	}, nil

}

//Group links the convolution with a group number
func (c *Convolution) Group(group int32) error {
	return c.descriptor.SetGroupCount(group)
}

//MakeGroup takes a slice of convolution pointers and links them into a group
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

//SetMathType sets the mathtype
func (c *Convolution) SetMathType(math gocudnn.MathType) error {
	return c.descriptor.SetMathType(math)
}

//AlgoLists Algo lists returns slices of performances for the fwd algos and bwd algos
func (c *Convolution) AlgoLists(handle *gocudnn.Handle, x, dx, w, dw, y, dy *tensor.Tensor) ([]gocudnn.ConvFwdAlgoPerformance, []gocudnn.ConvBwdDataAlgoPerformance, []gocudnn.ConvBwdFiltAlgoPerformance, error) {
	maxfwd, err := c.helper.Funcs.Fwd.GetConvolutionForwardAlgorithmMaxCount(handle)
	if err != nil {
		return nil, nil, nil, err
	}
	fwdlist, err := c.helper.Funcs.Fwd.FindConvolutionForwardAlgorithm(handle, x.TD(), w.FD(), c.descriptor, y.TD(), maxfwd)
	if err != nil {
		return nil, nil, nil, err
	}
	maxbwddata, err := c.helper.Funcs.Bwd.GetConvolutionBackwardDataAlgorithmMaxCount(handle)
	if err != nil {
		return nil, nil, nil, err
	}
	bwddata, err := c.helper.Funcs.Bwd.FindConvolutionBackwardDataAlgorithm(handle, w.FD(), dy.TD(), c.descriptor, dx.TD(), maxbwddata)
	if err != nil {
		return nil, nil, nil, err
	}
	maxbwdfilt, err := c.helper.Funcs.Bwd.GetConvolutionBackwardFilterAlgorithmMaxCount(handle)
	if err != nil {
		return nil, nil, nil, err
	}
	bwdfilt, err := c.helper.Funcs.Bwd.FindConvolutionBackwardFilterAlgorithm(handle, x.TD(), dy.TD(), c.descriptor, dw.FD(), maxbwdfilt)
	if bwdfilt != nil {
		return nil, nil, nil, err
	}
	return fwdlist, bwddata, bwdfilt, nil
}

//OutputDim will return the dims of what the output tensor should be
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

//WorkSizeFwd returns the worksize for the forward algo
func (c *Convolution) WorkSizeFwd(handle *gocudnn.Handle, x, w, y tensor.Tensor) (gocudnn.SizeT, error) {
	return c.helper.Funcs.Fwd.GetConvolutionForwardWorkspaceSize(handle, x.TD(), w.FD(), c.descriptor, y.TD(), c.fwdalgo)
}

//SetFwdAlgo sets fwd algo
func (c *Convolution) SetFwdAlgo(algo gocudnn.ConvFwdAlgo) {
	c.fwdalgo = algo
}

//SetBwdDataAlgo sets the backward data algo
func (c *Convolution) SetBwdDataAlgo(algo gocudnn.ConvBwdDataAlgo) {
	c.bwddata = algo
}

//SetBwdFiltAlgo sets the backward filters
func (c *Convolution) SetBwdFiltAlgo(algo gocudnn.ConvBwdFiltAlgo) {
	c.bwdfilt = algo
}

//SetAlgos sets all the algos
func (c *Convolution) SetAlgos(fwd gocudnn.ConvFwdAlgo, bwdd gocudnn.ConvBwdDataAlgo, bwdf gocudnn.ConvBwdFiltAlgo) {
	c.fwdalgo = fwd
	c.bwddata = bwdd
	c.bwdfilt = bwdf
}

//BwdPropData dx = alpha * BwdPropData(w,dy)+beta*dx
func (c *Convolution) BwdPropData(
	handle *gocudnn.Handle,
	alpha float64,
	w *tensor.Tensor,
	dy *tensor.Tensor,
	wspace gocudnn.Memer,
	beta float64,
	dx *tensor.Tensor) error {
	t := tensor.Flags()
	var a gocudnn.CScalar
	var b gocudnn.CScalar

	_, dtypew, _, err := w.Properties()
	if err != nil {
		return err
	}
	switch dtypew {
	case t.Data.Double():
		a = gocudnn.CDouble(alpha)
		b = gocudnn.CDouble(beta)
	case t.Data.Float():
		a = gocudnn.CFloat(alpha)
		b = gocudnn.CFloat(beta)
	case t.Data.Int32():
		a = gocudnn.CInt(alpha)
		b = gocudnn.CInt(beta)
	case t.Data.Int8():
		a = gocudnn.CInt8(alpha)
		b = gocudnn.CInt8(beta)
	case t.Data.UInt8():
		a = gocudnn.CUInt8(alpha)
		b = gocudnn.CUInt8(beta)

	default:
		return errors.New("Not supported Format")
	}
	return c.helper.Funcs.Bwd.ConvolutionBackwardData(handle, a, w.FD(), w.Memer(), dy.TD(), dy.Memer(), c.descriptor, c.bwddata, wspace, b, dx.TD(), dx.Memer())
}

//BwdPropFilt dw = alpha * BwdPropFilt(x,dy)+beta*dw
func (c *Convolution) BwdPropFilt(
	handle *gocudnn.Handle,
	alpha float64,
	x *tensor.Tensor,

	dy *tensor.Tensor,
	wspace gocudnn.Memer,
	beta float64,
	dw *tensor.Tensor) error {
	t := tensor.Flags()
	var a gocudnn.CScalar
	var b gocudnn.CScalar

	_, dtypew, _, err := dw.Properties()
	if err != nil {
		return err
	}
	switch dtypew {
	case t.Data.Double():
		a = gocudnn.CDouble(alpha)
		b = gocudnn.CDouble(beta)
	case t.Data.Float():
		a = gocudnn.CFloat(alpha)
		b = gocudnn.CFloat(beta)
	case t.Data.Int32():
		a = gocudnn.CInt(alpha)
		b = gocudnn.CInt(beta)
	case t.Data.Int8():
		a = gocudnn.CInt8(alpha)
		b = gocudnn.CInt8(beta)
	case t.Data.UInt8():
		a = gocudnn.CUInt8(alpha)
		b = gocudnn.CUInt8(beta)

	default:
		return errors.New("Not supported Format")
	}
	return c.helper.Funcs.Bwd.ConvolutionBackwardFilter(handle, a, x.TD(), x.Memer(), dy.TD(), dy.Memer(), c.descriptor, c.bwdfilt, wspace, b, dw.FD(), dw.Memer())
}

//FwdProp    y= alpha * Convolution(x,w)+ beta*y
func (c *Convolution) FwdProp(
	handle *gocudnn.Handle,
	alpha float64,
	x *tensor.Tensor,
	w *tensor.Tensor,
	wspace gocudnn.Memer,
	beta float64,
	y *tensor.Tensor) error {
	t := tensor.Flags()

	var a gocudnn.CScalar
	var b gocudnn.CScalar

	_, dtypew, _, err := w.Properties()
	if err != nil {
		return err
	}
	switch dtypew {
	case t.Data.Double():
		a = gocudnn.CDouble(alpha)
		b = gocudnn.CDouble(beta)
	case t.Data.Float():
		a = gocudnn.CFloat(alpha)
		b = gocudnn.CFloat(beta)
	case t.Data.Int32():
		a = gocudnn.CInt(alpha)
		b = gocudnn.CInt(beta)
	case t.Data.Int8():
		a = gocudnn.CInt8(alpha)
		b = gocudnn.CInt8(beta)
	case t.Data.UInt8():
		a = gocudnn.CUInt8(alpha)
		b = gocudnn.CUInt8(beta)

	default:
		return errors.New("Not supported Format")
	}
	return c.helper.Funcs.Fwd.ConvolutionForward(handle, a, x.TD(), x.Memer(), w.FD(), w.Memer(), c.descriptor, c.fwdalgo, wspace, b, y.TD(), y.Memer())
}

//Destroy destroys the convolution descriptor and sets everything else that isn't a pointer to zero
func (c *Convolution) Destroy() error {
	err := c.descriptor.DestroyDescriptor()
	if err != nil {
		return err
	}
	c.group = 0
	c.dims = 0
	return nil
}
