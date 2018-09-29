package convolution

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Ops is a struct
type Ops struct {
	desc    *gocudnn.ConvolutionD
	helper  gocudnn.Convolution
	fwdalgo gocudnn.ConvFwdAlgo
	bwddata gocudnn.ConvBwdDataAlgo
	bwdfilt gocudnn.ConvBwdFiltAlgo
	dims    int
	group   int
}
type OpInfo struct {
	CMode       gocudnn.ConvolutionMode `json:"ConvolutionMode"`
	Dtype       gocudnn.DataType        `json:"DataType"`
	Pad         []int32                 `json:"Pad"`
	Stride      []int32                 `json:"Stride"`
	Dilation    []int32                 `json:"Dilation"`
	FwdAlgo     gocudnn.ConvFwdAlgo     `json:"FwdAlgo"`
	BwdDataAlgo gocudnn.ConvBwdDataAlgo `json:"BwdDataAlgo"`
	BwdFiltAlgo gocudnn.ConvBwdFiltAlgo `json:"BwdFiltAlgo"`
	Group       int                     `json:"Group"`
}

//Stage stages/sets up an Ops and returns a pointer to it with the info stored in the info type
func (input OpInfo) Stage() (*Ops, error) {
	helper := gocudnn.Convolution{}
	if len(input.Pad) == 2 {
		desc, err := helper.NewConvolution2dDescriptor(input.CMode, input.Dtype, input.Pad, input.Stride, input.Dilation)
		if err != nil {
			return nil, err
		}
		return &Ops{
			desc:    desc,
			dims:    len(input.Pad),
			fwdalgo: input.FwdAlgo,
			bwddata: input.BwdDataAlgo,
			bwdfilt: input.BwdFiltAlgo,
			group:   input.Group,
		}, nil
	}
	desc, err := helper.NewConvolutionNdDescriptor(input.CMode, input.Dtype, input.Pad, input.Stride, input.Dilation)
	if err != nil {
		return nil, err
	}
	return &Ops{
		desc:    desc,
		dims:    len(input.Pad),
		fwdalgo: input.FwdAlgo,
		bwddata: input.BwdDataAlgo,
		bwdfilt: input.BwdFiltAlgo,
		group:   input.Group,
	}, nil
}

//Flags returns the flags that are used for convolution
func Flags() gocudnn.ConvolutionFlags {
	return gocudnn.ConvolutionFlags{}
}

//StageOperation set sets a convolution struct default algos go as follows fwd: direct, bwdfilt: algo0, bwddata:algo0
func StageOperation(mode gocudnn.ConvolutionMode, data gocudnn.DataType, pad, stride, dilation []int32) (*Ops, error) {
	helper := gocudnn.Convolution{}
	if len(pad) == 2 {

		desc, err := helper.NewConvolution2dDescriptor(mode, data, pad, stride, dilation)
		if err != nil {
			return nil, err
		}
		return &Ops{
			desc: desc,
			dims: len(pad),
		}, nil
	}
	desc, err := helper.NewConvolutionNdDescriptor(mode, data, pad, stride, dilation)
	if err != nil {
		return nil, err
	}
	return &Ops{
		desc:    desc,
		dims:    len(pad),
		fwdalgo: helper.Flgs.Fwd.Algo.Direct(),
		bwddata: helper.Flgs.Bwd.DataAlgo.Algo0(),
		bwdfilt: helper.Flgs.Bwd.FltrAlgo.Algo0(),
	}, nil

}

//Info returns an info struct and error.  Info is usually used for saving the data to a json file.
func (c *Ops) Info() (OpInfo, error) {
	mode, dtype, pad, stride, dilation, err := c.desc.GetDescriptor()
	return OpInfo{
		CMode:       mode,
		Dtype:       dtype,
		Pad:         pad,
		Stride:      stride,
		Dilation:    dilation,
		FwdAlgo:     c.fwdalgo,
		BwdDataAlgo: c.bwddata,
		BwdFiltAlgo: c.bwdfilt,
		Group:       c.group,
	}, err
}

//Group links the convolution with a group number
func (c *Ops) Group(group int32) error {
	return c.desc.SetGroupCount(group)
}

//Group is a group of convolution functions
type Group struct {
	g   []*Ops
	num int32
}

//MakeGroup takes a slice of convolution pointers and links them into a group
func MakeGroup(groupnumber int32, group []*Ops) (Group, error) {
	var err error
	for i := 0; i < len(group); i++ {
		err = group[i].desc.SetGroupCount(groupnumber)
		if err != nil {
			return Group{}, err
		}
	}
	return Group{
		g:   group,
		num: groupnumber,
	}, nil
}

//SetMathType sets the mathtype
func (c *Ops) SetMathType(math gocudnn.MathType) error {
	return c.desc.SetMathType(math)
}

//WorkSizeFwd returns the worksize for the forward algo
func (c *Ops) WorkSizeFwd(handle *gocudnn.Handle, x, w, y tensor.Volume) (gocudnn.SizeT, error) {
	return c.helper.Funcs.Fwd.GetConvolutionForwardWorkspaceSize(handle, x.TD(), w.FD(), c.desc, y.TD(), c.fwdalgo)
}

//SetFwdAlgo sets fwd algo
func (c *Ops) SetFwdAlgo(algo gocudnn.ConvFwdAlgo) {
	c.fwdalgo = algo
}

//SetBwdDataAlgo sets the backward data algo
func (c *Ops) SetBwdDataAlgo(algo gocudnn.ConvBwdDataAlgo) {
	c.bwddata = algo
}

//SetBwdFiltAlgo sets the backward filters
func (c *Ops) SetBwdFiltAlgo(algo gocudnn.ConvBwdFiltAlgo) {
	c.bwdfilt = algo
}

//SetAlgos sets all the algos
func (c *Ops) SetAlgos(fwd gocudnn.ConvFwdAlgo, bwdd gocudnn.ConvBwdDataAlgo, bwdf gocudnn.ConvBwdFiltAlgo) {
	c.fwdalgo = fwd
	c.bwddata = bwdd
	c.bwdfilt = bwdf
}

//BwdPropData dx = alpha * BwdPropData(w,dy)+beta*dx
func (c *Ops) BwdPropData(
	handle *gocudnn.Handle,
	alpha float64,
	w *tensor.Volume,
	dy *tensor.Volume,
	wspace gocudnn.Memer,
	beta float64,
	dx *tensor.Volume) error {
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
	return c.helper.Funcs.Bwd.ConvolutionBackwardData(handle, a, w.FD(), w.Memer(), dy.TD(), dy.Memer(), c.desc, c.bwddata, wspace, b, dx.TD(), dx.Memer())
}

//BwdPropFilt dw = alpha * BwdPropFilt(x,dy)+beta*dw
func (c *Ops) BwdPropFilt(
	handle *gocudnn.Handle,
	alpha float64,
	x *tensor.Volume,

	dy *tensor.Volume,
	wspace gocudnn.Memer,
	beta float64,
	dw *tensor.Volume) error {

	_, dtypew, _, err := dw.Properties()
	if err != nil {
		return err
	}
	a := gocudnn.CScalarByDataType(dtypew, alpha)
	b := gocudnn.CScalarByDataType(dtypew, beta)
	return c.helper.Funcs.Bwd.ConvolutionBackwardFilter(handle, a, x.TD(), x.Memer(), dy.TD(), dy.Memer(), c.desc, c.bwdfilt, wspace, b, dw.FD(), dw.Memer())
}

//FwdProp    y= alpha * Convolution(x,w)+ beta*y
func (c *Ops) FwdProp(
	handle *gocudnn.Handle,
	alpha float64,
	x *tensor.Volume,
	w *tensor.Volume,
	wspace gocudnn.Memer,
	beta float64,
	y *tensor.Volume) error {

	_, dtypew, _, err := w.Properties()
	if err != nil {
		return err
	}
	a := gocudnn.CScalarByDataType(dtypew, alpha)
	b := gocudnn.CScalarByDataType(dtypew, beta)
	/*
		fmt.Println("1: ", handle)
		fmt.Println("2: ", a)
		fmt.Println("3: ", x.TD())
		fmt.Println("5: ", x.Memer())
		fmt.Println("6: ", w.FD())
		fmt.Println("7: ", w.Memer())
		fmt.Println("8: ", c.desc)
		fmt.Println("9: ", c.fwdalgo)
		fmt.Println("10: ", wspace)
		fmt.Println("11: ", b)
		fmt.Println("12: ", y.TD())
		fmt.Println("13: ", y.Memer())
	*/
	return c.helper.Funcs.Fwd.ConvolutionForward(handle, a, x.TD(), x.Memer(), w.FD(), w.Memer(), c.desc, c.fwdalgo, wspace, b, y.TD(), y.Memer())
}

//BwdBias does the backward bias calculation
func (c *Ops) BwdBias(
	handle *gocudnn.Handle,
	alpha float64,
	dy *tensor.Volume,
	beta float64,
	dbias *tensor.Volume) error {

	_, dtypedy, _, err := dy.Properties()
	if err != nil {
		return err
	}
	_, dtypedbias, _, err := dbias.Properties()
	if err != nil {
		return err
	}
	if dtypedbias != dtypedy {
		return errors.New("bias and y not same")
	}
	a := gocudnn.CScalarByDataType(dtypedy, alpha)
	b := gocudnn.CScalarByDataType(dtypedy, beta)
	return c.helper.Funcs.Bwd.ConvolutionBackwardBias(
		handle,
		a,
		dy.TD(),
		dy.Memer(),
		b,
		dbias.TD(),
		dbias.Memer(),
	)
}

//Destroy destroys the convolution descriptor and sets everything else that isn't a pointer to zero
func (c *Ops) Destroy() error {
	err := c.desc.DestroyDescriptor()
	if err != nil {
		return err
	}
	c.group = 0
	c.dims = 0
	return nil
}
