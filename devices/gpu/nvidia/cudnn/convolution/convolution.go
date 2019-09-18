package convolution

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Ops is a struct
type Ops struct {
	op           *gocudnn.ConvolutionD
	group        int32
	setfilt      bool
	pwspacesize  uint
	perfforward  ForwardPerformance
	perfbackdata BackDataPerformance
	perfbackfilt BackFilterPerformance
	pad          []int32
	dilation     []int32
	stride       []int32
}

//Pad returns the padding per dim (usually H and W) for the convolution operation
func (c *Ops) Pad() []int32 {
	return c.pad
}

//Stride or slide returns the stride per dim for the operstion (usually H and W)
func (c *Ops) Stride() []int32 {
	return c.stride
}

//Dilation returns the dilation per dim for the operstion (usually H and W)
func (c *Ops) Dilation() []int32 {
	return c.dilation
}

//StageOperation set sets a convolution struct default algos go as follows fwd: direct, bwdfilt: algo0, bwddata:algo0
func StageOperation(mode gocudnn.ConvolutionMode, data gocudnn.DataType, pad, stride, dilation []int32) (*Ops, error) {

	desc, err := gocudnn.CreateConvolutionDescriptor()
	if err != nil {
		return nil, err
	}
	err = desc.Set(mode, data, pad, stride, dilation)
	if err != nil {
		return nil, err
	}
	return &Ops{
		op: desc,
	}, nil
}

//Group links the convolution with a group number
func (c *Ops) Group(group int32) error {
	c.group = group
	return c.op.SetGroupCount(group)
}

/*
//BwdFilterGroup links the convolution with a group number
func (c *Ops) BwdFilterGroup(group int32) error {
	c.bwdfgroup = group
	return c.bwdfdesc.SetGroupCount(group)
}

//BwdDataGroup links the convolution with a group number
func (c *Ops) BwdDataGroup(group int32) error {
	c.bwddgroup = group
	return c.bwdddesc.SetGroupCount(group)
}
*/
//Group is a group of convolution functions
type Group struct {
	g   []*Ops
	num int32
}

//MakeGroup takes a slice of convolution pointers and links them into a group
func MakeGroup(groupnumber int32, group []*Ops) (Group, error) {
	var err error
	for i := 0; i < len(group); i++ {
		err = group[i].Group(groupnumber)
		if err != nil {
			return Group{}, err
		}
	}
	return Group{
		g:   group,
		num: groupnumber,
	}, nil
}

/*
//BwdDataMakeGroup takes a slice of convolution pointers and links them into a group
func BwdDataMakeGroup(groupnumber int32, group []*Ops) (Group, error) {
	var err error
	for i := 0; i < len(group); i++ {
		err = group[i].BwdDataGroup(groupnumber)
		if err != nil {
			return Group{}, err
		}
	}
	return Group{
		g:   group,
		num: groupnumber,
	}, nil
}

//BwdFiltMakeGroup takes a slice of convolution pointers and links them into a group
func BwdFiltMakeGroup(groupnumber int32, group []*Ops) (Group, error) {
	var err error
	for i := 0; i < len(group); i++ {
		err = group[i].BwdFilterGroup(groupnumber)
		if err != nil {
			return Group{}, err
		}
	}
	return Group{
		g:   group,
		num: groupnumber,
	}, nil
}
*/

//SetMathType sets the mathtype
func (c *Ops) SetMathType(math gocudnn.MathType) error {
	return c.op.SetMathType(math)
}

/*
//SetBwdDataMathType sets the mathtype
func (c *Ops) SetBwdDataMathType(math gocudnn.MathType) error {
	return c.bwdddesc.SetMathType(math)
}

//SetBwdFiltType sets the mathtype
func (c *Ops) SetBwdFiltType(math gocudnn.MathType) error {
	return c.bwdfdesc.SetMathType(math)
}
*/

//BackwardData dx = alpha * BwdPropData(w,dy)+beta*dx
func (c *Ops) BackwardData(
	handle *cudnn.Handler,
	alpha float64,
	w *tensor.Volume,
	dy *tensor.Volume,
	wspace *nvidia.Malloced,
	beta float64,
	dx *tensor.Volume) error {

	return c.op.BackwardData(handle.Cudnn(), alpha, w.FD(), w.Memer(), dy.TD(), dy.Memer(), c.perfbackdata.Algo, wspace, wspace.TotalBytes(), beta, dx.TD(), dx.Memer())
}

//BackwardFilter dw = alpha * BwdPropFilt(x,dy)+beta*dw
func (c *Ops) BackwardFilter(
	handle *cudnn.Handler,
	alpha float64,
	x *tensor.Volume,
	dy *tensor.Volume,
	wspace *nvidia.Malloced,
	beta float64,
	dw *tensor.Volume) error {
	return c.op.BackwardFilter(handle.Cudnn(), alpha, x.TD(), x.Memer(), dy.TD(), dy.Memer(), c.perfbackfilt.Algo, wspace, wspace.TotalBytes(), beta, dw.FD(), dw.Memer())
}

//Forward    y= alpha * Convolution(x,w)+ beta*y
func (c *Ops) Forward(
	handle *cudnn.Handler,
	alpha float64,
	x *tensor.Volume,
	w *tensor.Volume,
	wspace *nvidia.Malloced,
	beta float64,
	y *tensor.Volume) error {

	/*
		fmt.Println("1: ", handle)
		fmt.Println("2: ", alpha)
		fmt.Println("3: ", x.TD())
		fmt.Println("5: ", x.Memer())
		fmt.Println("6: ", w.FD())
		fmt.Println("7: ", w.Memer())
		fmt.Println("8: ", c.desc)
		fmt.Println("9: ", c.fwdalgo)
		fmt.Println("10: ", wspace)
		fmt.Println("11: ", beta)
		fmt.Println("12: ", y.TD())
		fmt.Println("13: ", y.Memer())
	*/

	return c.op.Forward(handle.Cudnn(), alpha, x.TD(), x.Memer(), w.FD(), w.Memer(), c.perfforward.Algo, wspace, wspace.TotalBytes(), beta, y.TD(), y.Memer())
}

//BackwardBias does the backward bias calculation
func (c *Ops) BackwardBias(
	handle *cudnn.Handler,
	alpha float64,
	dy *tensor.Volume,
	beta float64,
	dbias *tensor.Volume) error {

	return c.op.BackwardBias(
		handle.Cudnn(),
		alpha,
		dy.TD(),
		dy.Memer(),
		beta,
		dbias.TD(),
		dbias.Memer(),
	)
}
