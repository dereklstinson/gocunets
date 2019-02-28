package convolution

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Ops is a struct
type Ops struct {
	fwddesc      *gocudnn.ConvolutionD
	fwdgroup     int32
	bwdddesc     *gocudnn.ConvolutionD
	bwddgroup    int32
	bwdfdesc     *gocudnn.ConvolutionD
	bwdfgroup    int32
	helper       gocudnn.Convolution
	setfilt      bool
	pwspacesize  gocudnn.SizeT
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

//Flags returns the flags that are used for convolution
func Flags() gocudnn.ConvolutionFlags {
	return gocudnn.ConvolutionFlags{}
}

//StageOperation set sets a convolution struct default algos go as follows fwd: direct, bwdfilt: algo0, bwddata:algo0
func StageOperation(mode gocudnn.ConvolutionMode, data cudnn.DataType, pad, stride, dilation []int32) (*Ops, error) {
	helper := gocudnn.Convolution{}
	if len(pad) == 2 {

		fwddesc, err := helper.NewConvolution2dDescriptor(mode, data.Cu(), pad, stride, dilation)
		if err != nil {
			return nil, err
		}
		bwdddesc, err := helper.NewConvolution2dDescriptor(mode, data.Cu(), pad, stride, dilation)
		if err != nil {
			return nil, err
		}

		bwdfdesc, err := helper.NewConvolution2dDescriptor(mode, data.Cu(), pad, stride, dilation)
		if err != nil {
			return nil, err
		}
		return &Ops{
			fwddesc:  fwddesc,
			bwdddesc: bwdddesc,
			bwdfdesc: bwdfdesc,
			stride:   stride,
			dilation: dilation,
			pad:      pad,
		}, nil
	}
	fwddesc, err := helper.NewConvolutionNdDescriptor(mode, data.Cu(), pad, stride, dilation)
	if err != nil {
		return nil, err
	}
	bwdddesc, err := helper.NewConvolutionNdDescriptor(mode, data.Cu(), pad, stride, dilation)
	if err != nil {
		return nil, err
	}
	bwdfdesc, err := helper.NewConvolutionNdDescriptor(mode, data.Cu(), pad, stride, dilation)
	if err != nil {
		return nil, err
	}
	return &Ops{
		fwddesc:  fwddesc,
		bwdddesc: bwdddesc,
		bwdfdesc: bwdfdesc,
		stride:   stride,
		dilation: dilation,
		pad:      pad,
	}, nil

}

//FwdGroup links the convolution with a group number
func (c *Ops) FwdGroup(group int32) error {
	c.fwdgroup = group
	return c.fwddesc.SetGroupCount(group)
}

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

//Group is a group of convolution functions
type Group struct {
	g   []*Ops
	num int32
}

//FwdMakeGroup takes a slice of convolution pointers and links them into a group
func FwdMakeGroup(groupnumber int32, group []*Ops) (Group, error) {
	var err error
	for i := 0; i < len(group); i++ {
		err = group[i].FwdGroup(groupnumber)
		if err != nil {
			return Group{}, err
		}
	}
	return Group{
		g:   group,
		num: groupnumber,
	}, nil
}

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

//SetFwdMathType sets the mathtype
func (c *Ops) SetFwdMathType(math gocudnn.MathType) error {
	return c.fwddesc.SetMathType(math)
}

//SetBwdDataMathType sets the mathtype
func (c *Ops) SetBwdDataMathType(math gocudnn.MathType) error {
	return c.bwdddesc.SetMathType(math)
}

//SetBwdFiltType sets the mathtype
func (c *Ops) SetBwdFiltType(math gocudnn.MathType) error {
	return c.bwdfdesc.SetMathType(math)
}

//BwdPropData dx = alpha * BwdPropData(w,dy)+beta*dx
func (c *Ops) BwdPropData(
	handle *cudnn.Handler,
	alpha float64,
	w *tensor.Volume,
	dy *tensor.Volume,
	wspace *gocudnn.Malloced,
	beta float64,
	dx *tensor.Volume) error {

	_, dtypew, _, err := w.Properties()
	if err != nil {
		return err
	}

	a := gocudnn.CScalarByDataType(dtypew.Cu(), alpha)
	b := gocudnn.CScalarByDataType(dtypew.Cu(), beta)
	if a == nil || b == nil {
		return errors.New("Unsuported Datatype")
	}

	return c.bwdddesc.BackwardData(handle.Cudnn(), a, w.FD(), w.Memer(), dy.TD(), dy.Memer(), c.perfbackdata.Algo, wspace, b, dx.TD(), dx.Memer())
}

//BwdPropFilt dw = alpha * BwdPropFilt(x,dy)+beta*dw
func (c *Ops) BwdPropFilt(
	handle *cudnn.Handler,
	alpha float64,
	x *tensor.Volume,
	dy *tensor.Volume,
	wspace *gocudnn.Malloced,
	beta float64,
	dw *tensor.Volume) error {

	_, dtypew, _, err := dw.Properties()
	if err != nil {
		return err
	}
	a := gocudnn.CScalarByDataType(dtypew.Cu(), alpha)
	b := gocudnn.CScalarByDataType(dtypew.Cu(), beta)

	return c.bwdfdesc.BackwardFilter(handle.Cudnn(), a, x.TD(), x.Memer(), dy.TD(), dy.Memer(), c.perfbackfilt.Algo, wspace, b, dw.FD(), dw.Memer())
}

//FwdProp    y= alpha * Convolution(x,w)+ beta*y
func (c *Ops) FwdProp(
	handle *cudnn.Handler,
	alpha float64,
	x *tensor.Volume,
	w *tensor.Volume,
	wspace *gocudnn.Malloced,
	beta float64,
	y *tensor.Volume) error {

	_, dtypew, _, err := w.Properties()
	if err != nil {
		return err
	}
	a := gocudnn.CScalarByDataType(dtypew.Cu(), alpha)
	b := gocudnn.CScalarByDataType(dtypew.Cu(), beta)
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

	return c.fwddesc.Forward(handle.Cudnn(), a, x.TD(), x.Memer(), w.FD(), w.Memer(), c.perfforward.Algo, wspace, b, y.TD(), y.Memer())
}

//BwdBias does the backward bias calculation
func (c *Ops) BwdBias(
	handle *cudnn.Handler,
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
	a := gocudnn.CScalarByDataType(dtypedy.Cu(), alpha)
	b := gocudnn.CScalarByDataType(dtypedy.Cu(), beta)
	return c.helper.Funcs.Bwd.ConvolutionBackwardBias(
		handle.Cudnn(),
		a,
		dy.TD(),
		dy.Memer(),
		b,
		dbias.TD(),
		dbias.Memer(),
	)
}
