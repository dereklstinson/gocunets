package deconvolution

import (
	"fmt"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Ops is a struct
type Ops struct {
	op           *gocudnn.DeConvolutionD
	group        int32
	setfilt      bool
	pwspacesize  uint
	perfforward  ForwardPerformance
	perfbackdata BackDataPerformance
	perfbackfilt BackFilterPerformance
	mtforward    gocudnn.MathType
	mtbwdd       gocudnn.MathType
	mtbwdf       gocudnn.MathType
	pad          []int32
	dilation     []int32
	stride       []int32
}

func (c *Ops) String() string {
	return fmt.Sprintf("DeConvolutionOps{\n%v\n%v\n%v\n%v\n%v\n}\n", c.op, c.perfforward, c.perfbackdata, c.perfbackfilt, c.mtforward)
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
func StageOperation(mode gocudnn.ConvolutionMode, data gocudnn.DataType, mathtype gocudnn.MathType, group int32, pad, stride, dilation []int32) (conv *Ops, err error) {
	conv = new(Ops)
	conv.pad, conv.stride, conv.dilation = pad, stride, dilation

	conv.op, err = gocudnn.CreateDeConvolutionDescriptor()
	if err != nil {
		return nil, err
	}
	err = conv.op.Set(mode, data, pad, stride, dilation)
	if err != nil {
		return nil, err
	}
	err = conv.op.SetMathType(mathtype)
	if err != nil {
		return nil, err
	}
	err = conv.op.SetGroupCount(group)
	if err != nil {
		return nil, err
	}
	return conv, nil
}

//SetMathType will set the mathtype
func (c *Ops) SetMathType(mtype gocudnn.MathType) error {
	return c.op.SetMathType(mtype)

}

//BackwardData dx = alpha * BwdPropData(w,dy)+beta*dx
func (c *Ops) BackwardData(
	handle *cudnn.Handler,
	alpha float64,
	w *tensor.Volume,
	dy *tensor.Volume,
	wspace *nvidia.Malloced,
	beta float64,
	dx *tensor.Volume) error {

	if wspace == nil {
		err := c.op.BackwardData(handle.Cudnn(), alpha, w.FD(), w, dy.TD(), dy, c.perfbackdata.Algo, nil, 0, beta, dx.TD(), dx)
		if err != nil {

			panic(fmt.Errorf("Error: %v\nParams:\n,dx: %v\ndy: %v\nw: %v\nOps: %v", err, dx, dy, w, c.op))
		}
		return err
	}
	err := c.op.BackwardData(handle.Cudnn(), alpha, w.FD(), w, dy.TD(), dy, c.perfbackdata.Algo, wspace, wspace.SIB(), beta, dx.TD(), dx)
	if err != nil {
		fmt.Println("dx,dy,w dims", dx.Dims(), dy.Dims(), w.Dims())
		panic(err)
	}
	return err
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
	if wspace == nil {
		return c.op.BackwardFilter(handle.Cudnn(), alpha, x.TD(), x, dy.TD(), dy, c.perfbackfilt.Algo, nil, 0, beta, dw.FD(), dw)
	}
	return c.op.BackwardFilter(handle.Cudnn(), alpha, x.TD(), x, dy.TD(), dy, c.perfbackfilt.Algo, wspace, wspace.SIB(), beta, dw.FD(), dw)
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
		fmt.Println("5: ", x)
		fmt.Println("6: ", w.FD())
		fmt.Println("7: ", w)
		fmt.Println("8: ", c.desc)
		fmt.Println("9: ", c.fwdalgo)
		fmt.Println("10: ", wspace)
		fmt.Println("11: ", beta)
		fmt.Println("12: ", y.TD())
		fmt.Println("13: ", y)
	*/
	if wspace == nil {
		c.op.Forward(handle.Cudnn(), alpha, x.TD(), x, w.FD(), w, c.perfforward.Algo, nil, 0, beta, y.TD(), y)
	}
	return c.op.Forward(handle.Cudnn(), alpha, x.TD(), x, w.FD(), w, c.perfforward.Algo, wspace, wspace.SIB(), beta, y.TD(), y)
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
		dy,
		beta,
		dbias.TD(),
		dbias,
	)
}
