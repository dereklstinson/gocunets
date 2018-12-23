package convolution

import (
	"errors"
	"fmt"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Ops is a struct
type Ops struct {
	desc        *gocudnn.ConvolutionD
	helper      gocudnn.Convolution
	stagedalgo  bool
	fwdalgo     gocudnn.ConvFwdAlgo
	setfwd      bool
	bwddata     gocudnn.ConvBwdDataAlgo
	setbwd      bool
	bwdfilt     gocudnn.ConvBwdFiltAlgo
	setfilt     bool
	dims        int
	group       int
	pwspacesize gocudnn.SizeT
	pad         []int32
	dilation    []int32
	stride      []int32
}

//OpInfo is the contains the info to make the op
type OpInfo struct {
	CMode       gocudnn.ConvolutionMode `json:"ConvolutionMode"`
	Dtype       gocudnn.DataType        `json:"DataType"`
	Pad         []int32                 `json:"Pad"`
	Stride      []int32                 `json:"Stride"`
	Dilation    []int32                 `json:"Dilation"`
	StagedAlgos bool                    `json:"StagedAlgos"`

	FwdAlgo     gocudnn.ConvFwdAlgo     `json:"FwdAlgo"`
	BwdDataAlgo gocudnn.ConvBwdDataAlgo `json:"BwdDataAlgo"`
	BwdFiltAlgo gocudnn.ConvBwdFiltAlgo `json:"BwdFiltAlgo"`
	Group       int                     `json:"Group"`
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

//Stage stages/sets up an Ops and returns a pointer to it with the info stored in the info type
func (input OpInfo) Stage() (*Ops, error) {
	helper := gocudnn.Convolution{}
	if len(input.Pad) == 2 {
		desc, err := helper.NewConvolution2dDescriptor(input.CMode, input.Dtype, input.Pad, input.Stride, input.Dilation)
		if err != nil {
			return nil, err
		}
		return &Ops{
			desc:       desc,
			dims:       len(input.Pad),
			stagedalgo: input.StagedAlgos,
			fwdalgo:    input.FwdAlgo,
			bwddata:    input.BwdDataAlgo,
			bwdfilt:    input.BwdFiltAlgo,
			group:      input.Group,
			stride:     input.Stride,
			dilation:   input.Dilation,
			pad:        input.Pad,
		}, nil
	}
	desc, err := helper.NewConvolutionNdDescriptor(input.CMode, input.Dtype, input.Pad, input.Stride, input.Dilation)
	if err != nil {
		return nil, err
	}
	return &Ops{
		desc:       desc,
		dims:       len(input.Pad),
		stagedalgo: input.StagedAlgos,
		fwdalgo:    input.FwdAlgo,
		bwddata:    input.BwdDataAlgo,
		bwdfilt:    input.BwdFiltAlgo,
		group:      input.Group,
		stride:     input.Stride,
		dilation:   input.Dilation,
		pad:        input.Pad,
	}, nil
}

//Flags returns the flags that are used for convolution
func Flags() gocudnn.ConvolutionFlags {
	return gocudnn.ConvolutionFlags{}
}

//StageOperation set sets a convolution struct default algos go as follows fwd: direct, bwdfilt: algo0, bwddata:algo0
func StageOperation(mode gocudnn.ConvolutionMode, data cudnn.DataType, pad, stride, dilation []int32) (*Ops, error) {
	helper := gocudnn.Convolution{}
	if len(pad) == 2 {

		desc, err := helper.NewConvolution2dDescriptor(mode, data.Cu(), pad, stride, dilation)
		if err != nil {
			return nil, err
		}
		return &Ops{
			desc:     desc,
			dims:     len(pad),
			stride:   stride,
			dilation: dilation,
			pad:      pad,
		}, nil
	}
	desc, err := helper.NewConvolutionNdDescriptor(mode, data.Cu(), pad, stride, dilation)
	if err != nil {
		return nil, err
	}
	return &Ops{
		desc:     desc,
		dims:     len(pad),
		stride:   stride,
		dilation: dilation,
		pad:      pad,
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
func (c *Ops) WorkSizeFwd(handle *cudnn.Handler, x, w, y tensor.Volume) (gocudnn.SizeT, error) {
	return c.helper.Funcs.Fwd.GetConvolutionForwardWorkspaceSize(handle.Cudnn(), x.TD(), w.FD(), c.desc, y.TD(), c.fwdalgo)
}

//SetFwdAlgo sets fwd algo
func (c *Ops) SetFwdAlgo(algo gocudnn.ConvFwdAlgo) {
	c.stagedalgo = true
	c.fwdalgo = algo
}

//SetBwdDataAlgo sets the backward data algo
func (c *Ops) SetBwdDataAlgo(algo gocudnn.ConvBwdDataAlgo) {
	c.stagedalgo = true
	c.bwddata = algo
}

//SetBwdFiltAlgo sets the backward filters
func (c *Ops) SetBwdFiltAlgo(algo gocudnn.ConvBwdFiltAlgo) {
	c.stagedalgo = true
	c.bwdfilt = algo
}

//SetAlgos sets all the algos
func (c *Ops) SetAlgos(fwd gocudnn.ConvFwdAlgo, bwdd gocudnn.ConvBwdDataAlgo, bwdf gocudnn.ConvBwdFiltAlgo) {
	c.stagedalgo = true
	c.fwdalgo = fwd
	c.bwddata = bwdd
	c.bwdfilt = bwdf
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
	if c.stagedalgo == false && (wspace.ByteSize() != c.pwspacesize || c.setbwd == false) {

		c.setbwd = true
		c.pwspacesize = wspace.ByteSize()
		var pflg gocudnn.ConvolutionBwdFlags
		if wspace != nil {
			algo, err := c.helper.Funcs.Bwd.GetConvolutionBackwardDataAlgorithm(handle.Cudnn(), w.FD(), dy.TD(), c.desc, dx.TD(), pflg.DataPref.SpecifyWorkSpaceLimit(), wspace.ByteSize())
			if err != nil {
				_, _, wdims, _ := w.Properties()
				fmt.Println("weights:", wdims, "dy: ", dy.TD().Dims(), "dx: ", dx.TD().Dims())
				return err
			}
			return c.desc.BackwardData(handle.Cudnn(), a, w.FD(), w.Memer(), dy.TD(), dy.Memer(), algo, wspace, b, dx.TD(), dx.Memer())
		}
		algo, err := c.helper.Funcs.Bwd.GetConvolutionBackwardDataAlgorithm(handle.Cudnn(), w.FD(), dy.TD(), c.desc, dx.TD(), pflg.DataPref.NoWorkSpace(), 0)
		if err != nil {
			fmt.Println(w.FD().TensorD().Dims(), dy.TD().Dims(), dx.TD().Dims())
			return err
		}
		return c.desc.BackwardData(handle.Cudnn(), a, w.FD(), w.Memer(), dy.TD(), dy.Memer(), algo, wspace, b, dx.TD(), dx.Memer())

	}

	return c.desc.BackwardData(handle.Cudnn(), a, w.FD(), w.Memer(), dy.TD(), dy.Memer(), c.bwddata, wspace, b, dx.TD(), dx.Memer())
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
	if c.stagedalgo == false && (wspace.ByteSize() != c.pwspacesize || c.setfilt == false) {

		c.setfilt = true
		c.pwspacesize = wspace.ByteSize()
		var pflg gocudnn.ConvolutionBwdFlags
		if wspace != nil {
			algo, err := c.helper.Funcs.Bwd.GetConvolutionBackwardFilterAlgorithm(handle.Cudnn(), x.TD(), dy.TD(), c.desc, dw.FD(), pflg.FltrPref.SpecifyWorkSpaceLimit(), wspace.ByteSize())
			if err != nil {
				return err
			}
			return c.desc.BackwardFilter(handle.Cudnn(), a, x.TD(), x.Memer(), dy.TD(), dy.Memer(), algo, wspace, b, dw.FD(), dw.Memer())
		}
		algo, err := c.helper.Funcs.Bwd.GetConvolutionBackwardFilterAlgorithm(handle.Cudnn(), x.TD(), dy.TD(), c.desc, dw.FD(), pflg.FltrPref.NoWorkSpace(), 0)
		if err != nil {
			return err
		}
		return c.desc.BackwardFilter(handle.Cudnn(), a, x.TD(), x.Memer(), dy.TD(), dy.Memer(), algo, wspace, b, dw.FD(), dw.Memer())

	}

	return c.desc.BackwardFilter(handle.Cudnn(), a, x.TD(), x.Memer(), dy.TD(), dy.Memer(), c.bwdfilt, wspace, b, dw.FD(), dw.Memer())
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
	if c.stagedalgo == false && (wspace.ByteSize() != c.pwspacesize || c.setfwd == false) {

		c.setfwd = true
		c.pwspacesize = wspace.ByteSize()
		var pflg gocudnn.ConvolutionFwdFlags
		if wspace != nil {
			algo, err := c.helper.Funcs.Fwd.GetConvolutionForwardAlgorithm(handle.Cudnn(), x.TD(), w.FD(), c.desc, y.TD(), pflg.Pref.SpecifyWorkSpaceLimit(), wspace.ByteSize())
			if err != nil {
				return err
			}
			return c.desc.Forward(handle.Cudnn(), a, x.TD(), x.Memer(), w.FD(), w.Memer(), algo, wspace, b, y.TD(), y.Memer())
		}
		algo, err := c.helper.Funcs.Fwd.GetConvolutionForwardAlgorithm(handle.Cudnn(), x.TD(), w.FD(), c.desc, y.TD(), pflg.Pref.NoWorkSpace(), 0)
		if err != nil {
			return err
		}
		return c.desc.Forward(handle.Cudnn(), a, x.TD(), x.Memer(), w.FD(), w.Memer(), algo, wspace, b, y.TD(), y.Memer())

	}
	return c.desc.Forward(handle.Cudnn(), a, x.TD(), x.Memer(), w.FD(), w.Memer(), c.fwdalgo, wspace, b, y.TD(), y.Memer())
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
