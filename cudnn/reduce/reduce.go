package reduce

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

type flagsforop struct {
	DType      cudnn.DataTypeFlag
	NanProp    cudnn.NanModeFlag
	ReduceMode OpFlags
	IndFlag    IndiciesFLag
	IndType    IndTypeFlag
}

//Flags are the reduce op
var Flags flagsforop

//Ops contains the reduce ops information
type Ops struct {
	desc *gocudnn.ReduceTensorD
}

//Stage stages the Reduce Operation
func Stage(op OpMode, dtype cudnn.DataType, nanprop cudnn.NanMode, reducetensorinds IndiciesMode, indicietype TypeMode) (*Ops, error) {
	var red gocudnn.Reduce
	desc, err := red.NewReduceTensorDescriptor(op.cu(), dtype.Cu(), nanprop.Cu(), reducetensorinds.cu(), indicietype.cu())
	if err != nil {
		return nil, err
	}
	return &Ops{
		desc: desc,
	}, nil
}

//Reduce performs the reduce operation with input/output being y where y= alpha* Op(x) +beta*y
func (o *Ops) Reduce(handle *cudnn.Handler, indicies *gocudnn.Malloced, workspace *gocudnn.Malloced, alpha float64, x *tensor.Volume, beta float64, y *tensor.Volume) error {
	_, dtypet, _, err := x.Properties()
	if err != nil {
		return err
	}
	a := gocudnn.CScalarByDataType(dtypet.Cu(), alpha)
	c := gocudnn.CScalarByDataType(dtypet.Cu(), beta)
	if a == nil || c == nil {
		return errors.New("Not supported Format")
	}

	return o.desc.ReduceTensorOp(handle.Cudnn(), indicies, workspace, a, x.TD(), x.Memer(), c, y.TD(), y.Memer())

}

//GetWorkSpaceSize returns the workspace size for the two tensors
func (o *Ops) GetWorkSpaceSize(handle *cudnn.Handler, x, y *tensor.Volume) (gocudnn.SizeT, error) {
	return o.desc.GetWorkSpaceSize(handle.Cudnn(), x.TD(), y.TD())
}

//GetIndiciesSize returns the size of indicies
func (o *Ops) GetIndiciesSize(handle *cudnn.Handler, x, y *tensor.Volume) (gocudnn.SizeT, error) {
	return o.desc.IndiciesSize(handle.Cudnn(), x.TD(), y.TD())
}

//Destroy destroys the op and turns the op to nil
func (o *Ops) Destroy() error {
	err := o.desc.Destroy()
	if err == nil {
		o = nil
	}
	return err
}
