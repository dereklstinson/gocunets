package reduce

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

type flagsforop struct {
	DType      cudnn.DataType
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
	op   OpMode
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
		op:   op,
	}, nil
}

//Reduce performs the reduce operation with input/output being y where y= alpha* Op(x) +beta*y
func (o *Ops) Reduce(handle *cudnn.Handler, indicies *nvidia.Malloced, workspace *nvidia.Malloced, alpha float64, x *tensor.Volume, beta float64, y *tensor.Volume) error {

	err := o.desc.ReduceTensorOp(handle.Cudnn(), indicies, indicies.TotalBytes(), workspace, workspace.TotalBytes(), alpha, x.TD(), x.Memer(), beta, y.TD(), y.Memer())
	if err != nil {
		return errors.New(o.op.Readable() + ":" + err.Error())
	}
	return nil

}

//GetWorkSpaceSize returns the workspace size for the two tensors
func (o *Ops) GetWorkSpaceSize(handle *cudnn.Handler, x, y *tensor.Volume) (uint, error) {
	return o.desc.GetWorkSpaceSize(handle.Cudnn(), x.TD(), y.TD())
}

//GetIndiciesSize returns the size of indicies
func (o *Ops) GetIndiciesSize(handle *cudnn.Handler, x, y *tensor.Volume) (uint, error) {
	return o.desc.IndiciesSize(handle.Cudnn(), x.TD(), y.TD())
}
