package reduce

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

type flagsforop struct {
	DType      gocudnn.DataType
	NanProp    gocudnn.NANProp
	ReduceMode gocudnn.ReduceTensorOp
	IndFlag    gocudnn.ReduceTensorIndices
	IndType    gocudnn.IndiciesType
}

//Flags are the reduce op
var Flags flagsforop

//Ops contains the reduce ops information
type Ops struct {
	desc *gocudnn.ReduceTensorD
	mode gocudnn.ReduceTensorOp
}

//Stage stages the Reduce Operation
func Stage(reduceop gocudnn.ReduceTensorOp, dtype gocudnn.DataType, nanprop gocudnn.NANProp, reducetensorinds gocudnn.ReduceTensorIndices, indicietype gocudnn.IndiciesType) (*Ops, error) {
	desc, err := gocudnn.CreateReduceTensorDescriptor()
	if err != nil {
		return nil, err
	}
	err = desc.Set(reduceop, dtype, nanprop, reducetensorinds, indicietype)

	if err != nil {
		return nil, err
	}
	return &Ops{
		desc: desc,
		//	op:   op,
	}, nil
}

//Reduce performs the reduce operation with input/output being y where y= alpha* Op(x) +beta*y
func (o *Ops) Reduce(handle *cudnn.Handler, indicies *nvidia.Malloced, workspace *nvidia.Malloced, alpha float64, x *tensor.Volume, beta float64, y *tensor.Volume) error {
	if indicies == nil {
		return o.desc.ReduceTensorOp(handle.Cudnn(), nil, 0, workspace, workspace.SIB(), alpha, x.TD(), x, beta, y.TD(), y)
	}
	return o.desc.ReduceTensorOp(handle.Cudnn(), indicies, indicies.SIB(), workspace, workspace.SIB(), alpha, x.TD(), x, beta, y.TD(), y)

}

//GetWorkSpaceSize returns the workspace size for the two tensors
func (o *Ops) GetWorkSpaceSize(handle *cudnn.Handler, x, y *tensor.Volume) (uint, error) {
	return o.desc.GetWorkSpaceSize(handle.Cudnn(), x.TD(), y.TD())
}

//GetIndiciesSize returns the size of indicies
func (o *Ops) GetIndiciesSize(handle *cudnn.Handler, x, y *tensor.Volume) (uint, error) {
	return o.desc.GetIndiciesSize(handle.Cudnn(), x.TD(), y.TD())
}
