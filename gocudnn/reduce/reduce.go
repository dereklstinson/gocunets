package reduce

import (
	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Ops contains the reduce ops information
type Ops struct {
	desc *gocudnn.ReduceTensorD
}

//Stage stages the Reduce Operation
func Stage(op gocudnn.ReduceTensorOp, dtype gocudnn.DataType, nanprop gocudnn.PropagationNAN, reducetensorinds gocudnn.ReduceTensorIndices, indicietype gocudnn.IndiciesType) (*Ops, error) {
	var red gocudnn.Reduce
	desc, err := red.CreateReduceTensorDescriptor(op, dtype, nanprop, reducetensorinds, indicietype)
	if err != nil {
		return nil, err
	}
	return &Ops{
		desc: desc,
	}, nil
}

//GetWorkSpaceSize returns the workspace size for the two tensors
func (o *Ops) GetWorkSpaceSize(handle *gocudnn.Handle, x, y *tensor.Volume) (gocudnn.SizeT, error) {
	return o.desc.GetWorkSpaceSize(handle, x.TD(), y.TD())
}

//GetIndiciesSize returns the size of indicies
func (o *Ops) GetIndiciesSize(handle *gocudnn.Handle, x, y *tensor.Volume) (gocudnn.SizeT, error) {
	return o.desc.IndiciesSize(handle, x.TD(), y.TD())
}
