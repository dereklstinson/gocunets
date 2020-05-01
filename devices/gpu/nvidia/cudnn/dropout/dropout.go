package dropout

import (
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn/tensor"
	gocudnn "github.com/dereklstinson/gocudnn"
)

//Ops does the operation.  It also holds the memory to run the tensor op
type Ops struct {
	dropout *gocudnn.DropOutD
	rss     uint
	sss     uint
	state   *nvidia.Malloced
	reserve *nvidia.Malloced
	seed    uint64
}

//BackProp does the back propagation for dropoutlayer
func (op *Ops) BackProp(handle *cudnn.Handler, dx, dy *tensor.Volume) error {

	return op.dropout.Backward(handle.Cudnn(), dy.TD(), dy, dx.TD(), dx, op.reserve, op.reserve.SIB())
}

//ForwardProp does the feed forward
func (op *Ops) ForwardProp(handle *cudnn.Handler, x, y *tensor.Volume) error {

	return op.dropout.Forward(handle.Cudnn(), x.TD(), x, y.TD(), y, op.reserve, op.reserve.SIB())
}

//Stage stages the op
func Stage(handle *cudnn.Handler, x *tensor.Volume, dropout float32, seed uint64) (*Ops, error) {
	desc, err := gocudnn.CreateDropOutDescriptor()

	if err != nil {
		return nil, err
	}
	rss, err := desc.GetReserveSpaceSize(x.TD())
	if err != nil {
		return nil, err
	}
	sss, err := desc.GetStateSize(handle.Cudnn())
	if err != nil {
		return nil, err
	}

	reserve, err := nvidia.MallocGlobal(handle, rss)
	if err != nil {
		return nil, err
	}
	state, err := nvidia.MallocGlobal(handle, sss)
	if err != nil {
		return nil, err
	}
	desc.Set(handle.Cudnn(), dropout, state, sss, seed)
	if err != nil {
		return nil, err
	}
	return &Ops{
		dropout: desc,
		rss:     rss,
		sss:     sss,
		state:   state,
		reserve: reserve,
		seed:    seed,
	}, nil
}
