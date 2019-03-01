package dropout

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
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

	return op.dropout.DropoutBackward(handle.Cudnn(), dy.TD(), dy.Memer(), dx.TD(), dx.Memer(), op.reserve, op.reserve.TotalBytes())
}

//ForwardProp does the feed forward
func (op *Ops) ForwardProp(handle *cudnn.Handler, x, y *tensor.Volume) error {

	return op.dropout.DropoutForward(handle.Cudnn(), x.TD(), x.Memer(), y.TD(), y.Memer(), op.reserve, op.reserve.TotalBytes())
}

//Stage stages the op
func Stage(handle *cudnn.Handler, x *tensor.Volume, dropout float32, seed uint64) (*Ops, error) {
	if x == nil {
		return nil, errors.New("x can't be nil")
	}
	rss, err := gocudnn.DropOut{}.Funcs.DropoutGetReserveSpaceSize(x.TD())
	if err != nil {
		return nil, err
	}
	sss, err := gocudnn.DropOut{}.Funcs.DropoutGetStateSize(handle.Cudnn())
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
	desc, err := gocudnn.DropOut{}.NewDropoutDescriptor(handle.Cudnn(), dropout, state, sss, seed)
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
