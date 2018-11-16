package dropout

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/tensor"
	"github.com/dereklstinson/GoCudnn"
)

//Ops does the operation.  It also holds the memory to run the tensor op
type Ops struct {
	dropout *gocudnn.DropOutD
	rss     gocudnn.SizeT
	sss     gocudnn.SizeT
	state   *gocudnn.Malloced
	reserve *gocudnn.Malloced
	seed    uint64
}

//BackProp does the back propagation for dropoutlayer
func (op *Ops) BackProp(handle *cudnn.Handler, dx, dy *tensor.Volume) error {
	return gocudnn.DropOut{}.Funcs.DropoutBackward(handle.Cudnn(), op.dropout, dy.TD(), dy.Memer(), dx.TD(), dx.Memer(), op.reserve)
}

//ForwardProp does the feed forward
func (op *Ops) ForwardProp(handle *cudnn.Handler, x, y *tensor.Volume) error {
	return gocudnn.DropOut{}.Funcs.DropoutForward(handle.Cudnn(), op.dropout, x.TD(), x.Memer(), y.TD(), y.Memer(), op.reserve)
}

//Destroy destroys the dropout layer
func (op *Ops) Destroy() error {

	var errf bool

	errorstring := "Dropout: "
	err := op.dropout.DestroyDescriptor()
	if err != nil {
		errf = true
		errorstring = errorstring + err.Error() + " , "
	}
	err = op.state.Free()
	if err != nil {
		errf = true
		errorstring = errorstring + err.Error() + " , "
	}
	err = op.reserve.Free()
	if err != nil {
		errf = true
		errorstring = errorstring + err.Error() + " , "
	}
	if errf == true {
		return errors.New(errorstring)
	}
	return nil
}

//Stage stages the op
func Stage(handle *cudnn.Handler, x *tensor.Volume, dropout float32, seed uint64, managed bool) (*Ops, error) {
	desc, err := gocudnn.DropOut{}.CreateDropoutDescriptor()
	if err != nil {
		return nil, err
	}
	rss, err := gocudnn.DropOut{}.Funcs.DropoutGetReserveSpaceSize(x.TD())
	if err != nil {
		return nil, err
	}
	sss, err := gocudnn.DropOut{}.Funcs.DropoutGetStateSize(handle.Cudnn())
	if err != nil {
		return nil, err
	}
	if managed == true {
		reserve, err := gocudnn.MallocManaged(rss, gocudnn.ManagedMemFlag{}.Global())
		if err != nil {
			return nil, err
		}
		state, err := gocudnn.MallocManaged(sss, gocudnn.ManagedMemFlag{}.Global())
		if err != nil {
			reserve.Free()
			return nil, err
		}
		err = desc.SetDropoutDescriptor(handle.Cudnn(), dropout, state, sss, seed)
		if err != nil {
			state.Free()
			reserve.Free()
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

	reserve, err := gocudnn.Malloc(rss)
	if err != nil {
		return nil, err
	}
	state, err := gocudnn.Malloc(sss)
	if err != nil {
		reserve.Free()
		return nil, err
	}
	err = desc.SetDropoutDescriptor(handle.Cudnn(), dropout, state, sss, seed)
	if err != nil {
		state.Free()
		reserve.Free()
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
