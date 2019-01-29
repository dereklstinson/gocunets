package pooling

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/Nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/Nvidia/cudnn/pool"
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Layer holds everything it needs on the pooling side in order
//to do the pooling operations.
type Layer struct {
	pD  *pool.Ops
	fwd xtras
	bwd xtras
}
type xtras struct {
	alpha float64
	beta  float64
}

//MakeOutputLayer will make the outputlayer for you
func (l *Layer) MakeOutputLayer(handle *cudnn.Handler, input *layers.IO) (*layers.IO, error) {
	frmt, dtype, _, err := input.Properties()
	if err != nil {

		return nil, err
	}
	dims, err := l.pD.OutputDims(input.T())
	if err != nil {
		return nil, err
	}

	return layers.BuildIO(handle, frmt, dtype, dims)

}

//SetupNoOutput will setup the pooling layer but not provide an output
func SetupNoOutput(mode gocudnn.PoolingMode, nan gocudnn.PropagationNAN, input *layers.IO, window, padding, stride []int32, managedmem bool) (*Layer, error) {
	pD, err := pool.StageOperation(mode, nan, input.T(), window, padding, stride)

	if err != nil {
		return nil, err
	}

	return &Layer{
		pD: pD,
		fwd: xtras{
			alpha: 1.0,
			beta:  0.0,
		},
		bwd: xtras{
			alpha: 1.0,
			beta:  0.0,
		},
	}, nil
}

//SetupDims will setup the pooling layer but not provide an output
func SetupDims(mode gocudnn.PoolingMode, nan gocudnn.PropagationNAN, numbofinputdims int, window, padding, stride []int32, managedmem bool) (*Layer, error) {
	pD, err := pool.StageOpDims(mode, nan, numbofinputdims, window, padding, stride)

	if err != nil {
		return nil, err
	}

	return &Layer{
		pD: pD,
		fwd: xtras{
			alpha: 1.0,
			beta:  0.0,
		},
		bwd: xtras{
			alpha: 1.0,
			beta:  0.0,
		},
	}, nil
}

//Setup setsup the pooling layer and returns a pointer to the struct. Scalars are set to the default alpha =1.0 and beta =0.0 for both fwd and bwd.
func Setup(handle *cudnn.Handler, mode gocudnn.PoolingMode, nan gocudnn.PropagationNAN, input *layers.IO, window, padding, stride []int32) (*Layer, *layers.IO, error) {
	pD, err := pool.StageOperation(mode, nan, input.T(), window, padding, stride)

	if err != nil {
		return nil, nil, err
	}
	fmt, dtype, _, err := input.Properties()
	if err != nil {
		return nil, nil, err
	}
	dims, err := pD.OutputDims(input.T())
	if err != nil {
		return nil, nil, err
	}
	output, err := layers.BuildIO(handle, fmt, dtype, dims)
	if err != nil {
		return nil, nil, err
	}
	return &Layer{
		pD: pD,
		fwd: xtras{
			alpha: 1.0,
			beta:  0.0,
		},
		bwd: xtras{
			alpha: 1.0,
			beta:  0.0,
		},
	}, output, nil
}

//Destroy destroys the pooling descriptor
func (l *Layer) Destroy() error {
	return l.pD.Destroy()
}

//SetFwdScalars will change the default fwd scalars to whatever is passsed
func (l *Layer) SetFwdScalars(alpha, beta float64) {
	l.fwd.alpha = alpha
	l.fwd.beta = beta
}

//SetBwdScalars will change the default bwd scalars to whatever is passed
func (l *Layer) SetBwdScalars(alpha, beta float64) {
	l.bwd.alpha = alpha
	l.bwd.beta = beta
}

//ForwardProp performs the pooling forward propigation
func (l *Layer) ForwardProp(handle *cudnn.Handler, x, y *layers.IO) error {
	return l.pD.FwdProp(handle, l.fwd.alpha, l.fwd.beta, x.T(), y.T())
}

//BackProp performs the pooling backward propigation
func (l *Layer) BackProp(handle *cudnn.Handler, x, y *layers.IO) error {
	return l.pD.BwdProp(handle, l.bwd.alpha, l.bwd.beta, x.T(), x.DeltaT(), y.T(), y.DeltaT())

}
