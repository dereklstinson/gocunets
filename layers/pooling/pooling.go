package pooling

import (
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn/pool"
	"github.com/dereklstinson/gocunets/layers"
	gocudnn "github.com/dereklstinson/gocudnn"
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

//GetOutputDims returns the output dims considering the input
func (l *Layer) GetOutputDims(input *layers.Tensor) ([]int32, error) {
	return l.pD.OutputDims(input.Volume)
}

//MakeOutputTensor will make the outputlayer for you
func (l *Layer) MakeOutputTensor(handle *cudnn.Handler, input *layers.Tensor) (*layers.Tensor, error) {
	frmt, dtype, _, err := input.Properties()
	if err != nil {

		return nil, err
	}
	dims, err := l.pD.OutputDims(input.Volume)
	if err != nil {
		return nil, err
	}

	return layers.CreateTensor(handle, frmt, dtype, dims)

}

//MakeOutputLayerInference makes the output inference IO which doesn't contain a volume for the deltas
func (l *Layer) MakeOutputLayerInference(handle *cudnn.Handler, input *layers.Tensor) (*layers.Tensor, error) {
	frmt, dtype, _, err := input.Properties()
	if err != nil {

		return nil, err
	}
	dims, err := l.pD.OutputDims(input.Volume)
	if err != nil {
		return nil, err
	}

	return layers.CreateTensor(handle, frmt, dtype, dims)

}

//SetupNoOutput will setup the pooling layer but not provide an output
func SetupNoOutput(mode gocudnn.PoolingMode, nan gocudnn.NANProp, window, padding, stride []int32) (*Layer, error) {
	pD, err := pool.StageOperation(mode, nan, window, padding, stride)

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

//SetupNoOutputReverse will setup the pooling layer but not provide an output
func SetupNoOutputReverse(mode gocudnn.PoolingMode, nan gocudnn.NANProp, window, padding, stride []int32) (*Layer, error) {
	pD, err := pool.StageOperationReverse(mode, nan, window, padding, stride)

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

//SetupReverse sets up a reverse pooling layer and returns a pointer to the struct. Scalars are set to the default alpha =1.0 and beta =0.0 for both fwd and bwd.
func SetupReverse(handle *cudnn.Handler, mode gocudnn.PoolingMode, nan gocudnn.NANProp, input *layers.Tensor, window, padding, stride []int32) (*Layer, *layers.Tensor, error) {
	pD, err := pool.StageOperationReverse(mode, nan, window, padding, stride)

	if err != nil {
		return nil, nil, err
	}
	fmt, dtype, _, err := input.Properties()
	if err != nil {
		return nil, nil, err
	}
	dims, err := pD.OutputDims(input.Volume)
	if err != nil {
		return nil, nil, err
	}
	output, err := layers.CreateTensor(handle, fmt, dtype, dims)
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

//Setup setsup the pooling layer and returns a pointer to the struct. Scalars are set to the default alpha =1.0 and beta =0.0 for both fwd and bwd.
func Setup(handle *cudnn.Handler, mode gocudnn.PoolingMode, nan gocudnn.NANProp, input *layers.Tensor, window, padding, stride []int32) (l *Layer, y, dy *layers.Tensor, err error) {
	pD, err := pool.StageOperation(mode, nan, window, padding, stride)

	if err != nil {
		return nil, nil, nil, err
	}
	fmt, dtype, _, err := input.Properties()
	if err != nil {
		return nil, nil, nil, err
	}
	dims, err := pD.OutputDims(input.Volume)
	if err != nil {
		return nil, nil, nil, err
	}
	y, err = layers.CreateTensor(handle, fmt, dtype, dims)
	if err != nil {
		return nil, nil, nil, err
	}
	dy, err = layers.CreateTensor(handle, fmt, dtype, dims)
	if err != nil {
		return nil, nil, nil, err
	}
	l = &Layer{
		pD: pD,
		fwd: xtras{
			alpha: 1.0,
			beta:  0.0,
		},
		bwd: xtras{
			alpha: 1.0,
			beta:  0.0,
		},
	}
	return l, y, dy, err
}

/*
//Destroy destroys the pooling descriptor
func (l *Layer) Destroy() error {
	return l.pD.Destroy()
}
*/
/*

//SetAlphaScalars takes a slice of float64 of length of 2 and sets the alphas in fwd and bwd order
func (l *Layer) SetAlphaScalars(alphas []float64) error {
	if len(alphas) != 2 {
		return errors.New("Length of alphas needs to be 2")
	}
	l.fwd.alpha = alphas[0]
	l.bwd.alpha = alphas[1]
	return nil
}

//SetBetaScalars takes a slice of float64 of length of 2 and sets the betas in fwd and bwd order
func (l *Layer) SetBetaScalars(betas []float64) error {
	if len(betas) != 2 {
		return errors.New("Length of betas needs to be 2")
	}
	l.fwd.beta = betas[0]
	l.bwd.beta = betas[1]
	return nil
}

//NumAlphaScalars returns the number of alpha scalars this layers has for the forward and backward prop
func (l *Layer) NumAlphaScalars() int {
	return 2
}

//NumBetaScalars returns the number of beta scalars this layers has for the forward and backward prop
func (l *Layer) NumBetaScalars() int {
	return 2
}
*/

//SetForwardScalars will change the default fwd scalars to whatever is passsed
func (l *Layer) SetForwardScalars(alpha, beta float64) {
	l.fwd.alpha = alpha
	l.fwd.beta = beta
}

//SetBackwardScalars will change the default bwd scalars to whatever is passed
func (l *Layer) SetBackwardScalars(alpha, beta float64) {
	l.bwd.alpha = alpha
	l.bwd.beta = beta
}

//ForwardProp performs the pooling forward propigation
func (l *Layer) ForwardProp(handle *cudnn.Handler, x, y *layers.Tensor) error {
	return l.pD.Forward(handle, l.fwd.alpha, l.fwd.beta, x.Volume, y.Volume)
}

//BackProp performs the pooling backward propigation
func (l *Layer) BackProp(handle *cudnn.Handler, x, dx, y, dy *layers.Tensor) error {
	return l.pD.Backward(handle, l.bwd.alpha, l.bwd.beta, x.Volume, dx.Volume, y.Volume, dy.Volume)

}
