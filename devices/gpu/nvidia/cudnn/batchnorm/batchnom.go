package batchnorm

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Ops contains the operation of batchnorm.
type Ops struct {
	epsilon           float64
	exponentialfactor uint
	op                *gocudnn.BatchNormD
	opex              *gocudnn.BatchNormDEx
	bnsbmvd           *gocudnn.TensorD
	rrm               *nvidia.Malloced
	rrv               *nvidia.Malloced
	rsm               *nvidia.Malloced
	rsv               *nvidia.Malloced
}

func buildfromdesc(handle *cudnn.Handler, desc *gocudnn.TensorD) (*nvidia.Malloced, error) {

	sizet, err := desc.GetSizeInBytes()
	if err != nil {
		return nil, err
	}

	gpumem, err := nvidia.MallocGlobal(handle, sizet)
	if err != nil {
		return nil, err
	}

	err = gocudnn.SetTensor(handle.Cudnn(), desc, gpumem, 0.0)
	if err != nil {

		return nil, err
	}
	return gpumem, err

}
func errorstacker(original, newerr error) error {
	x := newerr.Error()
	y := original.Error()
	return errors.New(x + "..." + y)
}

/*
//Free Frees the mem
func (o *Ops) Free() error {
	var err error
	var errstack error

	err = o.rrm.Free()
	if err != nil {
		errstack = errorstacker(errstack, err)
	}
	err = o.rrv.Free()
	if err != nil {
		errstack = errorstacker(errstack, err)
	}
	err = o.rsm.Free()
	if err != nil {
		errstack = errorstacker(errstack, err)
	}
	err = o.rsv.Free()
	if err != nil {
		errstack = errorstacker(errstack, err)
	}
	err = o.bnsbmvd.DestroyDescriptor()
	if err != nil {
		errstack = errorstacker(errstack, err)
	}
	return errstack
}
*/

//PreStageSpatial Normalization is performed over N+spatial dimensions.
//This mode is intended for use after convolutional layers (where spatial invariance is desired).
//In this mode the bnBias, bnScale tensor dimensions are 1xCx1x1.
func PreStageSpatial(handle *cudnn.Handler) (*Ops, error) {
	var x gocudnn.BatchNormMode
	op := gocudnn.CreateBatchNormDescriptor()
	op.Set(x.Spatial())
	return &Ops{
		op: op,
	}, nil
}

/*PreStageSpatialPersistant - similar to spatial but can be faster

An optimized path may be selected for CUDNN_DATA_FLOAT and CUDNN_DATA_HALF types, compute capability 6.0 or higher for the following two batch normalization API calls: cudnnBatchNormalizationForwardTraining(), and cudnnBatchNormalizationBackward().
In the case of cudnnBatchNormalizationBackward(), the savedMean and savedInvVariance arguments should not be NULL.

The rest of this section applies for NCHW mode only:

This mode may use a scaled atomic integer reduction that is deterministic but imposes more restrictions on the input data range.
When a numerical overflow occurs the algorithm may produce NaN-s or Inf-s (infinity) in output buffers.

When Inf-s/NaN-s are present in the input data, the output in this mode is the same as from a pure floating-point implementation.

For finite but very large input values, the algorithm may encounter overflows more frequently due to a lower dynamic range and emit Inf-s/NaN-s while CUDNN_BATCHNORM_SPATIAL will produce finite results.
The user can invoke cudnnQueryRuntimeError() to check if a numerical overflow occurred in this mode.
*/
func PreStageSpatialPersistant(handle *cudnn.Handler) (*Ops, error) {
	var x gocudnn.BatchNormMode
	op := gocudnn.CreateBatchNormDescriptor()
	op.Set(x.SpatialPersistent())
	return &Ops{
		op: op,
	}, nil

}

//PreStagePerActivation Normalization is performed per-activation. This mode is intended to be used after non-convolutional network layers.
//In this mode the tensor dimensions of bnBias and bnScale, the parameters used in the cudnnBatchNormalization* functions, are 1xCxHxW.
func PreStagePerActivation(handle *cudnn.Handler, managed bool) (*Ops, error) {
	var x gocudnn.BatchNormMode
	op := gocudnn.CreateBatchNormDescriptor()
	op.Set(x.PerActivation())
	return &Ops{
		op: op,
	}, nil

}

//Stage will stage the o Ops from the prestaged function
func (o *Ops) Stage(handle *cudnn.Handler, x *tensor.Volume) error {

	bnd, err := o.op.DeriveBNTensorDescriptor(x.TD())
	o.bnsbmvd = bnd
	if err != nil {
		return err
	}
	o.rrm, err = buildfromdesc(handle, bnd)
	if err != nil {

		return err
	}
	o.rrv, err = buildfromdesc(handle, bnd)
	if err != nil {

		return err
	}
	o.rsm, err = buildfromdesc(handle, bnd)
	if err != nil {

		return err
	}
	o.rsv, err = buildfromdesc(handle, bnd)
	if err != nil {

		return err
	}
	return nil
}

//BiasScaleProperties returns the bias and scale for the batch norm bias and scale forward and backprop
func (o *Ops) BiasScaleProperties() (gocudnn.TensorFormat, gocudnn.DataType, []int32) {
	return (o.bnsbmvd.Format()), (o.bnsbmvd.DataType()), o.bnsbmvd.Dims()
}

//Stage stages the bachnorm op. It also builds the memory for it so you don't have to worry about it.
func Stage(handle *cudnn.Handler,
	x *tensor.Volume,
	mode gocudnn.BatchNormMode) (*Ops, error) {
	op := gocudnn.CreateBatchNormDescriptor()
	err := op.Set(mode)
	if err != nil {
		return nil, err
	}
	bnd, err := op.DeriveBNTensorDescriptor(x.TD())
	if err != nil {
		return nil, err
	}
	rrm, err := buildfromdesc(handle, bnd)
	if err != nil {

		return nil, err
	}
	rrv, err := buildfromdesc(handle, bnd)
	if err != nil {

		return nil, err
	}
	rsm, err := buildfromdesc(handle, bnd)
	if err != nil {

		return nil, err
	}
	rsv, err := buildfromdesc(handle, bnd)
	if err != nil {

		return nil, err
	}

	return &Ops{
		bnsbmvd: bnd,
		op:      op,
		rrm:     rrm,
		rrv:     rrv,
		rsm:     rsm,
		rsv:     rsv,
	}, nil

}

//ForwardTraining is used for the forward training
func (o *Ops) ForwardTraining(handle *cudnn.Handler,
	alpha,
	beta,
	averagingfactor,
	epsilon float64,
	x,
	scale *tensor.Volume,
	bias *tensor.Volume,
	y *tensor.Volume) error {

	return o.op.ForwardTraining(
		handle.Cudnn(),
		alpha,
		beta,
		x.TD(),
		x.Memer(),
		y.TD(),
		y.Memer(),
		o.bnsbmvd,
		scale.Memer(),
		bias.Memer(),
		averagingfactor,
		o.rrm,
		o.rrv,
		epsilon,
		o.rsm,
		o.rsv,
	)

}

//ForwardInference is the forward prop used for testing and production
func (o *Ops) ForwardInference(handle *cudnn.Handler,
	alpha, beta, epsilon float64,
	x, scale, bias, y *tensor.Volume) error {
	return o.op.ForwardInference(
		handle.Cudnn(),
		alpha,
		beta,
		x.TD(),
		x.Memer(),
		y.TD(),
		y.Memer(),
		o.bnsbmvd,
		scale.Memer(),
		bias.Memer(),
		o.rrm,
		o.rrv,
		epsilon,
	)
}

//BackwardProp is used for the forward training
func (o *Ops) BackwardProp(handle *cudnn.Handler,
	alphadata,
	betadata,
	alphaparam,
	betaparam,
	epsilon float64,
	x,
	scale,
	dscale,
	dbias,
	dx,
	dy *tensor.Volume) error {

	return o.op.Backward(
		handle.Cudnn(),
		alphadata,
		betadata,
		alphaparam,
		betaparam,

		x.TD(),
		x.Memer(),
		dy.TD(),
		dy.Memer(),
		dx.TD(),
		dx.Memer(),
		o.bnsbmvd,
		scale.Memer(),
		dscale.Memer(),
		dbias.Memer(),
		epsilon,
		o.rsm,
		o.rsv,
	)

}
