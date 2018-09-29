package batchnorm

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"
	"github.com/dereklstinson/GoCudnn"
)

//Ops contains the operation of batchnorm.
type Ops struct {
	epsilon float64
	mode    gocudnn.BatchNormMode
	bnsbmvd *gocudnn.TensorD
	scale   *gocudnn.Malloced
	bias    *gocudnn.Malloced
	rrm     *gocudnn.Malloced
	rrv     *gocudnn.Malloced
	rsm     *gocudnn.Malloced
	rsv     *gocudnn.Malloced
	rbnscd  *gocudnn.Malloced
	rbnbd   *gocudnn.Malloced
}

func buildfromdesc(handle *gocudnn.Handle, desc *gocudnn.TensorD, managed bool) (*gocudnn.Malloced, error) {
	dtype, _, _, err := desc.GetDescrptor()
	if err != nil {
		return nil, err
	}
	sizet, err := desc.GetSizeInBytes()
	if err != nil {
		return nil, err
	}
	if managed == true {
		gpumem, err := gocudnn.MallocManaged(sizet, gocudnn.ManagedMemFlag{}.Global())
		if err != nil {
			return nil, err
		}
		zero := gocudnn.CScalarByDataType(dtype, 0.0)
		err = gocudnn.TensorFuncs{}.SetTensor(handle, desc, gpumem, zero)
		if err != nil {
			gpumem.Free()
			return nil, err
		}
		return gpumem, err

	}
	gpumem, err := gocudnn.Malloc(sizet)
	if err != nil {
		return nil, err
	}
	zero := gocudnn.CScalarByDataType(dtype, 0.0)
	err = gocudnn.TensorFuncs{}.SetTensor(handle, desc, gpumem, zero)
	if err != nil {
		gpumem.Free()
		return nil, err
	}
	return gpumem, err

}
func errorstacker(original, newerr error) error {
	x := newerr.Error()
	y := original.Error()
	return errors.New(x + "..." + y)
}

//Free Frees the mem
func (o *Ops) Free() error {
	var err error
	var errstack error
	err = o.rbnbd.Free()
	if err != nil {
		errstack = errorstacker(errstack, err)
	}
	err = o.rbnscd.Free()
	if err != nil {
		errstack = errorstacker(errstack, err)
	}
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

//Stage stages the bachnorm op. It also builds the memory for it so you don't have to worry about it.
func Stage(handle *gocudnn.Handle, x *tensor.Volume, mode gocudnn.BatchNormMode, managed bool) (*Ops, error) {

	bnd, err := gocudnn.BatchNorm{}.DeriveBNTensorDescriptor(x.TD(), mode)
	if err != nil {
		return nil, err
	}

	rrm, err := buildfromdesc(handle, bnd, managed)
	if err != nil {
		return nil, err
	}
	rrv, err := buildfromdesc(handle, bnd, managed)
	if err != nil {
		rrm.Free()
		return nil, err
	}
	rsm, err := buildfromdesc(handle, bnd, managed)
	if err != nil {
		rrv.Free()
		rrm.Free()
		return nil, err
	}
	rsv, err := buildfromdesc(handle, bnd, managed)
	if err != nil {
		rsm.Free()
		rrv.Free()
		rrm.Free()
		return nil, err
	}
	rbnscd, err := buildfromdesc(handle, bnd, managed)
	if err != nil {
		rsm.Free()
		rsv.Free()
		rrv.Free()
		rrm.Free()
		return nil, err
	}
	rbnbd, err := buildfromdesc(handle, bnd, managed)
	if err != nil {
		rbnscd.Free()
		rsm.Free()
		rsv.Free()
		rrv.Free()
		rrm.Free()
		return nil, err
	}
	bias, err := buildfromdesc(handle, bnd, managed)
	if err != nil {
		rbnscd.Free()
		rsm.Free()
		rsv.Free()
		rrv.Free()
		rrm.Free()
		rbnbd.Free()
		return nil, err
	}
	scale, err := buildfromdesc(handle, bnd, managed)
	if err != nil {
		rbnscd.Free()
		rsm.Free()
		rsv.Free()
		rrv.Free()
		rrm.Free()
		rbnbd.Free()
		bias.Free()
		return nil, err
	}

	return &Ops{
		bnsbmvd: bnd,
		mode:    mode,
		rrm:     rrm,
		rrv:     rrv,
		rsm:     rsm,
		rsv:     rsv,
		rbnbd:   rbnbd,
		rbnscd:  rbnscd,
		scale:   scale,
		bias:    bias,
	}, nil

}

//ForwardTraining is used for the forward training
func (o *Ops) ForwardTraining(handle *gocudnn.Handle, alpha, beta, averagingfactor, epsilon float64, x, y *tensor.Volume) error {
	_, dtype, _, err := x.Properties()
	if err != nil {
		return err
	}
	a := gocudnn.CScalarByDataType(dtype, alpha)
	b := gocudnn.CScalarByDataType(dtype, beta)
	return gocudnn.BatchNorm{}.Funcs.BatchNormalizationForwardTraining(handle,
		o.mode,
		a,
		b,
		x.TD(),
		x.Memer(),
		y.TD(),
		y.Memer(),
		o.bnsbmvd,
		o.scale,
		o.bias,
		averagingfactor,
		o.rrm,
		o.rrv,
		epsilon,
		o.rsm,
		o.rsv,
	)

}

//BackwardProp is used for the forward training
func (o *Ops) BackwardProp(handle *gocudnn.Handle, alphaparam, betaparam, alphadata, betadata, epsilon float64, x, dx, dy *tensor.Volume) error {
	_, dtype, _, err := x.Properties()
	if err != nil {
		return err
	}
	ad := gocudnn.CScalarByDataType(dtype, alphadata)
	bd := gocudnn.CScalarByDataType(dtype, betadata)
	ap := gocudnn.CScalarByDataType(dtype, alphaparam)
	bp := gocudnn.CScalarByDataType(dtype, betaparam)
	return gocudnn.BatchNorm{}.Funcs.BatchNormalizationBackward(handle,
		o.mode,
		ad,
		bd,
		ap,
		bp,
		x.TD(),
		x.Memer(),
		dx.TD(),
		dx.Memer(),
		dy.TD(),
		dy.Memer(),
		o.bnsbmvd,
		o.scale,
		o.rbnscd,
		o.rbnbd,
		epsilon,
		o.rsm,
		o.rsv,
	)

}
