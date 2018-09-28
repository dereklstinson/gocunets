package batchnorm

import (
	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"
	"github.com/dereklstinson/GoCudnn"
)

type Ops struct {
	epsilon float64
	mode    gocudnn.BatchNormMode
	bnsbmvd *gocudnn.TensorD
	rrm     *gocudnn.Malloced
	rrv     *gocudnn.Malloced
	rsm     *gocudnn.Malloced
	rsv     *gocudnn.Malloced
	rbnscd  *gocudnn.Malloced
	rbnbd   *gocudnn.Malloced
}

func Stage(x tensor.Volume, mode gocudnn.BatchNormMode, managed bool) (*Ops, error) {

	bnd, err := gocudnn.BatchNorm{}.DeriveBNTensorDescriptor(x.TD(), mode)
	if err != nil {
		return nil, err
	}
	_, dtype, _, err := x.Properties()
	zero := gocudnn.CScalarByDataType(dtype, 0.0)
	zptr, err := gocudnn.MakeGoPointer(zero)
	if managed == true {

		rrm, err := gocudnn.MallocManaged(zptr.ByteSize(), gocudnn.ManagedMemFlag{}.Global())
		if err != nil {
			return nil, err
		}
		err = gocudnn.CudaMemCopy(rrm, zptr, zptr.ByteSize(), gocudnn.MemcpyKindFlag{}.Default())
		if err != nil {
			return nil, err
		}
		rrv, err := gocudnn.MallocManaged(zptr.ByteSize(), gocudnn.ManagedMemFlag{}.Global())
		if err != nil {
			return nil, err
		}
		err = gocudnn.CudaMemCopy(rrv, zptr, zptr.ByteSize(), gocudnn.MemcpyKindFlag{}.Default())
		if err != nil {
			return nil, err
		}
		rsm, err := gocudnn.MallocManaged(zptr.ByteSize(), gocudnn.ManagedMemFlag{}.Global())
		if err != nil {
			return nil, err
		}
		err = gocudnn.CudaMemCopy(rsm, zptr, zptr.ByteSize(), gocudnn.MemcpyKindFlag{}.Default())
		if err != nil {
			return nil, err
		}
		rsv, err := gocudnn.MallocManaged(zptr.ByteSize(), gocudnn.ManagedMemFlag{}.Global())
		if err != nil {
			return nil, err
		}
		err = gocudnn.CudaMemCopy(rsv, zptr, zptr.ByteSize(), gocudnn.MemcpyKindFlag{}.Default())
		if err != nil {
			return nil, err
		}
		rbnscd, err := gocudnn.MallocManaged(zptr.ByteSize(), gocudnn.ManagedMemFlag{}.Global())
		if err != nil {
			return nil, err
		}
		err = gocudnn.CudaMemCopy(rbnscd, zptr, zptr.ByteSize(), gocudnn.MemcpyKindFlag{}.Default())
		if err != nil {
			return nil, err
		}
		rbnbd, err := gocudnn.MallocManaged(zptr.ByteSize(), gocudnn.ManagedMemFlag{}.Global())
		if err != nil {
			return nil, err
		}
		err = gocudnn.CudaMemCopy(rbnbd, zptr, zptr.ByteSize(), gocudnn.MemcpyKindFlag{}.Default())
		if err != nil {
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
		}, nil

	}

	rrm, err := gocudnn.Malloc(zptr.ByteSize())
	if err != nil {
		return nil, err
	}
	err = gocudnn.CudaMemCopy(rrm, zptr, zptr.ByteSize(), gocudnn.MemcpyKindFlag{}.HostToDevice())
	if err != nil {
		return nil, err
	}
	rrv, err := gocudnn.Malloc(zptr.ByteSize())
	if err != nil {
		return nil, err
	}
	err = gocudnn.CudaMemCopy(rrv, zptr, zptr.ByteSize(), gocudnn.MemcpyKindFlag{}.HostToDevice())
	if err != nil {
		return nil, err
	}
	rsm, err := gocudnn.Malloc(zptr.ByteSize())
	if err != nil {
		return nil, err
	}
	err = gocudnn.CudaMemCopy(rsm, zptr, zptr.ByteSize(), gocudnn.MemcpyKindFlag{}.HostToDevice())
	if err != nil {
		return nil, err
	}
	rsv, err := gocudnn.Malloc(zptr.ByteSize())
	if err != nil {
		return nil, err
	}
	err = gocudnn.CudaMemCopy(rsv, zptr, zptr.ByteSize(), gocudnn.MemcpyKindFlag{}.HostToDevice())
	if err != nil {
		return nil, err
	}
	rbnscd, err := gocudnn.Malloc(zptr.ByteSize())
	if err != nil {
		return nil, err
	}
	err = gocudnn.CudaMemCopy(rbnscd, zptr, zptr.ByteSize(), gocudnn.MemcpyKindFlag{}.HostToDevice())
	if err != nil {
		return nil, err
	}
	rbnbd, err := gocudnn.Malloc(zptr.ByteSize())
	if err != nil {
		return nil, err
	}
	err = gocudnn.CudaMemCopy(rbnbd, zptr, zptr.ByteSize(), gocudnn.MemcpyKindFlag{}.HostToDevice())
	if err != nil {
		return nil, err
	}
	return &Ops{
		bnsbmvd: bnd,
		mode:    mode,
		rrm:     rrm,
		rrv:     rrv,
		rsm:     rsm,
		rsv:     rsv,
		rbnscd:  rbnscd,
		rbnbd:   rbnbd,
	}, nil

}

//ForwardTraining is used for the forward training
func (o *Ops) ForwardTraining(handle *gocudnn.Handle, alpha, beta, averagingfactor, epsilon float64, x, y *tensor.Volume, scale, bias gocudnn.Memer) error {
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
		scale,
		bias,
		averagingfactor,
		o.rrm,
		o.rrv,
		epsilon,
		o.rsm,
		o.rsv,
	)

}

//ForwardTraining is used for the forward training
func (o *Ops) BackwardProp(handle *gocudnn.Handle, alphaparam, betaparam, alphadata, betadata, averagingfactor, epsilon float64, x, dx, dy *tensor.Volume, scale, bias gocudnn.Memer) error {
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
		scale,
		o.rbnscd,
		o.rbnbd,
		epsilon,
		o.rsm,
		o.rsv,
	)

}
