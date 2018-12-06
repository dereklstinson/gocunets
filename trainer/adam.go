package trainer

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Adam is a struct that does the holds the params for adam optimization
type Adam struct {
	loss1     []float32
	loss2     []float32
	goptr1    *gocudnn.GoPointer
	goptr2    *gocudnn.GoPointer
	gpuloss1  *gocudnn.Malloced
	gpuloss2  *gocudnn.Malloced
	gsum      *gocudnn.Malloced
	xsum      *gocudnn.Malloced
	trainer   *gocudnn.TrainerD
	params    gocudnn.TrainingParams
	regparams gocudnn.RegParams
	dims      []int32
}

const defaultadambeta1 = 0.9
const defaultadambeta2 = 0.999
const defaultadameps = float32(1e-8)
const defaultadamrate = .001

//SetTrainingMem creates the training mem for the adam trainer
func (a *Adam) SetTrainingMem(han *cudnn.Handler, weights *layers.IO) error {
	a.freememer()
	_, dtype, dims, err := weights.Properties()
	if err != nil {
		return err
	}
	a.dims = dims
	DeFault := gocudnn.MemcpyKindFlag{}.Default()
	Global := gocudnn.ManagedMemFlag{}.Global()
	var dflg cudnn.DataTypeFlag
	switch dtype {

	case dflg.Float():
		a.loss1 = make([]float32, 1)
		a.loss2 = make([]float32, 1)
		a.goptr1, err = gocudnn.MakeGoPointer(a.loss1)
		if err != nil {
			return err
		}
		a.goptr2, err = gocudnn.MakeGoPointer(a.loss2)
		if err != nil {
			return err
		}
		asize := dimsize(dims)
		x := make([]float32, asize)
		sizet, err := gocudnn.FindSizeT(x)
		if err != nil {
			return err
		}
		xp, err := gocudnn.MakeGoPointer(x)
		if err != nil {
			return err
		}
		a.gsum, err = gocudnn.MallocManaged(sizet, Global)
		if err != nil {
			return err
		}
		a.xsum, err = gocudnn.MallocManaged(sizet, Global)
		if err != nil {
			return err
		}
		err = gocudnn.CudaMemCopy(a.gsum, xp, sizet, DeFault)
		if err != nil {
			return err
		}
		err = gocudnn.CudaMemCopy(a.xsum, xp, sizet, DeFault)
		if err != nil {
			return err
		}
		a.gpuloss1, err = gocudnn.MallocManaged(gocudnn.SizeT(4), Global)
		if err != nil {
			return err
		}
		a.gpuloss2, err = gocudnn.MallocManaged(gocudnn.SizeT(4), Global)
		if err != nil {
			return err
		}

	default:

		return errors.New("Only Float datatype supported at the moment")
	}
	return nil
}

func (a *Adam) freememer() error {
	memerrorstrings := "FreeingMem"
	flag := false
	if a.goptr1 != nil {
		memerrorstrings = memerrorstrings + a.goptr1.Free().Error() + " , "
		flag = true
	}
	if a.goptr2 != nil {
		memerrorstrings = memerrorstrings + a.goptr2.Free().Error() + " , "
		flag = true
	}
	if a.gpuloss1 != nil {
		memerrorstrings = memerrorstrings + a.gpuloss1.Free().Error() + " , "
		flag = true
	}
	if a.gpuloss2 != nil {
		memerrorstrings = memerrorstrings + a.gpuloss2.Free().Error() + " , "
		flag = true

	}
	if a.gsum != nil {
		memerrorstrings = memerrorstrings + a.gsum.Free().Error() + " , "
		flag = true
	}
	if a.xsum != nil {
		memerrorstrings = memerrorstrings + a.xsum.Free().Error() + " , "
		flag = true
	}
	if flag {
		return errors.New(memerrorstrings)
	}
	return nil
}

//Dims returns the dims of the training parameter holders
func (a *Adam) Dims() []int32 {
	return a.dims
}

//UpdateWeights updates the weights
func (a *Adam) UpdateWeights(handle *cudnn.Handler, weights *layers.IO, batchsize int) error {
	var err error
	err = handle.Sync()
	if err != nil {
		return err
	}
	a.SetBatch(float32(batchsize))
	err = a.trainer.L1L2Regularization(handle.XHandle(), weights.DeltaT().Memer(), weights.T().Memer(), a.gpuloss1, a.gpuloss2, a.regparams)
	if err != nil {
		return err
	}

	err = handle.Sync()
	if err != nil {
		return err
	}
	err = a.trainer.TrainValues(handle.XHandle(), weights.DeltaT().Memer(), weights.T().Memer(), a.gsum, a.xsum, a.params)
	if err != nil {
		return err
	}
	err = handle.Sync()
	if err != nil {
		return err
	}
	err = a.l1l2loss()
	if err != nil {
		return err
	}
	return handle.Sync()
}
func (a *Adam) l1l2loss() error {
	var err error
	err = gocudnn.UnifiedMemCopy(a.goptr1, a.gpuloss1)
	if err != nil {
		return err
	}
	err = gocudnn.UnifiedMemCopy(a.goptr2, a.gpuloss2)
	if err != nil {
		return err
	}
	return nil

}

//L1L2Loss returns the l1l2 loss of the memory that adam was training
func (a *Adam) L1L2Loss() (float32, float32) {

	return a.loss1[0], a.loss2[0]
}
func dimsize(dims []int32) int32 {
	x := int32(1)
	for i := 0; i < len(dims); i++ {
		x *= dims[i]
	}
	return x
}

//SetupAdamWandB returns a trainer for both WandB
func SetupAdamWandB(tctx *gocudnn.XHandle, decay1, decay2 float32, batch int32) (*Adam, *Adam, error) {
	adam1, err := SetupAdam(tctx, decay1, decay2, batch)
	if err != nil {
		return nil, nil, err
	}
	adam2, err := SetupAdam(tctx, decay1, decay2, batch)
	return adam1, adam2, err
}

//SetupAdam sets up adam
func SetupAdam(tctx *gocudnn.XHandle, decay1, decay2 float32, batch int32) (*Adam, error) {

	adam := gocudnn.TrainingModeFlag{}.Adam()
	t, err := gocudnn.Xtra{}.NewTrainingDescriptor(tctx, adam, gocudnn.DataTypeFlag{}.Float())
	if err != nil {
		return nil, err
	}
	reg := gocudnn.Xtra{}.CreateRegParamsFloat32(decay1, decay2, float32(batch))
	x := gocudnn.Xtra{}.CreateParamsFloat32(defaultadameps, defaultadamrate, defaultadambeta1, defaultadambeta2)

	return &Adam{
		trainer:   t,
		params:    x,
		regparams: reg,
	}, nil
}

//SetDecay1 sets decay1
func (a *Adam) SetDecay1(decay1 float32) {
	a.regparams.SetDecay1(decay1)
}

//SetDecay2 sets decay 2
func (a *Adam) SetDecay2(decay2 float32) {
	a.regparams.SetDecay2(decay2)

}

//SetBeta1 sets beta1
func (a *Adam) SetBeta1(beta1 float32) {
	a.params.SetBeta1(beta1)
}

//SetBeta2 sets beta2
func (a *Adam) SetBeta2(beta2 float32) {
	a.params.SetBeta2(beta2)

}

//SetRate sets rate
func (a *Adam) SetRate(rate float32) {
	a.params.SetRate(rate)

}

//SetBatch sets batch
func (a *Adam) SetBatch(batch float32) {
	a.regparams.SetBatch(batch)
}

//SetEps sets eps
func (a *Adam) SetEps(eps float32) {
	a.params.SetEps(eps)

}

/*
func (a *Adam)Loss1()float32{
	return a.loss1
}

*/
/*
//CreateAdamHandle creates a handle for adam
func CreateAdamHandle(dev *gocudnn.Device, kerneldir string) (*gocudnn.XHandle, error) {
	var x gocudnn.Xtra
	return x.MakeXHandle(kerneldir, dev)

}
*/
