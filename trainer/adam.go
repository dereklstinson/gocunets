package trainer

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

type Adam struct {
	loss1    float32
	loss2    float32
	gpuloss1 gocudnn.Memer
	gpuloss2 gocudnn.Memer
	gsum     gocudnn.Memer
	xsum     gocudnn.Memer
	trainer  *gocudnn.TrainerD
	params   gocudnn.TrainingParams
}

const defaultadambeta1 = 0.9
const defaultadambeta2 = 0.999
const defaultadameps = float32(1e-8)
const defaultadamrate = .001

func (a *Adam) SetTrainingMem(ctx gocudnn.Contexter, weights *layers.IO) error {
	_, err := ctx.GetTContext()
	if err != nil {
		return err
	}
	_, dtype, dims, err := weights.Properties()
	DeFault := gocudnn.MemcpyKindFlag{}.Default()
	Global := gocudnn.ManagedMemFlag{}.Global()
	if err != nil {
		return err
	}

	switch dtype {
	case gocudnn.DataTypeFlag{}.Float():
		//err = ctx.Push()
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
	//_, err = gocudnn.Cuda{}.CtxPopCurrent()
	default:
		return errors.New("Only Float datatype supported at the moment")
	}
	return errors.New("Shouldn't have reached this spot")

}
func (a *Adam) UpdateWeights(ctx gocudnn.Contexter, weights *layers.IO) error {
	tctx, err := ctx.GetTContext()
	if err != nil {
		return err
	}
	return a.trainer.TrainValues(tctx, 32, weights.DeltaT().Memer(), weights.T().Memer(), a.gpuloss1, a.gpuloss2, a.gsum, a.xsum, a.params)
}
func (a *Adam) L1L2Loss() (float32, float32) {
	return a.loss1, a.loss2
}
func dimsize(dims []int32) int32 {
	x := int32(1)
	for i := 0; i < len(dims); i++ {
		x *= dims[i]
	}
	return x
}
func SetupAdam(ctx gocudnn.Contexter, decay1, decay2 float32, batch int) (*Adam, error) {
	tctx, err := ctx.GetTContext()
	if err != nil {
		return nil, err
	}
	l1l2 := gocudnn.RegularizationFlag{}.L1L2()
	adam := gocudnn.TrainingModeFlag{}.Adam()
	t, err := gocudnn.Xtra{}.NewTrainingDescriptor(tctx, adam, gocudnn.DataTypeFlag{}.Float(), l1l2)
	if err != nil {
		return nil, err
	}
	x := gocudnn.CreateParamsFloat32(decay1, decay2, float32(batch), defaultadameps, defaultadamrate, defaultadambeta1, defaultadambeta2)

	return &Adam{
		trainer: t,
		params:  x,
	}, nil
}

//SetDecay1 sets decay1
func (a *Adam) SetDecay1(decay1 float32) {
	a.params.SetDecay1(decay1)
}

//SetDecay2 sets decay 2
func (a *Adam) SetDecay2(decay2 float32) {
	a.params.SetDecay2(decay2)

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
func (a *Adam) SetBatch(batch float32) {
	a.params.SetBatch(batch)
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

//CreateAdamContext creates a context for adam
func CreateAdamContext(flags uint32, dev *gocudnn.Device, kerneldir string) (*gocudnn.TContext, error) {
	var x gocudnn.Xtra

	return x.MakeTrainingContext(flags, dev, kerneldir)
}
