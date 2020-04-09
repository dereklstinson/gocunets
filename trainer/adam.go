package trainer

import (
	"errors"
	"fmt"

	"github.com/dereklstinson/half"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/GoCudnn/xtra"
)

var debuggingadam bool

//DebuggingAdam is for debugging purposes
func DebuggingAdam() {
	debuggingadam = true
}

//Adam is a struct that does the holds the params for adam optimization
type Adam struct {
	dtype     gocudnn.DataType
	loss1h    []half.Float16
	loss2h    []half.Float16
	loss1     []float32
	loss2     []float32
	goptr1    *gocu.Wrapper
	goptr2    *gocu.Wrapper
	gpuloss1  *nvidia.Malloced
	gpuloss2  *nvidia.Malloced
	gsum      *nvidia.Malloced
	xsum      *nvidia.Malloced
	trainer   *xtra.TrainerD
	params    xtra.TrainingParams
	regparams xtra.RegParams
	dims      []int32
}

const defaultadambeta1 = 0.9
const defaultadambeta2 = 0.999
const defaultadameps = float32(1e-8)
const defaultadamrate = .001

//SetTrainingMem creates the training mem for the adam trainer
func (a *Adam) SetTrainingMem(han *cudnn.Handler, w *layers.Tensor) error {
	// /a.freememer()
	_, dtype, dims, err := w.Properties()
	if err != nil {
		if debuggingadam {
			panic(err)
		}
		return err
	}
	a.dims = dims
	//DeFault := gocudnn.MemcpyKindFlag{}.Default()

	var dflg gocudnn.DataType
	switch dtype {

	case dflg.Float():
		a.dtype.Float()
		a.loss1 = make([]float32, 1)
		a.loss2 = make([]float32, 1)
		a.goptr1, err = gocu.MakeGoMem(a.loss1)
		if err != nil {
			if debuggingadam {
				panic(err)
			}
			return err
		}
		a.goptr2, err = gocu.MakeGoMem(a.loss2)
		if err != nil {
			if debuggingadam {
				panic(err)
			}
			return err
		}
		//asize := dimsize()
		sizet := gocudnn.FindSizeTfromVol(dims, dtype)

		a.gsum, err = nvidia.MallocGlobal(han, sizet)
		if err != nil {
			if debuggingadam {
				panic(err)
			}
			return err
		}
		a.xsum, err = nvidia.MallocGlobal(han, sizet)
		if err != nil {
			if debuggingadam {
				panic(err)
			}
			return err
		}
		err = a.xsum.SetAll(0)
		if err != nil {
			if debuggingadam {
				fmt.Println("Dims are", dims)
				fmt.Println("Adress for a.xsum,and a.gsum", a.xsum, a.gsum)
				fmt.Println("a.xsum Cudasize", a.gsum.SIB())
				panic(err)
			}
		}
		err = a.gsum.SetAll(0)
		if err != nil {
			if debuggingadam {
				fmt.Println("a.gsum Cudasize", a.gsum.SIB())
				panic(err)
			}
		}

		a.gpuloss1, err = nvidia.MallocGlobal(han, 4)
		if err != nil {
			if debuggingadam {
				panic(err)
			}
			return err
		}
		a.gpuloss2, err = nvidia.MallocGlobal(han, 4)
		if err != nil {
			if debuggingadam {
				panic(err)
			}
			return err
		}
	case dflg.Half():

		a.dtype.Half()
		a.loss1h = make([]half.Float16, 1)
		a.loss2h = make([]half.Float16, 1)
		a.goptr1, err = gocu.MakeGoMem(a.loss1h)
		if err != nil {
			if debuggingadam {
				panic(err)
			}
			return err
		}
		a.goptr2, err = gocu.MakeGoMem(a.loss2h)
		if err != nil {
			if debuggingadam {
				panic(err)
			}
			return err
		}
		//asize := dimsize()
		sizet := gocudnn.FindSizeTfromVol(dims, dtype)

		a.gsum, err = nvidia.MallocGlobal(han, sizet)
		if err != nil {
			if debuggingadam {
				panic(err)
			}
			return err
		}
		a.xsum, err = nvidia.MallocGlobal(han, sizet)
		if err != nil {
			if debuggingadam {
				panic(err)
			}
			return err
		}
		err = a.xsum.SetAll(0)
		if err != nil {
			if debuggingadam {
				fmt.Println("Dims are", dims)
				fmt.Println("Adress for a.xsum,and a.gsum", a.xsum, a.gsum)
				fmt.Println("a.xsum Cudasize", a.gsum.SIB())
				panic(err)
			}
		}
		err = a.gsum.SetAll(0)
		if err != nil {
			if debuggingadam {
				fmt.Println("a.gsum Cudasize", a.gsum.SIB())
				panic(err)
			}
		}

		a.gpuloss1, err = nvidia.MallocGlobal(han, 2)
		if err != nil {
			if debuggingadam {
				panic(err)
			}
			return err
		}
		a.gpuloss2, err = nvidia.MallocGlobal(han, 2)
		if err != nil {
			if debuggingadam {
				panic(err)
			}
			return err
		}

	default:

		return errors.New("Only Float datatype supported at the moment")
	}
	return nil
}

//Dims returns the dims of the training parameter holders
func (a *Adam) Dims() []int32 {
	return a.dims
}

//UpdateWeights updates the weights
func (a *Adam) UpdateWeights(handle *cudnn.Handler, dw, w *layers.Tensor, batchsize, counter int) error {
	var err error
	err = handle.Sync()
	if err != nil {
		return err
	}
	flg := a.dtype
	switch a.dtype {
	case flg.Float():
		a.SetBatch(float32(batchsize))
		err = a.trainer.L1L2Regularization(handle.XHandle(), dw.TD(), dw, w, a.gpuloss1, a.gpuloss2, a.regparams)
		if err != nil {
			return err
		}

		err = handle.Sync()
		if err != nil {
			return err
		}
		if debuggingadam {
			fmt.Println(a.params)
			//gsum, err := gocudnn.GetStringer(w.TD(), a.gsum)
			//if err != nil {
			//	panic(err)
			//}
			//fmt.Println("GsumBefore", gsum)
			//xsum, err := gocudnn.GetStringer(w.TD(), a.xsum)
			//if err != nil {
			//	panic(err)
			//}
			//fmt.Println("xsum before", xsum)
			fmt.Println("Before Update")
			dw.TogglePrintValueForStringer()
			fmt.Println(dw)
			w.TogglePrintValueForStringer()
			fmt.Println("Weights", w)

		}
		err = a.trainer.TrainValues(handle.XHandle(), dw.TD(), dw, w, a.gsum, a.xsum, a.params, (int32)(counter))
		if err != nil {
			return err
		}
		if debuggingadam {

			//gsum, err := gocudnn.GetStringer(w.TD(), a.gsum)
			//if err != nil {
			//	panic(err)
			//}
			//fmt.Println("GsumAfter", gsum)
			//xsum, err := gocudnn.GetStringer(w.TD(), a.xsum)
			//if err != nil {
			//	panic(err)
			//}
			//fmt.Println("xsum after", xsum)
			fmt.Println("After Update")
			fmt.Println("Weights", w)
			fmt.Println(dw)
			fmt.Println("(a *Adam) UpdateWeights")

			for {

			}
		}
		err = handle.Sync()
		if err != nil {
			return err
		}
		err = a.l1l2loss()
		if err != nil {
			return err
		}

	case flg.Half():
		a.SetBatch(float32(batchsize))
		err = a.trainer.L1L2Regularization(handle.XHandle(), dw.TD(), dw, w, a.gpuloss1, a.gpuloss2, a.regparams)
		if err != nil {
			return err
		}

		err = handle.Sync()
		if err != nil {
			return err
		}
		err = a.trainer.TrainValues(handle.XHandle(), dw.TD(), dw, w, a.gsum, a.xsum, a.params, (int32)(counter))
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

	}
	return handle.Sync()
}
func (a *Adam) l1l2loss() error {
	var err error
	err = nvidia.Memcpy(a.goptr1, a.gpuloss1, a.goptr1.TotalBytes())
	if err != nil {
		return err
	}
	err = nvidia.Memcpy(a.goptr2, a.gpuloss2, a.goptr2.TotalBytes())
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
func SetupAdamWandB(tctx *xtra.Handle, decay1, decay2 float32, batch int32) (*Adam, *Adam, error) {
	adam1, err := SetupAdam(tctx, decay1, decay2, batch)
	if err != nil {
		return nil, nil, err
	}
	adam2, err := SetupAdam(tctx, decay1, decay2, batch)
	return adam1, adam2, err
}

//SetupAdamWandB2 returns a trainer for both WandB and includes the learning rate
func SetupAdamWandB2(handle *cudnn.Handler, rate, dwalpha, decay1, decay2 float32, batch int32) (*Adam, *Adam, error) {
	adam1, err := SetupAdam(handle.XHandle(), decay1, decay2, batch)
	adam1.SetRates(rate, dwalpha)
	if err != nil {
		return nil, nil, err
	}
	adam2, err := SetupAdam(handle.XHandle(), decay1, decay2, batch)
	adam2.SetRates(rate, dwalpha)
	return adam1, adam2, err
}

//SetupAdam sets up adam
func SetupAdam(tctx *xtra.Handle, decay1, decay2 float32, batch int32) (*Adam, error) {

	adam := xtra.TrainingModeFlag{}.Adam()
	var dflt gocudnn.DataType
	t, err := xtra.NewTrainingDescriptor(tctx, adam, dflt.Float())
	if err != nil {
		return nil, err
	}
	reg := xtra.CreateRegParamsFloat32(decay1, decay2, float32(batch))
	x := xtra.CreateParamsFloat32(defaultadameps, defaultadamrate, defaultadambeta1, defaultadambeta2, 0)

	return &Adam{
		trainer:   t,
		params:    x,
		regparams: reg,
	}, nil
}

//SetDecays sets the decay rates for the trainer
func (a *Adam) SetDecays(l1, l2 float32) {
	a.regparams.SetDecay1(l1)
	a.regparams.SetDecay2(l2)
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

//SetRates sets rate
func (a *Adam) SetRates(rate, dwalpha float32) {
	a.params.SetRate(rate)
	a.params.SetDWalpha(dwalpha)

}

//SetBatch sets batch
func (a *Adam) SetBatch(batch float32) {
	a.regparams.SetBatch(batch)
}

//SetEps sets eps
func (a *Adam) SetEps(eps float32) {
	a.params.SetEps(eps)

}
