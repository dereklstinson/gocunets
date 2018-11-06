package xactivation

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/gocudnn/xactivation"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/utils"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Layer is an activation layer
type Layer struct {
	act        *xactivation.Ops
	amode      gocudnn.XActivationMode
	updateable bool
	memmanaged bool
	alphas     *layers.IO
	betas      *layers.IO
	xsumab     *layers.IO
	gsumab     *layers.IO
	l2         *gocudnn.Malloced
	l1         *gocudnn.Malloced
	l1g        []float32
	l2g        []float32
	l1gptr     *gocudnn.GoPointer
	l2gptr     *gocudnn.GoPointer
	t          gocudnn.TrainingParams
	r          gocudnn.RegParams
	d1         float32
	d2         float32
}

const defaultadambeta1 = 0.9
const defaultadambeta2 = 0.999
const defaultadameps = float32(1e-8)
const defaultadamrate = .001
const defaulttrainmode = gocudnn.TrainingMode(4) //This is adam
const defaultcoef = 6
const defaultdecay1 = float32(.00001)
const defaultdecay2 = float32(.00001)

//SetupLeaky sets up the basic Leaky xactivation.
func SetupLeaky(h *gocudnn.XHandle, dtype gocudnn.DataType) (*Layer, error) {

	var xactmodeflg gocudnn.XActivationModeFlag

	op, err := xactivation.Stage(h, xactmodeflg.Leaky(), defaulttrainmode, dtype, 128.0)
	if err != nil {
		return nil, err
	}
	return &Layer{
		act:        op,
		amode:      xactmodeflg.Leaky(),
		updateable: false,
	}, nil

}

//SetupParaChan sets up a static parametric HWC takes up a ton of mem especially if adam and adagrad is the trainmode
func SetupParaChan(h *gocudnn.XHandle, channels int32, frmt gocudnn.TensorFormat, dtype gocudnn.DataType, trainmode gocudnn.TrainingMode, managedmem bool) (*Layer, error) {
	var tmflg gocudnn.TrainingModeFlag
	var trp gocudnn.TrainingParams
	var xactmodeflg gocudnn.XActivationModeFlag
	var tflg gocudnn.TensorFormatFlag
	trp.SetBeta1(defaultadambeta1)
	trp.SetBeta2(defaultadambeta2)
	trp.SetEps(defaultadameps)
	trp.SetRate(defaultadamrate)
	var reg gocudnn.RegParams
	//reg.SetBatch(float32(0))
	reg.SetDecay1(float32(defaultdecay1))
	reg.SetDecay2(float32(defaultdecay2))
	var dims []int32
	if frmt == tflg.NHWC() {
		dims = []int32{int32(1), int32(1), int32(1), channels}
	} else {
		dims = []int32{int32(1), channels, int32(1), int32(1)}
	}
	alphas, err := layers.BuildIO(frmt, dtype, dims, managedmem)
	if err != nil {
		return nil, err
	}
	betas, err := layers.BuildIO(frmt, dtype, dims, managedmem)
	if err != nil {
		return nil, err
	}
	err = alphas.T().SetRandomNormal(.9, 1.1)
	if err != nil {
		return nil, err
	}
	err = betas.T().SetRandomNormal(0.005, 0.02)
	if err != nil {
		return nil, err
	}

	gsumab, err := layers.BuildIO(frmt, dtype, dims, managedmem)
	if err != nil {
		return nil, err
	}
	var xsumab *layers.IO
	if trainmode == tmflg.AdaGrad() || trainmode == tmflg.Adam() {
		xsumab, err = layers.BuildIO(frmt, dtype, dims, managedmem)
		if err != nil {
			return nil, err
		}
	}

	op, err := xactivation.Stage(h, xactmodeflg.ParaChan(), trainmode, dtype, 0)

	lyr := &Layer{
		act:        op,
		amode:      xactmodeflg.ParaChan(),
		memmanaged: managedmem,
		xsumab:     xsumab,
		gsumab:     gsumab,
		alphas:     alphas,
		betas:      betas,
		d1:         defaultdecay1,
		d2:         defaultdecay2,
		r:          reg,
		updateable: true,
	}
	if managedmem == true {
		lyr.l1, err = gocudnn.UnifiedMangedGlobal(4)
		if err != nil {
			return nil, err
		}
		lyr.l2, err = gocudnn.UnifiedMangedGlobal(4)
		if err != nil {
			return nil, err
		}
	} else {
		lyr.l1, err = gocudnn.Malloc(4)
		if err != nil {
			return nil, err
		}
		lyr.l2, err = gocudnn.Malloc(4)
		if err != nil {
			return nil, err
		}
	}
	lyr.l1g = make([]float32, 1)
	lyr.l2g = make([]float32, 1)
	//lyr.l1g[0] = defaultdecay1
	//lyr.l2g[0] = defaultdecay2
	lyr.l1gptr, err = gocudnn.MakeGoPointer(lyr.l1g)
	if err != nil {
		return nil, err
	}
	lyr.l2gptr, err = gocudnn.MakeGoPointer(lyr.l2g)
	if err != nil {
		return nil, err
	}
	return lyr, nil
}

//SetupParametricStaicHWC sets up a static parametric HWC takes up a ton of mem especially if adam and adagrad is the trainmode
func SetupParametricStaicHWC(h *gocudnn.XHandle, inputdims []int32, frmt gocudnn.TensorFormat, dtype gocudnn.DataType, trainmode gocudnn.TrainingMode, managedmem bool) (*Layer, error) {
	var tmflg gocudnn.TrainingModeFlag
	var trp gocudnn.TrainingParams
	var xactmodeflg gocudnn.XActivationModeFlag
	trp.SetBeta1(defaultadambeta1)
	trp.SetBeta2(defaultadambeta2)
	trp.SetEps(defaultadameps)
	trp.SetRate(defaultadamrate)
	var reg gocudnn.RegParams
	reg.SetBatch(float32(inputdims[0]))
	reg.SetDecay1(float32(defaultdecay1))
	reg.SetDecay2(float32(defaultdecay2))
	dims := []int32{int32(1), inputdims[1], inputdims[2], inputdims[3]}
	alphas, err := layers.BuildIO(frmt, dtype, dims, managedmem)
	if err != nil {
		return nil, err
	}

	betas, err := layers.BuildIO(frmt, dtype, dims, managedmem)
	if err != nil {
		return nil, err
	}
	err = alphas.T().SetRandom(.1, .01, float64(utils.FindVolumeInt32(dims)))
	if err != nil {
		return nil, err
	}
	err = betas.T().SetRandom(1, .05, float64(utils.FindVolumeInt32(dims)))
	if err != nil {
		return nil, err
	}

	gsumab, err := layers.BuildIO(frmt, dtype, dims, managedmem)
	if err != nil {
		return nil, err
	}
	var xsumab *layers.IO
	if trainmode == tmflg.AdaGrad() || trainmode == tmflg.Adam() {
		xsumab, err = layers.BuildIO(frmt, dtype, dims, managedmem)
		if err != nil {
			return nil, err
		}
	}

	op, err := xactivation.Stage(h, xactmodeflg.ParaPlus(), trainmode, dtype, 0)

	lyr := &Layer{
		act:        op,
		amode:      xactmodeflg.ParaPlus(),
		memmanaged: managedmem,
		xsumab:     xsumab,
		gsumab:     gsumab,
		alphas:     alphas,
		betas:      betas,
		d1:         defaultdecay1,
		d2:         defaultdecay2,
		updateable: true,
	}
	if managedmem == true {
		lyr.l1, err = gocudnn.UnifiedMangedGlobal(4)
		if err != nil {
			return nil, err
		}
		lyr.l2, err = gocudnn.UnifiedMangedGlobal(4)
		if err != nil {
			return nil, err
		}
	} else {
		lyr.l1, err = gocudnn.Malloc(4)
		if err != nil {
			return nil, err
		}
		lyr.l2, err = gocudnn.Malloc(4)
		if err != nil {
			return nil, err
		}
	}
	lyr.l1g = make([]float32, 1)
	lyr.l2g = make([]float32, 1)
	lyr.l1g[0] = defaultdecay1
	lyr.l2g[0] = defaultdecay2
	lyr.l1gptr, err = gocudnn.MakeGoPointer(lyr.l1g)
	if err != nil {
		return nil, err
	}
	lyr.l2gptr, err = gocudnn.MakeGoPointer(lyr.l2g)
	if err != nil {
		return nil, err
	}
	return lyr, nil
}

//Setl1l2 sets the decay for l1l2 regularization
func (l *Layer) Setl1l2(decay1, decay2 float32) {
	l.d1 = decay1
	l.d2 = decay2
	l.r.SetDecay1(decay1)
	l.r.SetDecay2(decay2)
}

//SetBatch sets the batch for regularization
func (l *Layer) SetBatch(batch float32) {
	l.r.SetBatch(batch)
}

//ForwardProp does the forward prop
func (l *Layer) ForwardProp(handle *gocudnn.XHandle, x, y *layers.IO) error {
	var aflg gocudnn.XActivationModeFlag
	switch l.amode {
	case aflg.Leaky():
		return l.act.FwdProp(handle, x.T(), y.T(), nil, nil)
	case aflg.ParaChan():
		return l.act.FwdProp(handle, x.T(), y.T(), l.alphas.T(), l.betas.T())
	case aflg.ParaPlus():
		return l.act.FwdProp(handle, x.T(), y.T(), l.alphas.T(), l.betas.T())
	}

	return errors.New("Unsupported Actrivation Mode")
}

//BackProp does the backprop operation
func (l *Layer) BackProp(handle *gocudnn.XHandle, x, y *layers.IO) error {
	//_, _, dims, _ := x.Properties()
	var aflg gocudnn.XActivationModeFlag
	switch l.amode {
	case aflg.Leaky():
		return l.act.BwdProp(handle, x.T(), x.DeltaT(), y.DeltaT(), nil, nil, nil, nil)
	case aflg.ParaChan():
		return l.act.BwdProp(handle, x.T(), x.DeltaT(), y.DeltaT(), l.alphas.T(), l.alphas.DeltaT(), l.betas.T(), l.betas.DeltaT())
	case aflg.ParaPlus():
		return l.act.BwdProp(handle, x.T(), x.DeltaT(), y.DeltaT(), l.alphas.T(), l.alphas.DeltaT(), l.betas.T(), l.betas.DeltaT())
	}

	return errors.New("Unsupported Actrivation Mode")

}

//UpdateParams updates the params as long as it is set up that way
func (l *Layer) UpdateParams(handle *gocudnn.XHandle, batchsize int) error {
	if l.updateable == false {
		return nil
	}
	l.r.SetBatch(float32(batchsize))
	var err error
	if l.xsumab == nil {

		err = l.act.UpdateParams(
			handle,
			batchsize,
			l.alphas.T(),
			l.alphas.DeltaT(),
			l.betas.T(),
			l.betas.DeltaT(),
			l.gsumab.T().Memer(),
			nil,
			l.gsumab.DeltaT().Memer(),
			nil,
			l.l1,
			l.l2,
			l.t,
			l.r)
	} else {
		/*
			l.alphas.T().PrintDeviceMem("alphas: ")
			l.alphas.DeltaT().PrintDeviceMem("dalpha: ")
			l.betas.T().PrintDeviceMem("betas: ")
			l.betas.DeltaT().PrintDeviceMem("dbeta: ")
			l.gsumab.T().PrintDeviceMem("alpha gsum :")
			l.gsumab.DeltaT().PrintDeviceMem("beta gsum :")
			l.xsumab.T().PrintDeviceMem("alpha xsum :")
			l.xsumab.DeltaT().PrintDeviceMem("beta xsum :")
		*/
		err = l.act.UpdateParams(
			handle,
			batchsize,
			l.alphas.T(),
			l.alphas.DeltaT(),
			l.betas.T(),
			l.betas.DeltaT(),
			l.gsumab.T().Memer(),
			l.xsumab.T().Memer(),
			l.gsumab.DeltaT().Memer(),
			l.xsumab.DeltaT().Memer(),
			l.l1,
			l.l2,
			l.t,
			l.r)
		/*
					l.alphas.T().PrintDeviceMem("alphas: ")
					l.alphas.DeltaT().PrintDeviceMem("dalpha: ")
					l.betas.T().PrintDeviceMem("betas: ")
					l.betas.DeltaT().PrintDeviceMem("dbeta: ")
					l.gsumab.T().PrintDeviceMem("alpha gsum :")
					l.gsumab.DeltaT().PrintDeviceMem("beta gsum :")
					l.xsumab.T().PrintDeviceMem("alpha xsum :")
					l.xsumab.DeltaT().PrintDeviceMem("beta xsum :")
			for {

				}
		*/

	}
	if err != nil {
		return err
	}
	if l.memmanaged == true {

		if l.d1 != 0 {
			err = gocudnn.CudaMemCopy(l.l1gptr, l.l1, 4, gocudnn.MemcpyKindFlag{}.Default())
			if err != nil {
				return err
			}
		}
		if l.d2 != 0 {
			err = gocudnn.CudaMemCopy(l.l1gptr, l.l1, 4, gocudnn.MemcpyKindFlag{}.Default())
			if err != nil {
				return err
			}
		}
	} else {
		if l.d1 != 0 {
			err = gocudnn.CudaMemCopy(l.l1gptr, l.l1, 4, gocudnn.MemcpyKindFlag{}.DeviceToHost())
			if err != nil {
				return err
			}
		}
		if l.d2 != 0 {
			err = gocudnn.CudaMemCopy(l.l1gptr, l.l1, 4, gocudnn.MemcpyKindFlag{}.DeviceToHost())
			if err != nil {
				return err
			}
		}

	}
	return nil
}
