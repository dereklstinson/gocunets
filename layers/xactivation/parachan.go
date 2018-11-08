package xactivation

import (
	"github.com/dereklstinson/GoCuNets/cudnn/xactivation"
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//SetupParaChan sets up a static parametric HWC takes up a ton of mem especially if adam and adagrad is the trainmode
func SetupParaChan(h *gocudnn.XHandle, channels int32, frmt gocudnn.TensorFormat, dtype gocudnn.DataType, trainmode gocudnn.TrainingMode, managedmem bool) (*Layer, error) {

	var trp gocudnn.TrainingParams
	var xactmodeflg gocudnn.XActivationModeFlag
	var tflg gocudnn.TensorFormatFlag
	trp.SetBeta1(defaultadambeta1)
	trp.SetBeta2(defaultadambeta2)
	trp.SetEps(defaultadameps)
	trp.SetRate(defaultadamrate)
	var reg gocudnn.RegParams
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

	err = alphas.T().SetRandomNormal(.24, .26)
	if err != nil {
		return nil, err
	}

	xsumgsum, err := layers.BuildIO(frmt, dtype, dims, managedmem)
	if err != nil {
		return nil, err
	}

	op, err := xactivation.Stage(h, xactmodeflg.ParaChan(), trainmode, dtype, 0)

	lyr := &Layer{
		act:        op,
		amode:      xactmodeflg.ParaChan(),
		memmanaged: managedmem,
		xsumgsum:   xsumgsum,
		alphas:     alphas,
		d1:         defaultdecay1,
		d2:         defaultdecay2,
		r:          reg,
		t:          trp,
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
