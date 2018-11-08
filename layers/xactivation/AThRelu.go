package xactivation

import (
	"github.com/dereklstinson/GoCuNets/gocudnn/xactivation"
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//SetupAdvanceThreshRelu is statically set but there are thresholds that have to be made to pass and such
func SetupAdvanceThreshRelu(h *gocudnn.XHandle, inputdims []int32, frmt gocudnn.TensorFormat, dtype gocudnn.DataType, trainmode gocudnn.TrainingMode, managedmem bool) (*Layer, error) {

	var xactmodeflg gocudnn.XActivationModeFlag

	dims := []int32{int32(1), inputdims[1], inputdims[2], inputdims[3]}
	alphas, err := layers.BuildIO(frmt, dtype, dims, managedmem)
	if err != nil {
		return nil, err
	}

	err = alphas.T().SetRandomNormal(.01, .1)
	if err != nil {
		return nil, err
	}
	err = alphas.DeltaT().SetRandomNormal(0, .07)
	if err != nil {
		return nil, err
	}

	op, err := xactivation.Stage(h, xactmodeflg.AdvanceThreshRandomRelu(), trainmode, dtype, 0)

	lyr := &Layer{
		act:        op,
		amode:      xactmodeflg.AdvanceThreshRandomRelu(),
		memmanaged: managedmem,
		alphas:     alphas,
		updateable: false,
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
