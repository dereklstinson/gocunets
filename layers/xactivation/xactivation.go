package xactivation

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/gocudnn/xactivation"
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Layer is an activation layer
type Layer struct {
	act        *xactivation.Ops
	amode      gocudnn.XActivationMode
	updateable bool
	memmanaged bool
	alphas     *layers.IO
	xsumgsum   *layers.IO
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
const defaultdecay1 = float32(0.0)
const defaultdecay2 = float32(0.0)

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
		return l.act.FwdProp(handle, x.T(), y.T(), l.alphas.T(), l.alphas.DeltaT())
	case aflg.AdvanceThreshRandomRelu():
		return l.act.FwdProp(handle, x.T(), y.T(), l.alphas.T(), l.alphas.DeltaT())
	}

	return errors.New("Unsupported Actrivation Mode")
}

//BackProp does the backprop operation
func (l *Layer) BackProp(handle *gocudnn.XHandle, x, y *layers.IO) error {
	//_, _, dims, _ := x.Properties()
	var aflg gocudnn.XActivationModeFlag
	switch l.amode {
	case aflg.Leaky():
		return l.act.BwdProp(handle, x.T(), x.DeltaT(), y.DeltaT(), nil, nil)
	case aflg.ParaChan():
		return l.act.BwdProp(handle, x.T(), x.DeltaT(), y.DeltaT(), l.alphas.T(), l.alphas.DeltaT())
	case aflg.AdvanceThreshRandomRelu():
		return l.act.BwdProp(handle, x.T(), x.DeltaT(), y.DeltaT(), l.alphas.T(), l.alphas.DeltaT())
	}

	return errors.New("Unsupported Actrivation Mode")

}
