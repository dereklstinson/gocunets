package activation

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	gocudnn "github.com/dereklstinson/GoCudnn"
	"github.com/dereklstinson/GoCudnn/xtra"
)

//ModeFlag passes Mode flags
type ModeFlag struct {
	m gocudnn.ActivationModeFlag
	x xtra.XActivationModeFlag
}

//Mode is the flag that holds the activation mode
type Mode uint

//Relu returns relu flag
func (m ModeFlag) Relu() Mode {
	return Mode(m.m.Relu())
}

//Identity passes identity.  It is used for bwd and fwd convolutionactivationbiasfwd
func (m ModeFlag) Identity() Mode {
	return Mode(m.m.Identity())
}

//Tanh sends a flag for the tanh activation
func (m ModeFlag) Tanh() Mode {
	return Mode(m.m.Tanh())
}

//ClippedRelu places a ceiling on the output
func (m ModeFlag) ClippedRelu() Mode {
	return Mode(m.m.ClippedRelu())
}

//Elu is the exponential linear unit
func (m ModeFlag) Elu() Mode {
	return Mode(m.m.Elu())
}

//Sigmoid returns sigmoid flag
func (m ModeFlag) Sigmoid() Mode {
	return Mode(m.m.Sigmoid())
}

//Leaky is the leaky linear unit
func (m ModeFlag) Leaky() Mode {
	return Mode(m.x.Leaky())
}

//Threshhold passes a Threshhold mode flag.
//It is an experimental function.
func (m ModeFlag) Threshhold() Mode {
	return Mode(m.x.Threshhold())
}

//PRelu is the Parametric activation function.
//This is an experimental function
func (m ModeFlag) PRelu() Mode {
	return Mode(m.x.Prelu())
}

//Flag is a helper struct used to pass flags
type Flag struct {
	Mode    ModeFlag
	NanFlag cudnn.NanMode
}

//private
func (m Mode) c() gocudnn.ActivationMode {
	return gocudnn.ActivationMode(m)
}
func (m Mode) x() xtra.XActivationMode {
	return xtra.XActivationMode(m)
}
