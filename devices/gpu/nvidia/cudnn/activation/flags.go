package activation

import (
	gocudnn "github.com/dereklstinson/GoCudnn"
	"github.com/dereklstinson/GoCudnn/xtra"
)

//Mode passes Mode flags
type Mode struct {
	m  gocudnn.ActivationMode
	xt xtra.XActivationMode
}

//Relu returns relu flag
func (m *Mode) Relu() Mode {
	m.m.Relu()
	return *m
}

//Identity passes identity.  It is used for bwd and fwd convolutionactivationbiasfwd
func (m *Mode) Identity() Mode {
	m.m.Identity()
	return *m
}

//Tanh sends a flag for the tanh activation
func (m *Mode) Tanh() Mode {
	m.m.Tanh()
	return *m
}

//ClippedRelu places a ceiling on the output
func (m *Mode) ClippedRelu() Mode {
	m.m.ClippedRelu()
	return *m
}

//Elu is the exponential linear unit
func (m *Mode) Elu() Mode {
	m.m.Elu()
	return *m

}

//Sigmoid returns sigmoid flag
func (m *Mode) Sigmoid() Mode {
	m.m.Sigmoid()
	return *m

}

//Leaky is the leaky linear unit
func (m *Mode) Leaky() Mode {
	m.xt.Leaky()
	return *m

}

//Threshhold passes a Threshhold mode flag.
//It is an experimental function.
func (m *Mode) Threshhold() Mode {
	m.xt.Threshhold()
	return *m

}

//PRelu is the Parametric activation function.
//This is an experimental function
func (m *Mode) PRelu() Mode {
	m.xt.Prelu()
	return *m

}

//Flag is a helper struct used to pass flags
type Flag struct {
	Mode    Mode
	NanFlag gocudnn.NANProp
}

//private
func (m Mode) c() gocudnn.ActivationMode {
	return gocudnn.ActivationMode(m.m)
}
func (m Mode) x() xtra.XActivationMode {
	return xtra.XActivationMode(m.xt)
}
