package cudnn

import gocudnn "github.com/dereklstinson/GoCudnn"

//NanMode is used for flags
type NanMode gocudnn.PropagationNAN

//Cu returns a gocudnn.PropagationNAN
func (n NanMode) Cu() gocudnn.PropagationNAN {
	return gocudnn.PropagationNAN(n)
}

//NanModeFlag passes NanMode flags through methods
type NanModeFlag struct {
	c gocudnn.PropagationNANFlag
}

//PropNAN sends a flag to allow Nan to propigate
func (n NanModeFlag) PropNAN() NanMode {
	return NanMode(n.c.PropagateNan())
}

//NoPropNAN sends a flag to not allow Nan to propigate
func (n NanModeFlag) NoPropNAN() NanMode {
	return NanMode(n.c.NotPropagateNan())
}
