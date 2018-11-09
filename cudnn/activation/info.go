package activation

import (
	"github.com/dereklstinson/GoCuNets/cudnn"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//OpInfo contains the necissary information to build an activation Ops
type OpInfo struct {
	Mode    Mode          `json:"Mode"`
	NanProp cudnn.NanMode `json:"NanProp"`
	Coef    float64       `json:"Coef"`
}

//Flags returns the flags that are needed to create an Activation struct
func Flags() (gocudnn.ActivationModeFlag, gocudnn.PropagationNANFlag) {
	return gocudnn.ActivationModeFlag{}, gocudnn.PropagationNANFlag{}
}

//Stage builds and returns *Op from the info inside of the info type
func (input OpInfo) Stage(h *cudnn.Handler) (*Ops, error) {
	return Stage(h, input.Mode, input.NanProp, input.Coef)
}
