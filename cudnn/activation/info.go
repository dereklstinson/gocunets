package activation

import gocudnn "github.com/dereklstinson/GoCudnn"

//OpInfo contains the necissary information to build an activation Ops
type OpInfo struct {
	Mode    gocudnn.ActivationMode `json:"Mode"`
	NanProp gocudnn.PropagationNAN `json:"NanProp"`
	Coef    float64                `json:"Coef"`
}

//Flags returns the flags that are needed to create an Activation struct
func Flags() (gocudnn.ActivationModeFlag, gocudnn.PropagationNANFlag) {
	return gocudnn.ActivationModeFlag{}, gocudnn.PropagationNANFlag{}
}

//Stage builds and returns *Op from the info inside of the info type
func (input OpInfo) Stage() (*Ops, error) {
	return StageOperation(input.Mode, input.NanProp, input.Coef)
}
