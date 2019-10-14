package activation

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//OpInfo contains the necissary information to build an activation Ops
type OpInfo struct {
	Mode     Mode             `json:"mode,omitempty"`
	NanProp  gocudnn.NANProp  `json:"nan_prop,omitempty"`
	Coef     float64          `json:"coef,omitempty"`
	DataType gocudnn.DataType `json:"data_type,omitempty"`
}

//Flags returns the flags that are needed to create an Activation struct
func Flags() (act gocudnn.ActivationMode, Nan gocudnn.NANProp) {
	return
}

//Stage builds and returns *Op from the info inside of the info type
func (input OpInfo) Stage(h *cudnn.Handler) (*Ops, error) {
	return Stage(h, input.Mode, input.DataType, input.NanProp, input.Coef)
}
