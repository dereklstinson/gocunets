package reshape

import (
	"github.com/dereklstinson/GoCuNets/gocudnn/reshapes"
	"github.com/dereklstinson/GoCudnn"
)

//Layer is the that type that handles reshape methods
type Layer struct {
	op *reshapes.Ops
}

//Build builds the layer
func Build(handle *gocudnn.XHandle) (*Layer, error) {
	op, err := reshapes.Stage(handle)
	return &Layer{op: op}, err
}
