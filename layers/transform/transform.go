package transform

/*
import (
	"errors"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCudnn"
)

type Layer struct {
	t *gocudnn.TransformD
}

var frmtflag gocudnn.TensorFormat

const alphadefault = float64(1)
const betadefault = float64(0)

func (l *Layer) GetOutputIO(x *layers.IO) (y *layers.IO, err error) {
	switch x.T().Format() {
	case frmtflag.NCHW():
		x.T().Dims():

	case frmtflag.NHWC():
	default:
		return nil, errors.New("Unsupported Format")
	}

}
func (l *Layer) Forward(h *cudnn.Handler, x, y *layers.IO) (err error) {
	gocudnn.TransformTensor()
}
func (l *Layer) Backward()
*/
