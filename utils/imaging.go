package utils

import (
	"image"

	"github.com/dereklstinson/GoCuNets/gocudnn/reshapes"
	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Imager takes tensors and to the best its ability turn it into an image.Image
type Imager struct {
	shaper *reshapes.Ops
}

//MakeImager makes an imager
func MakeImager(handle *gocudnn.XHandle) (*Imager, error) {
	shpr, err := reshapes.Stage(handle)
	return &Imager{
		shaper: shpr,
	}, nil
}

//TileBatches will take the batches and lay place them withing the HWC space like tiles
func (im *Imager) TileBatches(handle *gocudnn.XHandle, x *tensor.Volume, h, w int) (image.Image, error) {
	frmt, dtype, dims, err := x.Properties()
	if err != nil {
		return nil, err
	}
	frmt, dtype, dims, err := m.shaper.GetB2SOutputProperties(handle, x, h, w)
	if err != nil {
		return nil, err
	}

	return nil, nil
}
