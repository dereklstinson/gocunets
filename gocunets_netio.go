package gocunets

import (
	"errors"
	"image"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	"github.com/dereklstinson/GoCuNets/layers/cnntranspose"
)

//LayerIOMinMax contains the minmax for an layer
type LayerIOMinMax struct {
	Name    string   `json:"name,omitempty"`
	IO      bool     `json:"io,omitempty"`
	Weights IOMinMax `json:"weights,omitempty"`
	Bias    IOMinMax `json:"bias,omitempty"`
}

//IOMinMax contains the IO minmax
type IOMinMax struct {
	Name  string  `json:"name,omitempty"`
	Minx  float32 `json:"minx,omitempty"`
	Maxx  float32 `json:"maxx,omitempty"`
	Mindx float32 `json:"mindx,omitempty"`
	Maxdx float32 `json:"maxdx,omitempty"`
}

//ImagesLIO is used to hold and label the images of a section of the network.
type ImagesLIO struct {
	Layer bool        `json:"layer,omitempty"`
	Name  string      `json:"name,omitempty"`
	X     image.Image `json:"x,omitempty"`
	DX    image.Image `json:"dx,omitempty"`
}
type netios struct {
	cnn     *cnn.Layer
	cnntran *cnntranspose.Layer
	io      *layers.IO
}

//GetLayerImages will return an array of images of the io of the network
func (m *Network) GetLayerImages(handle *cudnn.Handler, x, y int) ([]*ImagesLIO, error) {
	imgios := make([]*ImagesLIO, len(m.totalionets))
	var err error
	for i := range m.totalionets {
		imgios[i], err = m.totalionets[i].images(handle, x, y)
		if err != nil {
			return nil, err
		}
	}
	return imgios, nil
}

//GetMinMaxes returns the min maxes for all the weights and biases and hidden ios in the network
func (m *Network) GetMinMaxes(handle *cudnn.Handler) ([]*LayerIOMinMax, error) {
	x := make([]*LayerIOMinMax, len(m.totalionets))
	var err error
	for i := range m.totalionets {
		x[i], err = m.totalionets[i].minmaxes(handle)
		if err != nil {
			return nil, err
		}
	}
	return x, nil
}
func (n *netios) images(handle *cudnn.Handler, x, y int) (*ImagesLIO, error) {
	switch {
	case n.cnn != nil:

		img, err := n.cnn.Weights().T().ToOneImageColor(x, y)
		if err != nil {
			return nil, err
		}
		dimg, err := n.cnn.Weights().DeltaT().ToOneImageColor(x, y)
		if err != nil {
			return nil, err
		}
		return &ImagesLIO{
			Layer: true,
			Name:  "CNN",
			X:     img,
			DX:    dimg,
		}, nil
	case n.cnntran != nil:

		img, err := n.cnntran.Weights().T().ToOneImageColor(x, y)
		if err != nil {
			return nil, err
		}
		dimg, err := n.cnntran.Weights().DeltaT().ToOneImageColor(x, y)
		if err != nil {
			return nil, err
		}
		return &ImagesLIO{
			Layer: true,
			Name:  "TransCNN",
			X:     img,
			DX:    dimg,
		}, nil
	case n.io != nil:

		img, err := n.io.T().ToOneImageColor(x, y)
		if err != nil {
			return nil, err
		}
		dimg, err := n.io.DeltaT().ToOneImageColor(x, y)
		if err != nil {
			return nil, err
		}
		return &ImagesLIO{
			Layer: false,
			Name:  "IO",
			X:     img,
			DX:    dimg,
		}, nil
	}
	return nil, errors.New("SHouln't have got here")
}
func (l *layer) wrapnetio() *netios {
	switch {
	case l.cnn != nil:
		return &netios{cnn: l.cnn}
	case l.cnntranspose != nil:
		return &netios{cnntran: l.cnntranspose}
	default:
		return nil
	}

}
func wrapnetio(input interface{}) *netios {
	switch l := input.(type) {

	case *layer:
		return l.wrapnetio()
	case *layers.IO:
		return &netios{
			io: l,
		}
	default:
		return nil
	}
}
func (n *netios) rebuildminmaxhidden(handle *cudnn.Handler, batches bool) error {
	switch {
	case n.io != nil:
		return n.io.SetMinMaxReducers(handle, batches)
	default:
		return nil
	}
}
func (n *netios) buildminmax(handle *cudnn.Handler, batches bool) error {
	switch {
	case n.cnn != nil:
		return n.cnn.SetupMinMaxReducers(handle, batches)
	case n.cnntran != nil:
		return n.cnntran.SetupMinMaxReducers(handle, batches)
	case n.io != nil:
		return n.io.SetMinMaxReducers(handle, batches)
	default:
		return errors.New("Doesn't Contain the layer")
	}
}

func (m *Network) buildminmax(handle *cudnn.Handler, batches bool) error {
	switch {
	case !m.totalionetsinit:
		for i := range m.totalionets {
			err := m.totalionets[i].buildminmax(handle, batches)
			if err != nil {
				return err
			}
		}
		m.totalionetsinit = true
	case m.totalionetsinit:
		for i := range m.totalionets {
			err := m.totalionets[i].rebuildminmaxhidden(handle, batches)
			if err != nil {
				return err
			}
		}

	}

	return nil
}
func (n *netios) minmaxes(handle *cudnn.Handler) (*LayerIOMinMax, error) {
	switch {
	case n.cnn != nil:
		var err error
		wmin, err := n.cnn.WMin(handle)
		if err != nil {
			return nil, err
		}
		wmax, err := n.cnn.WMax(handle)
		if err != nil {
			return nil, err
		}
		dwmin, err := n.cnn.DWMin(handle)
		if err != nil {
			return nil, err
		}
		dwmax, err := n.cnn.DWMax(handle)
		if err != nil {
			return nil, err
		}
		weights := IOMinMax{
			Name:  "Weights",
			Minx:  wmin[0],
			Maxx:  wmax[0],
			Mindx: dwmin[0],
			Maxdx: dwmax[0],
		}
		bmax, err := n.cnn.BMax(handle)
		if err != nil {
			return nil, err
		}
		bmin, err := n.cnn.BMin(handle)
		if err != nil {
			return nil, err
		}
		dbmax, err := n.cnn.DBMax(handle)
		if err != nil {
			return nil, err
		}
		dbmin, err := n.cnn.DBMin(handle)
		if err != nil {
			return nil, err
		}
		bias := IOMinMax{
			Name:  "Bias",
			Minx:  bmin[0],
			Maxx:  bmax[0],
			Mindx: dbmin[0],
			Maxdx: dbmax[0],
		}
		return &LayerIOMinMax{
			Name:    "CNN",
			Weights: weights,
			Bias:    bias,
		}, nil
	case n.cnntran != nil:
		var err error
		wmin, err := n.cnntran.WMin(handle)
		if err != nil {
			return nil, err
		}
		wmax, err := n.cnntran.WMax(handle)
		if err != nil {
			return nil, err
		}
		dwmin, err := n.cnntran.DWMin(handle)
		if err != nil {
			return nil, err
		}
		dwmax, err := n.cnntran.DWMax(handle)
		if err != nil {
			return nil, err
		}
		weights := IOMinMax{
			Name:  "Weights",
			Minx:  wmin[0],
			Maxx:  wmax[0],
			Mindx: dwmin[0],
			Maxdx: dwmax[0],
		}
		bmax, err := n.cnntran.BMax(handle)
		if err != nil {
			return nil, err
		}
		bmin, err := n.cnntran.BMin(handle)
		if err != nil {
			return nil, err
		}
		dbmax, err := n.cnntran.DBMax(handle)
		if err != nil {
			return nil, err
		}
		dbmin, err := n.cnntran.DBMin(handle)
		if err != nil {
			return nil, err
		}
		bias := IOMinMax{
			Name:  "Bias",
			Minx:  bmin[0],
			Maxx:  bmax[0],
			Mindx: dbmin[0],
			Maxdx: dbmax[0],
		}
		return &LayerIOMinMax{
			Name:    "CNN-Transpose",
			Weights: weights,
			Bias:    bias,
		}, nil
	case n.io != nil:
		var err error
		wmin, err := n.io.MinX(handle)
		if err != nil {
			return nil, err
		}
		wmax, err := n.io.MaxX(handle)
		if err != nil {
			return nil, err
		}
		dwmin, err := n.io.MinDX(handle)
		if err != nil {
			return nil, err
		}
		dwmax, err := n.io.MaxDX(handle)
		if err != nil {
			return nil, err
		}
		weights := IOMinMax{
			Name:  "Ouput",
			Minx:  wmin[0],
			Maxx:  wmax[0],
			Mindx: dwmin[0],
			Maxdx: dwmax[0],
		}

		return &LayerIOMinMax{
			Name:    "IO-Hidden",
			Weights: weights,
		}, nil
	}

	return nil, errors.New("Unsupported netio in wrapper")
}
