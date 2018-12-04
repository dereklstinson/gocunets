package gocunets

import (
	"errors"
	"image"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/layers/cnn"
	"github.com/dereklstinson/GoCuNets/layers/cnntranspose"
)

//LayerIOStats contains the minmax for an layer
type LayerIOStats struct {
	Name    string  `json:"name,omitempty"`
	IO      bool    `json:"io,omitempty"`
	Weights IOStats `json:"weights,omitempty"`
	Bias    IOStats `json:"bias,omitempty"`
}

//IOStats contains the IO minmax
type IOStats struct {
	Name  string  `json:"name,omitempty"`
	Min   float32 `json:"minx,omitempty"`
	Max   float32 `json:"maxx,omitempty"`
	Avg   float32 `json:"avg,omitempty"`
	Norm1 float32 `json:"norm_1,omitempty"`
	Norm2 float32 `json:"norm_2,omitempty"`
}

//ImagesLIO is used to hold and label the images of a section of the network.
type ImagesLIO struct {
	Layer bool        `json:"layer,omitempty"`
	Name  string      `json:"name,omitempty"`
	X     image.Image `json:"x,omitempty"`
}
type netios struct {
	name    string
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

//GetStats returns the min maxes for all the weights and biases and hidden ios in the network
func (m *Network) GetStats(handle *cudnn.Handler) ([]*LayerIOStats, error) {
	x := make([]*LayerIOStats, len(m.totalionets))
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
		/*
			dimg, err := n.cnn.Weights().DeltaT().ToOneImageColor(x, y)
			if err != nil {
				return nil, err
			}
		*/
		return &ImagesLIO{
			Layer: true,
			Name:  "CNN",
			X:     img,
			//	DX:    dimg,
		}, nil
	case n.cnntran != nil:

		img, err := n.cnntran.Weights().T().ToOneImageColor(x, y)
		if err != nil {
			return nil, err
		}
		/*
			dimg, err := n.cnntran.Weights().DeltaT().ToOneImageColor(x, y)
			if err != nil {
				return nil, err
			}
		*/
		return &ImagesLIO{
			Layer: true,
			Name:  "TransCNN",
			X:     img,
			//	DX:    dimg,
		}, nil
	case n.io != nil:

		img, err := n.io.T().ToOneImageColor(x, y)
		if err != nil {
			return nil, err
		}
		//dimg, err := n.io.DeltaT().ToOneImageColor(x, y)
		if err != nil {
			return nil, err
		}
		return &ImagesLIO{
			Layer: false,
			Name:  "IO",
			X:     img,
			//	DX:    dimg,
		}, nil
	}
	return nil, errors.New("SHouln't have got here")
}
func (l *layer) wrapnetio() *netios {
	switch {
	case l.cnn != nil:
		return &netios{name: "CNN", cnn: l.cnn}
	case l.cnntranspose != nil:
		return &netios{name: "CNN-TransPose", cnntran: l.cnntranspose}
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
func (n *netios) rebuildstathidden(handle *cudnn.Handler) error {
	switch {
	case n.io != nil:
		return n.io.SetXStatReducers(handle)
	default:
		return nil
	}
}
func (n *netios) buildminmax(handle *cudnn.Handler) error {
	switch {
	case n.cnn != nil:
		return n.cnn.SetupWStatReducers(handle)
	case n.cnntran != nil:
		return n.cnntran.SetupWStatReducers(handle)
	case n.io != nil:
		return n.io.SetXStatReducers(handle)
	default:
		return errors.New("Doesn't Contain the layer")
	}
}

func (m *Network) buildminmax(handle *cudnn.Handler, batches bool) error {
	switch {
	case !m.totalionetsinit:
		for i := range m.totalionets {
			err := m.totalionets[i].buildminmax(handle)
			if err != nil {
				return err
			}
		}
		m.totalionetsinit = true
	case m.totalionetsinit:
		for i := range m.totalionets {
			err := m.totalionets[i].rebuildstathidden(handle)
			if err != nil {
				return err
			}
		}

	}

	return nil
}

func (n *netios) minmaxes(handle *cudnn.Handler) (*LayerIOStats, error) {
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
		wavg, err := n.cnn.WAvg(handle)
		if err != nil {
			return nil, err
		}
		wnorm1, err := n.cnn.WNorm1(handle)
		if err != nil {
			return nil, err
		}
		wnorm2, err := n.cnn.WNorm2(handle)
		if err != nil {
			return nil, err
		}
		weights := IOStats{
			Name:  "Weights",
			Min:   wmin,
			Max:   wmax,
			Avg:   wavg,
			Norm1: wnorm1,
			Norm2: wnorm2,
		}
		bmax, err := n.cnn.BMax(handle)
		if err != nil {
			return nil, err
		}
		bmin, err := n.cnn.BMin(handle)
		if err != nil {
			return nil, err
		}
		bavg, err := n.cnn.BAvg(handle)
		if err != nil {
			return nil, err
		}
		bnorm1, err := n.cnn.BNorm1(handle)
		if err != nil {
			return nil, err
		}
		bnorm2, err := n.cnn.BNorm2(handle)
		if err != nil {
			return nil, err
		}
		bias := IOStats{
			Name:  "Bias",
			Min:   bmin,
			Max:   bmax,
			Avg:   bavg,
			Norm1: bnorm1,
			Norm2: bnorm2,
		}
		return &LayerIOStats{
			Name:    n.name,
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
		wavg, err := n.cnntran.WAvg(handle)
		if err != nil {
			return nil, err
		}
		wnorm1, err := n.cnntran.WNorm1(handle)
		if err != nil {
			return nil, err
		}
		wnorm2, err := n.cnntran.WNorm2(handle)
		if err != nil {
			return nil, err
		}
		weights := IOStats{
			Name:  "Weights",
			Min:   wmin,
			Max:   wmax,
			Avg:   wavg,
			Norm1: wnorm1,
			Norm2: wnorm2,
		}
		bmax, err := n.cnntran.BMax(handle)
		if err != nil {
			return nil, err
		}
		bmin, err := n.cnntran.BMin(handle)
		if err != nil {
			return nil, err
		}
		bavg, err := n.cnntran.BAvg(handle)
		if err != nil {
			return nil, err
		}
		bnorm1, err := n.cnntran.BNorm1(handle)
		if err != nil {
			return nil, err
		}
		bnorm2, err := n.cnntran.BNorm2(handle)
		if err != nil {
			return nil, err
		}
		bias := IOStats{
			Name:  "Bias",
			Min:   bmin,
			Max:   bmax,
			Avg:   bavg,
			Norm1: bnorm1,
			Norm2: bnorm2,
		}
		return &LayerIOStats{
			Name:    n.name,
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
		wavg, err := n.io.AvgX(handle)
		if err != nil {
			return nil, err
		}
		wnorm1, err := n.io.Norm1X(handle)
		if err != nil {
			return nil, err
		}
		wnorm2, err := n.io.Norm2X(handle)
		if err != nil {
			return nil, err
		}
		weights := IOStats{
			Name:  "Ouput",
			Min:   wmin,
			Max:   wmax,
			Avg:   wavg,
			Norm1: wnorm1,
			Norm2: wnorm2,
		}

		return &LayerIOStats{
			Name:    n.name,
			Weights: weights,
			IO:      true,
		}, nil
	}

	return nil, errors.New("Unsupported netio in wrapper")
}
