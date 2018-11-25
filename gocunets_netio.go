package gocunets

import (
	"errors"

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
	Name  string    `json:"name,omitempty"`
	Minx  []float32 `json:"minx,omitempty"`
	Maxx  []float32 `json:"maxx,omitempty"`
	Mindx []float32 `json:"mindx,omitempty"`
	Maxdx []float32 `json:"maxdx,omitempty"`
}
type netios struct {
	cnn     *cnn.Layer
	cnntran *cnntranspose.Layer
	io      *layers.IO
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
func wrapnetio(input interface{}) *netios {
	switch l := input.(type) {

	case *cnntranspose.Layer:
		return &netios{
			cnntran: l,
		}
	case *cnn.Layer:
		return &netios{
			cnn: l,
		}
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
			Minx:  wmin,
			Maxx:  wmax,
			Mindx: dwmin,
			Maxdx: dwmax,
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
			Minx:  bmin,
			Maxx:  bmax,
			Mindx: dbmin,
			Maxdx: dbmax,
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
			Minx:  wmin,
			Maxx:  wmax,
			Mindx: dwmin,
			Maxdx: dwmax,
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
			Minx:  bmin,
			Maxx:  bmax,
			Mindx: dbmin,
			Maxdx: dbmax,
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
			Minx:  wmin,
			Maxx:  wmax,
			Mindx: dwmin,
			Maxdx: dwmax,
		}

		return &LayerIOMinMax{
			Name:    "IO-Hidden",
			Weights: weights,
		}, nil
	}

	return nil, errors.New("Unsupported netio in wrapper")
}
