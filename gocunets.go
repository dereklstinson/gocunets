package gocunets

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Settings are the setttings for the topology of the network
type Settings struct {
	Format gocudnn.TensorFormat `json:"format,omitempty"`
	DType  gocudnn.DataType     `json:"d_type,omitempty"`
	Manage bool                 `json:"manage,omitempty"`
	Layers []Layer              `json:"layers,omitempty"`
}

//Layer is a layer that has the name the dims and what flags it needs
type Layer struct {
	Name  string `json:"name,omitempty"`
	Dims  []int  `json:"dims,omitempty"`
	Flags []Flag `json:"flags,omitempty"`
}

//Flag is a flag that has a new and value and such.
type Flag struct {
	FlagType  string `json:"flag_type,omitempty"`
	Valuename string `json:"valuename,omitempty"`
	Valueint  int32  `json:"valueint,omitempty"`
}

//Network holds pointers to layers and the hidden memory between the layers
type Network struct {
	layer []*layer
	mem   []*layers.IO
}

//Handles holds both handle and xhandle handle
type Handles struct {
	cudnn   *gocudnn.Handle
	xhandle *gocudnn.XHandle
	stream  *gocudnn.Stream
}

//Cudnn returns a pointer to the cudnn handle
func (h *Handles) Cudnn() *gocudnn.Handle {
	return h.cudnn
}

//XHandle returns a pointer to the XHandle
func (h *Handles) XHandle() *gocudnn.XHandle {
	return h.xhandle
}

//CreateHandle creates a multi use handle that uses both gocudnn handle and xhandle
func CreateHandle(dev *gocudnn.Device, xtrakernsfolder string) *Handles {

	x := gocudnn.NewHandle()
	y, err := gocudnn.Xtra{}.MakeXHandle(xtrakernsfolder, dev)
	if err != nil {
		panic(err)
	}
	return &Handles{
		cudnn:   x,
		xhandle: y,
	}
}

//SetStream sets the stream for the handles
func (h *Handles) SetStream(stream *gocudnn.Stream) error {
	err := h.cudnn.SetStream(stream)
	if err != nil {
		return err
	}
	err = h.xhandle.SetStream(stream)
	if err != nil {
		return err
	}
	h.stream = stream
	return nil
}

//CreateNetwork creates an empty Netowrk
func CreateNetwork() *Network {
	return &Network{}
}

//AddLayer adds a layer without setting the mem
func (m *Network) AddLayer(layer interface{}) error {
	l := wraplayer(layer)
	if l == nil {
		return errors.New("Not a supported layer")
	}
	m.layer = append(m.layer, l)
	return nil
}

//BuildHiddenSameSize will build a hidden network layer based on the input given used for my generator network
func (m *Network) BuildHiddenSameSize(handle *Handles, input *layers.IO) error {
	if len(m.mem) > 0 || m.mem != nil {
		return errors.New("Mem Already Set")
	}
	for i := 0; i < len(m.layer)-1; i++ {
		mem, err := input.ZeroClone(handle.cudnn)
		if err != nil {
			for j := 0; j < len(m.mem); j++ {
				m.mem[j].Destroy()
			}
			return err
		}
		m.mem = append(m.mem, mem)

	}
	return nil
}

//BuildHiddenDynamicSize will build a hidden network layer based on the input given used for my generator network
func (m *Network) BuildHiddenDynamicSize(handle *Handles, input *layers.IO) error {
	if len(m.mem) > 0 || m.mem != nil {
		return errors.New("Mem Already Set")
	}

	var previous *layers.IO
	previous = input
	for i := 1; i < len(m.layer)-1; i++ {
		mem, err := m.layer[i].getoutput(handle, previous)
		if err != nil {
			for j := 0; j < len(m.mem); j++ {
				m.mem[j].Destroy()
			}
			return err
		}
		previous = mem
		m.mem = append(m.mem, mem)

	}
	return nil
}
func (m *Network) freehiddenlayers() error {
	var flag bool
	for i := 0; i < len(m.mem); i++ {
		err := m.mem[i].Destroy()
		if err != nil {
			flag = true
		}
	}
	if flag == true {
		return errors.New("Not all mem destroyed")
	}
	m.mem = nil
	return nil
}

/*
//AddLayerandOutput appends an output and a layer to the slices of output and layer in the network.
func (m *Network) AddLayerandOutput(layer interface{}, output *layers.IO) error {
	l := wraplayer(layer)
	if l == nil {
		return errors.New("Not a supported layer")
	}
	m.layer = append(m.layer, l)
	m.mem = append(m.mem, output)
	return nil
}
*/

//ForwardProp does the forward prop for a prebuilt Network
func (m *Network) ForwardProp(handle *Handles, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	err := m.BuildHiddenDynamicSize(handle, x)
	if err != nil {
		return err
	}
	return m.forwardprop(handle, wspace, x, y)

}

//BackProp does the backprop of the hidden layers
func (m *Network) BackProp(handle *Handles, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	err := m.backprop(handle, wspace, x, y)
	if err != nil {
		return err
	}
	return m.freehiddenlayers()

}

func (m *Network) forwardprop(handle *Handles, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	err := handle.stream.Sync()
	if err != nil {
		return err
	}
	err = m.layer[0].forwardprop(handle, wspace, x, m.mem[0])
	if err != nil {
		return err
	}
	lnum := len(m.layer)
	for i := 1; i < lnum-1; i++ {
		err = m.layer[i].forwardprop(handle, wspace, m.mem[i-1], m.mem[i])
		if err != nil {
			return err
		}
	}
	//for m.layer[lnum-1] is the last layer in the slice and m.mem[lnum-2] should be the output from the layer before.
	err = m.layer[lnum-1].forwardprop(handle, wspace, m.mem[lnum-2], y)
	if err != nil {
		return err
	}

	return handle.stream.Sync()
}

//BackProp does the backprop of a Network
func (m *Network) backprop(handle *Handles, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	err := handle.stream.Sync()
	if err != nil {
		return err
	}
	lnum := len(m.layer)
	err = m.layer[lnum-1].backprop(handle, wspace, m.mem[lnum-2], y)

	if err != nil {
		return err
	}

	for i := lnum - 2; i > 1; i-- {
		err = m.layer[i].backprop(handle, wspace, m.mem[i-1], m.mem[i])
		if err != nil {
			return err
		}
	}

	err = m.layer[0].backprop(handle, wspace, x, m.mem[0])
	if err != nil {
		return err
	}

	return handle.stream.Sync()
}

//UpdateWeights updates the weights of a Network
func (m *Network) UpdateWeights(handle *Handles, batch int) error {
	for i := 0; i < len(m.layer); i++ {
		err := m.layer[i].updateWeights(handle, batch)
		if err != nil {
			return err
		}
	}
	return handle.stream.Sync()
}
