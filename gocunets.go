package gocunets

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/trainer"
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
	layer       []*layer
	mem         []*layers.IO
	err         chan error
	position    int
	hiomode     hiddenmode
	hybridcount int
	hybridsize  int
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
func networkerrors(err <-chan error) {
	for i := range err {
		if i != nil {
			panic(i)
		}
	}
}

//CreateNetwork creates an empty Netowrk
func CreateNetwork() *Network {
	err := make(chan error, 10)
	go networkerrors(err)
	return &Network{
		err: err,
	}
}

//TrainersNeeded returns the number of trainers that are needed.
func (m *Network) TrainersNeeded() int {
	var counter int
	for i := range m.layer {
		if m.layer[i].needstrainer() == true {
			counter++
		}
	}
	return counter
}

type hiddenmode int

const dynamichiddenio = hiddenmode(0)
const statichiddenio = hiddenmode(1)
const hibridhiddenio = hiddenmode(2)

//DynamicHidden changes the hidden IO between layers to be dynamic  // Bad performance pretty flexable
func (m *Network) DynamicHidden() {
	m.hiomode = hiddenmode(0)
}

//StaticHidden changes the IO between layers to be set in place //Good performance not as flexable
func (m *Network) StaticHidden() {
	m.hiomode = hiddenmode(1)
}

//HybridHidden does a static size for a certain count passed and will change the size maybe randomly.
func (m *Network) HybridHidden(hybridsize int) {
	m.hybridsize = hybridsize
	m.hiomode = hiddenmode(2)
}

//LoadTrainers will load the trainers in the order that the layers were placed in the network
func (m *Network) LoadTrainers(handle *Handles, trainerweights, trainerbias []trainer.Trainer) {
	if len(trainerweights) != len(trainerbias) {
		m.err <- errors.New("(*Network)LoadTrainers -- Sizes Don't Match with trainers and bias")
	}
	if len(trainerweights) != m.TrainersNeeded() {
		m.err <- errors.New("(*Network)LoadTrainers -- TrainersNeeded don't match the length of trainers passed")
	}
	counter := 0
	for i := 0; i < len(m.layer); i++ {
		if m.layer[i].needstrainer() == true {
			m.err <- m.layer[i].loadtrainer(handle, trainerweights[counter], trainerbias[counter])
			counter++
		}
	}
}

//AddLayer adds a layer without setting the mem
func (m *Network) AddLayer(layer interface{}, err error) {
	if err != nil {
		m.err <- err
	}
	l := wraplayer(layer)
	if l == nil {

		m.err <- err
	}
	m.layer = append(m.layer, l)
	return
}

//BuildHiddenSameSize will build a hidden network layer based on the input given used for my generator network
func (m *Network) BuildHiddenSameSize(handle *Handles, input *layers.IO) error {
	if len(m.mem) > 0 || m.mem != nil {
		return errors.New("Mem Already Set")
	}
	for i := 0; i < len(m.layer)-1; i++ {
		mem, err := input.ZeroClone()
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
	for i := 0; i < len(m.layer)-1; i++ {
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
	if m.mem == nil {
		return nil
	}
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

	err := m.freehiddenlayers()
	if err != nil {
		return err
	}
	err = m.BuildHiddenDynamicSize(handle, x)
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
	return nil

}

func (m *Network) forwardprop(handle *Handles, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	var err error

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

	err = m.layer[lnum-1].forwardprop(handle, wspace, m.mem[lnum-2], y)
	if err != nil {
		return err
	}
	return nil

}

//BackProp does the backprop of a Network
func (m *Network) backprop(handle *Handles, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	var err error
	//	err := handle.stream.Sync()
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
	return nil
	//	return handle.stream.Sync()
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
