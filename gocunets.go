package gocunets

//TODO:  Take SoftMax out of here.  It should in another section for calculating the errors of the network.

import (
	"errors"
	"strconv"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/layers/reshape"
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
	layer         []*layer
	mem           []*layers.IO
	err           chan error
	position      int
	hiomode       hiddenmode
	resizecounter int
	hybridsize    int
	reshaper      *reshape.Layer
	//	originalinput *layers.IO //original input that might have to be held onto so that errors can backprop back through it
	resizeinput   *layers.IO //resized input that will be used to forward propagate.  It will have to be deleted after back propigation
	previousdims  []int32
	descriminator bool
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

/*
//CreateHandleV2 creates a multi use handle that uses both gocudnn handle and xhandle
func CreateHandleV2(dev *gocudnn.Device) *Handles {

	x := gocudnn.NewHandle()
	y, err := gocudnn.Xtra{}.MakeXHandleV2(dev)
	if err != nil {
		panic(err)
	}
	return &Handles{
		cudnn:   x,
		xhandle: y,
	}
}
*/
/*
//CreateHandle creates a handle
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
*/
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
		err:          err,
		previousdims: []int32{-1, -1, -1, -1}, //this initalizes the previous dims to be something that they would never be. and that is negative
	}
}

//SetDescriminatorFlag - Sets the network up as a descriminator network
//This will require the network to have two outputs if using the softmax output
func (m *Network) SetDescriminatorFlag() {
	m.descriminator = true
}

//UnSetDescriminator This will turn the network descriminator flag off
func (m *Network) UnSetDescriminator() {
	m.descriminator = false
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
func (m *Network) DynamicHidden() error {
	m.hiomode = hiddenmode(0)
	return nil
}

//StaticHidden changes the IO between layers to be set in place //Good performance not as flexable
func (m *Network) StaticHidden(handle *cudnn.Handler) error {
	var flg reshape.ModeFlag
	var err error
	m.reshaper, err = reshape.Build(handle, flg.Resize(), nil, false)
	if err != nil {
		return err
	}
	m.hiomode = hiddenmode(1)
	return nil
}

//HybridHidden does a static size for a certain count passed and will change the size maybe randomly (probably going to be to the first image it comes accross).
func (m *Network) HybridHidden(handle *cudnn.Handler, hybridsize int) error {
	var flg reshape.ModeFlag
	var err error
	m.reshaper, err = reshape.Build(handle, flg.Resize(), nil, false)
	if err != nil {
		return err
	}
	m.hybridsize = hybridsize
	m.hiomode = hiddenmode(2)
	return nil
}

//LoadTrainers will load the trainers in the order that the layers were placed in the network
func (m *Network) LoadTrainers(handle *cudnn.Handler, trainerweights, trainerbias []trainer.Trainer) {
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

//buildhiddenios will build a hidden network layer based on the input given used for my generator network
func (m *Network) buildhiddenios(handle *cudnn.Handler, input *layers.IO) error {
	if len(m.mem) > 0 || m.mem != nil {
		return errors.New("Mem Already Set")
	}

	var previous *layers.IO
	previous = input
	for i := 0; i < len(m.layer)-1; i++ {
		//	fmt.Println("GEtoutput from ", i)
		mem, err := m.layer[i].getoutput(handle, previous)
		if err != nil {
			for j := 0; j < len(m.mem); j++ {
				m.mem[j].Destroy()
			}

			return wraperror("getoutputio index: "+strconv.Itoa(i)+" :", err)
		}
		previous = mem
		m.mem = append(m.mem, mem)

	}
	return nil
}
func (m *Network) freehiddenios() error {
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

//ForwardProp does the forward prop for a prebuilt Network
func (m *Network) ForwardProp(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	switch m.hiomode {
	case dynamichiddenio:
		_, _, xdims, err := x.Properties()
		if err != nil {
			return err
		}
		if comparedims(m.previousdims, xdims) == true {
			return m.forwardprop(handle, wspace, x, y)
		}
		m.previousdims = xdims
		err = m.freehiddenios()
		if err != nil {
			return err
		}
		err = m.buildhiddenios(handle, x)
		if err != nil {
			return wraperror("dynamichiddenio", err)
		}
		return m.forwardprop(handle, wspace, x, y)
	case statichiddenio:
		if m.resizecounter == 0 {
			m.resizecounter++
			err := m.freehiddenios()
			if err != nil {
				return wraperror("free in statichiddenio", err)
			}
			err = m.buildhiddenios(handle, x)
			if err != nil {
				return err
			}

			m.resizeinput, err = x.ZeroClone()
			if err != nil {
				m.resizeinput.Destroy()
				return err
			}
			return m.forwardprop(handle, wspace, x, y)
		}
		_, _, dimsx, err := x.Properties()
		if err != nil {
			return err
		}
		_, _, dimsre, err := m.resizeinput.Properties()
		if err != nil {
			return err
		}
		if comparedims(dimsx, dimsre) == true {
			return m.forwardprop(handle, wspace, x, y)
		}
		m.reshaper.ForwardProp(handle, x, m.resizeinput)
		return m.forwardprop(handle, wspace, m.resizeinput, y)
	case hibridhiddenio:
		m.resizecounter++
		if m.resizecounter >= m.hybridsize {
			m.resizecounter = 0
			err := m.freehiddenios()
			if err != nil {
				return err
			}
			err = m.buildhiddenios(handle, x)
			if err != nil {
				return err
			}
			if m.resizeinput != nil {
				err = m.resizeinput.Destroy()
				if err != nil {
					return err
				}
			}
			m.resizeinput, err = x.ZeroClone()
			if err != nil {
				return err
			}
			return m.forwardprop(handle, wspace, x, y)
		}
		_, _, dimsx, err := x.Properties()
		if err != nil {
			return err
		}
		_, _, dimsre, err := m.resizeinput.Properties()
		if err != nil {
			return err
		}
		if comparedims(dimsx, dimsre) == true {
			return m.forwardprop(handle, wspace, x, y)
		}
		m.reshaper.ForwardProp(handle, x, m.resizeinput)
		return m.forwardprop(handle, wspace, m.resizeinput, y)

	}
	return errors.New("ForwardProp-Unsupported Hidden Mode")

}
func comparedims(x, y []int32) bool {
	if len(x) != len(y) {
		return false
	}
	for i := 0; i < len(x); i++ {
		if x[i] != y[i] {
			return false
		}
	}
	return true
}

//BackPropFilterData does the backprop of the hidden layers
func (m *Network) BackPropFilterData(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {

	switch m.hiomode {
	case dynamichiddenio:
		return m.backpropfilterdata(handle, wspace, x, y)

	case statichiddenio:
		_, _, dimsx, err := x.Properties()
		if err != nil {
			return err
		}
		_, _, dimsre, err := m.resizeinput.Properties()
		if err != nil {
			return err
		}
		if comparedims(dimsx, dimsre) == true {
			return m.backpropfilterdata(handle, wspace, x, y)
		}

		err = m.backpropfilterdata(handle, wspace, m.resizeinput, y)
		if err != nil {
			return err
		}
		return m.reshaper.BackProp(handle, x, m.resizeinput)
	case hibridhiddenio:
		_, _, dimsx, err := x.Properties()
		if err != nil {
			return err
		}
		_, _, dimsre, err := m.resizeinput.Properties()
		if err != nil {
			return err
		}
		if comparedims(dimsx, dimsre) == true {
			return m.backpropfilterdata(handle, wspace, x, y)
		}

		err = m.backpropfilterdata(handle, wspace, m.resizeinput, y)
		if err != nil {
			return err
		}
		return m.reshaper.BackProp(handle, x, m.resizeinput)
	}
	return errors.New("BackProp-Unsupported Hidden Mode")

}

//BackPropData only does the data backprop
func (m *Network) BackPropData(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {

	switch m.hiomode {
	case dynamichiddenio:
		return m.backpropdata(handle, wspace, x, y)

	case statichiddenio:
		_, _, dimsx, err := x.Properties()
		if err != nil {
			return err
		}
		_, _, dimsre, err := m.resizeinput.Properties()
		if err != nil {
			return err
		}
		if comparedims(dimsx, dimsre) == true {
			return m.backpropdata(handle, wspace, x, y)
		}

		err = m.backpropdata(handle, wspace, m.resizeinput, y)
		if err != nil {
			return err
		}
		return m.reshaper.BackProp(handle, x, m.resizeinput)
	case hibridhiddenio:
		_, _, dimsx, err := x.Properties()
		if err != nil {
			return err
		}
		_, _, dimsre, err := m.resizeinput.Properties()
		if err != nil {
			return err
		}
		if comparedims(dimsx, dimsre) == true {
			return m.backpropdata(handle, wspace, x, y)
		}

		err = m.backpropdata(handle, wspace, m.resizeinput, y)
		if err != nil {
			return err
		}
		return m.reshaper.BackProp(handle, x, m.resizeinput)
	}
	return errors.New("BackProp-Unsupported Hidden Mode")

}
func wraperror(comment string, err error) error {
	return errors.New(comment + "-" + err.Error())
}
func (m *Network) forwardprop(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	var err error

	err = m.layer[0].forwardprop(handle, wspace, x, m.mem[0])
	if err != nil {
		return wraperror("forward index:"+strconv.Itoa(0), err)
	}
	lnum := len(m.layer)
	for i := 1; i < lnum-1; i++ {
		err = m.layer[i].forwardprop(handle, wspace, m.mem[i-1], m.mem[i])
		if err != nil {
			return wraperror("forward index:"+strconv.Itoa(i), err)
		}
	}

	err = m.layer[lnum-1].forwardprop(handle, wspace, m.mem[lnum-2], y)
	if err != nil {
		return wraperror("forward index:"+strconv.Itoa(lnum-1), err)
	}
	return nil

}
func (m *Network) backpropdata(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	var err error
	//	err := handle.stream.Sync()
	if err != nil {
		return err
	}
	lnum := len(m.layer)
	err = m.layer[lnum-1].backpropdata(handle, wspace, m.mem[lnum-2], y)

	if err != nil {
		return err
	}

	for i := lnum - 2; i > 0; i-- {
		err = m.layer[i].backpropdata(handle, wspace, m.mem[i-1], m.mem[i])
		if err != nil {
			return err
		}
	}

	err = m.layer[0].backpropdata(handle, wspace, x, m.mem[0])
	if err != nil {
		return err
	}
	return nil
	//	return handle.stream.Sync()
}

//BackProp does the backprop of a Network
func (m *Network) backpropfilterdata(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	var err error
	//	err := handle.stream.Sync()
	if err != nil {
		return err
	}
	lnum := len(m.layer)
	err = m.layer[lnum-1].backpropfilterdata(handle, wspace, m.mem[lnum-2], y)

	if err != nil {
		return err
	}

	for i := lnum - 2; i > 0; i-- {
		//	fmt.Println("index", i)
		err = m.layer[i].backpropfilterdata(handle, wspace, m.mem[i-1], m.mem[i])
		if err != nil {
			return err
		}
	}

	err = m.layer[0].backpropfilterdata(handle, wspace, x, m.mem[0])
	if err != nil {
		return err
	}
	return nil
	//	return handle.stream.Sync()
}

//UpdateWeights updates the weights of a Network
func (m *Network) UpdateWeights(handle *cudnn.Handler, batch int) error {
	err := handle.SyncContext()
	if err != nil {
		return err
	}
	for i := 0; i < len(m.layer); i++ {

		err = m.layer[i].updateWeights(handle, batch)
		if err != nil {
			return err
		}
	}
	return handle.SyncContext()
}
