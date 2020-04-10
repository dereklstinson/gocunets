package gocunets

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCudnn/cudart"
	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/nccl"
)

//Tensor is contains 2 tensors the x and dx.  Input IOs will contain only the X tensor.
type Tensor struct {
	*layers.Tensor
}

//type Workspace struct {
//	*nvidia.Malloced
//}

////Trainer is a trainer.Trainer
//type Trainer interface {
//	trainer.Trainer
//}

//Handle handles the functions of the libraries used in gocunet
type Handle struct {
	*cudnn.Handler
	w *gocu.Worker
}

//GetWorker returns the gocu.Worker.
func (h *Handle) GetWorker() *gocu.Worker {
	return h.w
}

//Comm is a communicator
type Comm struct {
	c   *nccl.Comm
	h   *Handle
	uid nccl.UniqueID
}

//CreateComms creates Communicators for parallel processes.
func CreateComms(hs []*Handle) (comm []*Comm, err error) {
	uid, err := nccl.GetUniqueID()
	if err != nil {
		return nil, err
	}
	nrank := int32(len(hs))
	comm = make([]*Comm, len(hs))
	for i := range hs {
		err = hs[i].Work(func() error {
			comm[i].c, err = nccl.CommInitRank(nrank, uid, int32(i))
			if err != nil {
				return err
			}
			comm[i].h = hs[i]
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	return comm, err
}

//Stream is a stream for gpu instructions
type Stream struct {
	*cudart.Stream
}

//CreateHandle creates a handle for gocunets
func CreateHandle(w *gocu.Worker, d Device, seed uint64) (h *Handle) {
	h = new(Handle)
	h.Handler = cudnn.CreateHandler(w, d.Device, seed)
	h.w = gocu.NewWorker(d.Device)

	return h
}

//CreateHandles creates parrallel handles.  With there own workers.  It also creates non blocking streams
func CreateHandles(ws []*gocu.Worker, ds []Device, seeds []uint64) []*Handle {

	hs := make([]*Handle, len(ds))
	var err error
	for i := range ds {
		hs[i] = CreateHandle(ws[i], ds[i], seeds[i])
		err = hs[i].Work(func() error {
			var err error
			stream, err := cudart.CreateNonBlockingStream()
			if err != nil {
				panic(err)
			}
			hs[i].SetStream(stream)
			return nil
		})
		if err != nil {
			panic(err)
		}
	}
	return hs
}

//Close closes the work thread
func (h *Handle) Close() {
	h.w.Close()
}

//CreateStream creates a stream
func CreateStream() (s *Stream, err error) {
	s = new(Stream)
	s.Stream, err = cudart.CreateNonBlockingStream()
	return s, err
}

//Device is a gpu device
type Device struct {
	cudart.Device
	num int32
}

//Num is the numerical id of the device
func (d Device) Num() int32 {
	return d.num
}

//GetDeviceList gets a device from a list
func GetDeviceList() (devices []Device, err error) {
	n, err := cudart.GetDeviceCount()
	if err != nil {
		return nil, err
	}
	devices = make([]Device, n)
	for i := (int32)(0); i < n; i++ {
		devices[i].Device = cudart.CreateDevice(i)
		if err != nil {
			return nil, err
		}
		devices[i].num = i
	}
	return devices, nil
}

//func trainerstooriginal(t []Trainer) (x []trainer.Trainer) {
//	x = make([]trainer.Trainer, len(t))
//	for i := range t {
//		x[i] = t[i]
//	}
//	return x
//}

//import (
//	"errors"
//	"math/rand"
//
//	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
//	"github.com/dereklstinson/GoCuNets/layers/reshape"
//	"github.com/dereklstinson/GoCuNets/trainer"
//	gocudnn "github.com/dereklstinson/GoCudnn"
//	//	"strconv"
//)
//
//const debuggingmaingocunets = true
//
////Settings are the setttings for the topology of the network
//type Settings struct {
//	Format gocudnn.TensorFormat `json:"format,omitempty"`
//	DType  gocudnn.DataType     `json:"d_type,omitempty"`
//	Manage bool                 `json:"manage,omitempty"`
//	Layers []LayerInfo          `json:"layers,omitempty"`
//}
//
////LayerInfo is a layer that has the name the dims and what flags it needs
//type LayerInfo struct {
//	Dims  []int  `json:"dims,omitempty"`
//	Flags []Flag `json:"flags,omitempty"`
//}
//
////Flag is a flag that has a new and value and such.
//type Flag struct {
//	FlagType  string `json:"flag_type,omitempty"`
//	Valuename string `json:"valuename,omitempty"`
//	Valueint  int32  `json:"valueint,omitempty"`
//}
//
////Network holds pointers to layers and the hidden memory between the layers
//type Network struct {
//	handle          *Handle
//	dtype           gocudnn.DataType
//	frmt            gocudnn.TensorFormat
//	cmode           gocudnn.ConvolutionMode
//	mathtype        gocudnn.MathType
//	nanprop         gocudnn.NANProp
//	layers          []*Layer
//	usingwsfwd      bool
//	usingwsbwdd     bool
//	usingwsbwdf     bool
//	wsfwd           *nvidia.Malloced
//	wsbwdd          *nvidia.Malloced
//	wsbwdf          *nvidia.Malloced
//	totalionets     []*netios
//	err             chan error
//	position        int
//	reshaper        *reshape.Layer
//	l1losses        []float32
//	l2losses        []float32
//	totalionetsinit bool
//	trainers        []trainer.Trainer
//	savedparams     *NetworkSavedTensor
//	loadedsaved     bool
//	rng             *rand.Rand
//	rngsource       rand.Source
//	idcounter       int64
//}
//
///*
//type hiddenio struct {
//	mem          []*layers.Tensor
//	dmem         []*layers.Tensor
//	previousdims []int32
//}
//*/
//func networkerrors(err <-chan error) {
//	for i := range err {
//		if i != nil {
//			panic(i)
//		}
//	}
//}
//
////CreateNetwork creates an empty Netowrk
//func CreateNetwork(h *Handle) *Network {
//	err := make(chan error, 10)
//	go networkerrors(err)
//
//	return &Network{
//		handle: h,
//		err:    err,
//	}
//}
//
////SetInputandOutputTensors sets the input and output tensors
//func (m *Network) SetInputandOutputTensors(x, dx, y, dy *Tensor) {
//	m.layers[0].x = x
//	m.layers[0].dx = dx
//	m.layers[len(m.layers)-1].y = y
//	m.layers[len(m.layers)-1].dy = dy
//}
//
////SetInputTensorDX sets DX
//func (m *Network) SetInputTensorDX(dx *Tensor) {
//	m.layers[0].dx = dx
//}
//
////SetOutputTensorY sets Y
//func (m *Network) SetOutputTensorY(y *Tensor) {
//	m.layers[len(m.layers)-1].y = y
//}
//
////SetOutputTensorDY sets DY
//func (m *Network) SetOutputTensorDY(dy *Tensor) {
//	m.layers[len(m.layers)-1].dy = dy
//}
//
////SetInputTensorX sets X
//func (m *Network) SetInputTensorX(x *Tensor) {
//	m.layers[0].x = x
//}
//
////GetIndputTensorDX sets DX
//func (m *Network) GetIndputTensorDX() (dx *Tensor) {
//	dx = m.layers[0].dx
//	return dx
//}
//
////GetInputTensorX gets X
//func (m *Network) GetInputTensorX() (x *Tensor) {
//	x = m.layers[0].x
//	return x
//}
//
////GetOutputTensorY gets Y
//func (m *Network) GetOutputTensorY() (y *Tensor) {
//	y = m.layers[len(m.layers)-1].y
//	return y
//}
//
////GetOutputTensorDY gets DY
//func (m *Network) GetOutputTensorDY() (dy *Tensor) {
//	dy = m.layers[len(m.layers)-1].dy
//	return dy
//}
//
////GetOutputTensorDX gets DX
//func (m *Network) GetOutputTensorDX() (dy *Tensor) {
//	dy = m.layers[len(m.layers)-1].dy
//	return dy
//}
//
////ConnectHiddenLayersSimple simply connects the hidden layers.
//func (m *Network) ConnectHiddenLayersSimple(x, dx, y, dy *Tensor, b *Builder) (err error) {
//	m.layers[0].x = x
//	m.layers[0].dx = dx
//	m.layers[len(m.layers)-1].y = y
//	m.layers[len(m.layers)-1].dy = dy
//	for i := 0; i < len(m.layers)-1; i++ {
//		err = b.ConnectLayers(m.layers[i], m.layers[i+1])
//		if err != nil {
//			return err
//		}
//	}
//	return nil
//}
//
////TrainersNeeded returns the number of trainers that are needed.
//func (m *Network) TrainersNeeded() int {
//	var counter int
//	for i := range m.layers {
//
//		counter += m.layers[i].trainersneeded()
//
//	}
//
//	return counter
//}
//
////LoadTrainers will load the trainers in the order that the layers were placed in the network
//func (m *Network) LoadTrainers(handle *Handle, batchsize int, trainers []trainer.Trainer) error {
//
//	if len(trainers) != m.TrainersNeeded() {
//		return errors.New("(*Network)LoadTrainers -- TrainersNeeded don't match the length of trainers passed")
//	}
//	m.trainers = trainers
//	counter := 0
//	traineroffset := 0
//	var err error
//	for i := 0; i < len(m.layers); i++ {
//		if debuggingmaingocunets {
//			//	fmt.Println("Going Through Layer at Index", i)
//		}
//		trainersneeded := m.layers[i].trainersneeded()
//		if trainersneeded > 0 {
//			err = m.layers[i].loadtrainer(handle.Handler, batchsize, m.trainers[traineroffset:traineroffset+trainersneeded]...)
//			if err != nil {
//				panic(err)
//			}
//			counter++
//			traineroffset += trainersneeded
//		}
//	}
//	m.l1losses, m.l2losses = make([]float32, counter), make([]float32, counter)
//	return nil
//}
//
////AddLayer adds a layer without setting the mem
//func (m *Network) AddLayer(layer *Layer) {
//	m.layers = append(m.layers, layer)
//	return
//}
//
//const panicflag = true
//
////AddLayerEx is like AddLayer, but since most functions return errors, this will allow the building of a network to be condensed.
//func (m *Network) AddLayerEx(layer *Layer, err error) error {
//	m.layers = append(m.layers, layer)
//	if panicflag {
//		if err != nil {
//			panic(err)
//		}
//
//	}
//	return err
//}
//
////ForwardProp does the forward prop for a prebuilt Network
//func (m *Network) ForwardProp() error {
//	var err error
//	if m.usingwsfwd && m.wsfwd == nil {
//		return errors.New("forward workspace performance is being used, but actual forward wspace memory has not been set")
//	}
//	if m.usingwsbwdd && m.wsbwdd == nil {
//		return errors.New("set network to use workspace for bwd data, but bwd data wspace is nil")
//	}
//	if m.usingwsbwdf && m.wsbwdf == nil {
//		return errors.New("set network to use workspace for bwd filt, but bwd filt wspace is nil")
//
//	}
//
//	for i := range m.layers {
//		err = m.layers[i].forwardprop()
//		if err != nil {
//			return err
//		}
//	}
//	err = m.handle.Sync()
//	if err != nil {
//		return err
//	}
//	return nil
//
//}
//
//func comparedims(x, y []int32) bool {
//	if len(x) != len(y) {
//		return false
//	}
//	for i := 0; i < len(x); i++ {
//		if x[i] != y[i] {
//			return false
//		}
//	}
//	return true
//}
//
////BackPropFilterData does the backprop of the hidden layers
//func (m *Network) BackPropFilterData() (err error) {
//
//	for i := len(m.layers) - 1; i >= 0; i-- {
//		err = m.layers[i].backpropfilterdata()
//		if err != nil {
//			return err
//		}
//
//	}
//
//	return m.handle.Sync()
//
//}
//
////BackPropData only does the data backprop
//func (m *Network) BackPropData() (err error) {
//	for i := len(m.layers) - 1; i >= 0; i-- {
//		err = m.layers[i].backpropdata()
//		if err != nil {
//			return err
//		}
//	}
//
//	return m.handle.Sync()
//
//}
//
////BackPropFilter does the backprop for the filter
//func (m *Network) BackPropFilter() (err error) {
//	for i := len(m.layers) - 1; i >= 0; i-- {
//		err = m.layers[i].backpropfilter()
//		if err != nil {
//			return err
//		}
//	}
//	return m.handle.Sync()
//
//}
//func wraperror(comment string, err error) error {
//	return errors.New(comment + "-" + err.Error())
//}
//
////ZeroHiddenIOs will zero out the hidden ios. This is used for training the feedback loops for the scalars.
//func (m *Network) ZeroHiddenIOs() (err error) {
//
//	err = m.handle.Sync()
//
//	for i := 0; i < len(m.layers)-1; i++ {
//		if m.layers[i].y != nil {
//			m.layers[i].y.SetAll(0)
//		}
//		if m.layers[i].dy != nil {
//			m.layers[i].dy.SetAll(0)
//		}
//
//	}
//	return m.handle.Sync()
//
//}
//
////UpdateWeights updates the weights of a Network
//func (m *Network) UpdateWeights(epoch int) (err error) {
//
//	counter := 0
//	for i := 0; i < len(m.layers); i++ {
//
//		err = m.layers[i].updateWeights(epoch)
//		if err != nil {
//			return err
//		}
//		a, b := m.layers[i].l1l2loss()
//		if a > -1 && b > -1 {
//			m.l1losses[counter], m.l2losses[counter] = a, b
//			counter++
//		}
//
//	}
//	return nil
//}
//
////TotalL1L2Loss returns the total l1l2 loss of the network
//func (m *Network) TotalL1L2Loss() (L1, L2 float32) {
//	L1, L2 = 0, 0
//	for i := range m.l1losses {
//		L1 += m.l1losses[i]
//		L2 += m.l2losses[i]
//	}
//
//	return L1, L2
//}
//
////L1L2Loss returns the L1L2 loss arrays for every layer that has a trainer
//func (m *Network) L1L2Loss() (L1, L2 []float32) {
//	return m.l1losses, m.l2losses
//}
//
