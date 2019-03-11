package gocunets

/*
TODO:  1)  Take SoftMax out of here.  It should in another section for calculating the errors of the network.
	   2)  Make a more "robust" hidden IO system.  With the inclusion of multiple sets of different sized inputs with hidden io equivilant.
			   And also the ability for the neural network to dynamically change (at least the input layer). The slide and padding to help fit already existing sizes.


			 		  --Thoughts while writing--
			   		  **Probably need to make a new struct of ios that can be pooled with a flag of in use.
					   eg  type hiddleio struct{
						   io layers.IO
						   inuse bool
						   timesused int64
					   }.
					   **Put these in a pool to use.
					  **Convolution will need a rotating set of descriptors.
					  **Need to set a limit on the size of pool.
					  **With this implementation will need to have multiple batch norm layers that can be swapped out due to the nature of it.



*/

import (
	"errors"
	"fmt"
	"strconv"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/layers/reshape"
	"github.com/dereklstinson/GoCuNets/trainer"
	"github.com/dereklstinson/GoCuNets/trainer/pso"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

const debuggingmaingocunets = true

//Settings are the setttings for the topology of the network
type Settings struct {
	Format gocudnn.TensorFormat `json:"format,omitempty"`
	DType  gocudnn.DataType     `json:"d_type,omitempty"`
	Manage bool                 `json:"manage,omitempty"`
	Layers []Layer              `json:"layers,omitempty"`
}

//Layer is a layer that has the name the dims and what flags it needs
type Layer struct {
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
	layer             []*layer
	mem               []*layers.IO
	totalionets       []*netios
	totalionetcounter int
	err               chan error
	position          int
	resizecounter     int
	hybridsize        int
	reshaper          *reshape.Layer
	previousdims      []int32
	descriminator     bool
	l1losses          []float32
	l2losses          []float32
	totalionetsinit   bool
	wtrainers         []trainer.Trainer
	btrainers         []trainer.Trainer
	savedparams       *NetworkSavedTensor
	loadedsaved       bool
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
		err:          err,
		previousdims: []int32{-1, -1, -1, -1}, //this initalizes the previous dims to be something that they would never be. and that is negative
	}
}

//Initialize initializes the IO between the hidden layers. It also returns some performance meterics that you can choose to increase the speed of the network at the cost of memory.
func (m *Network) Initialize(handle *cudnn.Handler, input, output *layers.IO, workspace *nvidia.Malloced) ([]ConvolutionPerformance, error) {
	m.previousdims = input.T().Dims()
	err := m.buildhiddenios(handle, input)
	if err != nil {
		fmt.Println("Error in buildinghiddenios")
		return nil, err
	}
	err = handle.DeviceSync()
	if err != nil {
		return nil, err
	}
	return m.performance(handle, input, output, workspace)
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
	m.l1losses, m.l2losses = make([]float32, counter), make([]float32, counter)
	return counter
}

//LoadTrainers will load the trainers in the order that the layers were placed in the network
func (m *Network) LoadTrainers(handle *cudnn.Handler, trainerweights, trainerbias []trainer.Trainer) error {
	if len(trainerweights) != len(trainerbias) {
		return errors.New("(*Network)LoadTrainers -- Sizes Don't Match with trainers and bias")
	}
	if len(trainerweights) != m.TrainersNeeded() {
		return errors.New("(*Network)LoadTrainers -- TrainersNeeded don't match the length of trainers passed")
	}
	counter := 0
	var err error
	for i := 0; i < len(m.layer); i++ {
		if debuggingmaingocunets {
			//	fmt.Println("Going Through Layer at Index", i)
		}
		if m.layer[i].needstrainer() == true {

			if debuggingmaingocunets {
				//	fmt.Println("Loading Trainer at Index", i)
			}
			m.wtrainers = append(m.wtrainers, trainerweights[counter])
			m.btrainers = append(m.btrainers, trainerbias[counter])

			err = m.layer[i].loadtrainer(handle, trainerweights[counter], trainerbias[counter])
			if err != nil {
				panic(err)
			}
			counter++
		}
	}
	return nil
}
func (m *Network) totalnumofscalars() int {
	adder := 0
	for i := range m.layer {
		adder += m.layer[i].scalarnum
	}
	return adder
}

//GetTrainers returns the trainers for the network.  ...convienence function
func (m *Network) GetTrainers() (weights, bias []trainer.Trainer) {
	return m.wtrainers, m.btrainers
}

//ScalarOptimizer optimizes the scalars of the operators
type ScalarOptimizer struct {
	hasscalars     []*layer
	pso            pso.Swarm64
	index          int
	numofparticles int
}

func (m *Network) initializeslphascalarstuff() ([]*layer, int) {
	adder := 0
	layers := make([]*layer, 0)
	for i := range m.layer {
		x := m.layer[i].initalphascalarsamount()
		if x > 0 {
			layers = append(layers, m.layer[i])
		}
		adder += x
	}
	return layers, adder
}
func (m *Network) initializebetascalarstuff() ([]*layer, int) {
	adder := 0
	layers := make([]*layer, 0)
	for i := range m.layer {
		x := m.layer[i].initbetascalarsamount()
		if x > 0 {
			layers = append(layers, m.layer[i])
		}
		adder += x
	}
	return layers, adder
}

//SetupScalarPSO returns a pso to optimize the scalars in the network
func SetupScalarAlphaPSO(mode pso.Mode, numofparticles, seed, kmax int, cognative, social, vmax, minstartposition, maxstartposition, alphamax, inertiamax float64, x ...*Network) ScalarOptimizer {
	hasscalars := make([]*layer, 0)
	totalscalars := 0
	for i := range x {
		for _, layer := range x[i].layer {
			amount := layer.initalphascalarsamount()

			if amount != 0 {
				hasscalars = append(hasscalars, layer)
				totalscalars += amount
			}

		}
	}
	swarm := pso.CreateSwarm64(mode, numofparticles, totalscalars, seed, kmax, cognative, social, vmax, minstartposition, maxstartposition, alphamax, inertiamax)
	position := swarm.GetParticlePosition(0)

	for i := range hasscalars {

		position = hasscalars[i].updatealphascalar(position)
	}
	return ScalarOptimizer{
		hasscalars: hasscalars,
		pso:        swarm,
	}
}

//SetupScalarPSO returns a pso to optimize the scalars in the network
func SetupScalarBetaPSO(mode pso.Mode, numofparticles, seed, kmax int, cognative, social, vmax, minstartposition, maxstartposition, alphamax, inertiamax float64, x ...*Network) ScalarOptimizer {
	hasscalars := make([]*layer, 0)
	totalscalars := 0
	for i := range x {
		for _, layer := range x[i].layer {
			amount := layer.initbetascalarsamount()

			if amount != 0 {
				hasscalars = append(hasscalars, layer)
				totalscalars += amount
			}

		}
	}
	swarm := pso.CreateSwarm64(mode, numofparticles, totalscalars, seed, kmax, cognative, social, vmax, minstartposition, maxstartposition, alphamax, inertiamax)
	position := swarm.GetParticlePosition(0)

	for i := range hasscalars {

		position = hasscalars[i].updateabetascalar(position)
	}
	return ScalarOptimizer{
		hasscalars: hasscalars,
		pso:        swarm,
	}
}

//AsyncUpdating updates the Swarm after each particle use
func (m *ScalarOptimizer) AsyncUpdatingBeta(fitness float32) error {

	err := m.pso.AsyncUpdate(m.index, float64(fitness))
	if err != nil {
		return err
	}
	if m.index < m.numofparticles-1 {
		m.index++
	} else {
		m.index = 0
	}

	position := m.pso.GetParticlePosition(m.index)
	for i := range m.hasscalars {
		position = m.hasscalars[i].updateabetascalar(position)

	}
	return nil
}

//AsyncUpdating updates the Swarm after each particle use
func (m *ScalarOptimizer) AsyncUpdatingAlpha(fitness float32) error {

	err := m.pso.AsyncUpdate(m.index, float64(fitness))
	if err != nil {
		return err
	}
	if m.index < m.numofparticles-1 {
		m.index++
	} else {
		m.index = 0
	}

	position := m.pso.GetParticlePosition(m.index)
	for i := range m.hasscalars {
		position = m.hasscalars[i].updatealphascalar(position)

	}
	return nil
}

//MetaOptimizer uses a PSO to optimize meta values
type MetaOptimizer struct {
	trainers       []trainer.Trainer
	pso            pso.Swarm
	index          int
	numofparticles int
}

//AsyncUpdating updates the Swarm after each particle use
func (m *MetaOptimizer) AsyncUpdating(fitness float32) error {
	err := m.pso.AsyncUpdate(m.index, fitness)
	if err != nil {
		return err
	}
	if m.index < m.numofparticles-1 {
		m.index++
	} else {
		m.index = 0
	}

	pctr := 0
	position := m.pso.GetParticlePosition(m.index)
	for i := range m.trainers {
		m.trainers[i].SetRate(position[pctr])
		m.trainers[i].SetDecays(position[pctr+1], position[pctr+2])
		pctr = pctr + 3
	}
	return nil
}

//SetUpPSO will set up the pso
func SetUpPSO(mode pso.Mode, numofparticles, seed, kmax int, cognative, social, vmax, minstartposition, maxstartposition, alphamax, inertiamax float32, x ...[]trainer.Trainer) MetaOptimizer {

	trainers := make([]trainer.Trainer, 0)
	for i := range x {
		trainers = append(trainers, x[i]...)
	}
	totaldims := len(trainers) * 3
	swarm := pso.CreateSwarm(mode, numofparticles, totaldims, seed, kmax, cognative, social, vmax, minstartposition, maxstartposition, alphamax, inertiamax)
	position := swarm.GetParticlePosition(0)
	pctr := 0
	for i := range trainers {
		trainers[i].SetRate(position[pctr])
		trainers[i].SetDecays(position[pctr+1], position[pctr+2])
		pctr = pctr + 3
	}
	return MetaOptimizer{
		trainers:       trainers,
		pso:            swarm,
		numofparticles: numofparticles,
	}
}

//AddLayers will take a list of layers and shove them into the network
func (m *Network) AddLayers(layer ...interface{}) {
	for i := range layer {
		switch x := layer[i].(type) {
		case error:
			if x != nil {
				m.err <- x
			}
		default:
			l, hasweights := wraplayer(x)
			if l == nil {

				m.err <- errors.New("Unsupported Layer")
			}
			if hasweights {
				m.totalionetcounter += 2
			} else {
				m.totalionetcounter++
			}
			m.layer = append(m.layer, l)
		}

	}

}

//AddLayer adds a layer without setting the mem
func (m *Network) AddLayer(layer interface{}, err error) {
	if err != nil {
		panic(err)
		//	m.err <- err
	}
	l, hasweights := wraplayer(layer)
	if l != nil {
		m.layer = append(m.layer, l)
		if hasweights {
			m.totalionetcounter += 2

		} else {
			m.totalionetcounter++
		}
		return
	}
	m.err <- errors.New("Unsupported Layer")
	return
}

//HiddenIOandWeightCount will return the number of hidden ios and weights that are not exposed to other networks
func (m *Network) HiddenIOandWeightCount() int {
	return m.totalionetcounter - 1
}

//buildhiddenios will build a hidden network layer based on the input given used for my generator network
func (m *Network) buildhiddenios(handle *cudnn.Handler, input *layers.IO) error {
	if len(m.mem) > 0 || m.mem != nil {
		return errors.New("Mem Already Set")
	}
	m.totalionets = make([]*netios, 0)
	var previous *layers.IO
	previous = input
	for i := 0; i < len(m.layer)-1; i++ {
		layerwbs := wrapnetio(m.layer[i])
		if layerwbs != nil {
			m.totalionets = append(m.totalionets, layerwbs)
		}
		mem, err := m.layer[i].getoutput(handle, previous)
		if err != nil {

			fmt.Println("error in get output")
			return wraperror("getoutputio index: "+strconv.Itoa(i)+" :", err)
		}
		netiomem := wrapnetio(mem)
		netiomem.name = m.layer[i].name + "-Output"
		m.totalionets = append(m.totalionets, netiomem)
		previous = mem
		m.mem = append(m.mem, mem)
	}
	layerwbs := wrapnetio(m.layer[len(m.layer)-1])
	if layerwbs != nil {
		m.totalionets = append(m.totalionets, layerwbs)
	}
	if len(m.totalionets) != m.totalionetcounter-1 {
		fmt.Println(len(m.totalionets), m.totalionetcounter-1)
		panic("len(m.totalionets)!= m.totalionetcounter-1  please fix")

	}
	if m.savedparams != nil && !m.loadedsaved {
		err := m.LoadNetworkTensorparams(handle, m.savedparams)
		if err != nil {
			fmt.Println("error in loading saved params")
			return err
		}
		m.loadedsaved = true
	}

	return m.buildminmax(handle)
}
func (m *Network) resizehiddenios(handle *cudnn.Handler, newinput []int32) error {
	var err error
	for i := 0; i < len(m.mem); i++ {
		olddims := m.mem[i].T().Dims()
		newdims := make([]int32, len(olddims))
		copy(newdims, olddims)
		newdims[0] = newinput[0]
		//Since it should only be the batch changing we will just change the batch
		err = m.mem[i].ResizeIO(handle, newdims)
		if err != nil {
			return err
		}

	}
	return nil
}

//ForwardProp does the forward prop for a prebuilt Network
func (m *Network) ForwardProp(handle *cudnn.Handler, wspace *nvidia.Malloced, x, y *layers.IO) error {
	var err error
	if m.mem == nil {

		err = m.buildhiddenios(handle, x)
		if err != nil {
			fmt.Println("Error in building hidden os")
			return err
		}
		m.previousdims = x.T().Dims()
		return m.forwardprop(handle, wspace, x, y)

	}
	_, _, xdims, err := x.Properties()
	if err != nil {
		return err
	}
	if comparedims(m.previousdims, xdims) {
		err = m.forwardprop(handle, wspace, x, y)
		if err != nil {

			fmt.Println("Error in doing the forward prop after compair dims")

			return err
		}
		return nil
	}

	m.previousdims = xdims
	err = m.resizehiddenios(handle, xdims)
	if err != nil {
		fmt.Println("Error in resize hiddenios")
		return err
	}
	err = m.forwardprop(handle, wspace, x, y)
	if err != nil {
		fmt.Println("Error in doing the forward prop after resize")
	}
	err = handle.Sync()
	if err != nil {
		return err
	}
	return nil

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
func (m *Network) BackPropFilterData(handle *cudnn.Handler, datawspace, filterwspace *nvidia.Malloced, x, y *layers.IO) error {
	err := m.backpropfilterdata(handle, datawspace, filterwspace, x, y)
	if err != nil {
		return err
	}
	return handle.Sync()

}

//BackPropData only does the data backprop
func (m *Network) BackPropData(handle *cudnn.Handler, wspace *nvidia.Malloced, x, y *layers.IO) error {
	err := m.backpropdata(handle, wspace, x, y)
	if err != nil {
		return err
	}
	return handle.Sync()

}
func wraperror(comment string, err error) error {
	return errors.New(comment + "-" + err.Error())
}
func (m *Network) forwardprop(handle *cudnn.Handler, wspace *nvidia.Malloced, x, y *layers.IO) error {
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
func (m *Network) inference(handle *cudnn.Handler, wspace *nvidia.Malloced, x, y *layers.IO) error {
	var err error

	err = m.layer[0].inference(handle, wspace, x, m.mem[0])
	if err != nil {
		return wraperror("forward index:"+strconv.Itoa(0), err)
	}
	lnum := len(m.layer)
	for i := 1; i < lnum-1; i++ {

		err = m.layer[i].inference(handle, wspace, m.mem[i-1], m.mem[i])
		if err != nil {
			return wraperror("forward index:"+strconv.Itoa(i), err)
		}
	}

	err = m.layer[lnum-1].inference(handle, wspace, m.mem[lnum-2], y)
	if err != nil {
		return wraperror("forward index:"+strconv.Itoa(lnum-1), err)
	}
	return nil
}
func (m *Network) backpropdata(handle *cudnn.Handler, wspace *nvidia.Malloced, x, y *layers.IO) error {
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
func (m *Network) backpropfilterdata(handle *cudnn.Handler, wspacedata, wspacefilter *nvidia.Malloced, x, y *layers.IO) error {
	var err error
	//	err := handle.stream.Sync()
	if err != nil {
		return err
	}
	lnum := len(m.layer)
	err = m.layer[lnum-1].backpropfilterdata(handle, wspacedata, wspacefilter, m.mem[lnum-2], y)

	if err != nil {
		return err
	}

	for i := lnum - 2; i > 0; i-- {

		err = m.layer[i].backpropfilterdata(handle, wspacedata, wspacefilter, m.mem[i-1], m.mem[i])
		if err != nil {
			return err
		}
	}

	err = m.layer[0].backpropfilterdata(handle, wspacedata, wspacefilter, x, m.mem[0])
	if err != nil {
		return err
	}
	return nil
	//	return handle.stream.Sync()
}

//UpdateWeights updates the weights of a Network
func (m *Network) UpdateWeights(handle *cudnn.Handler, batch int) error {
	var err error
	counter := 0
	for i := 0; i < len(m.layer); i++ {

		err = m.layer[i].updateWeights(handle, batch)
		if err != nil {
			return err
		}
		a, b := m.layer[i].l1l2loss()
		if a > -1 && b > -1 {
			m.l1losses[counter], m.l2losses[counter] = a, b
			counter++
		}

	}
	return nil
}

//TotalL1L2Loss returns the total l1l2 loss of the network
func (m *Network) TotalL1L2Loss() (L1, L2 float32) {
	L1, L2 = 0, 0
	for i := range m.l1losses {
		L1 += m.l1losses[i]
		L2 += m.l2losses[i]
	}

	return L1, L2
}

//L1L2Loss returns the L1L2 loss arrays for every layer that has a trainer
func (m *Network) L1L2Loss() (L1, L2 []float32) {
	return m.l1losses, m.l2losses
}
