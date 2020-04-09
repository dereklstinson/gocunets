package gocunets

import (
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/loss"
)

//LossLayer performs two functions. Be able to calculate loss, and to be able to calculate the inference forward.
type LossLayer interface {
	PerformError(x, dx, y, dy *layers.Tensor) (err error)
	Inference(x, y *layers.Tensor) (err error)
	TestForward(x, y, target *layers.Tensor) (err error)
	GetAverageBatchLoss() float32
}

//ClassifierModule is used to classify outputs
type ClassifierModule struct {
	id           int64
	b            *Builder
	l            LossLayer
	x, dx, y, dy *Tensor
}

//CreateCustomLossLayer creates a module that uses l
func CreateCustomLossLayer(id int64, b *Builder, l LossLayer) (m *ClassifierModule) {
	m = new(ClassifierModule)
	m.id = id
	m.b = b
	m.l = l
	return m
}

//ID returns the id set for the module
func (m *ClassifierModule) ID() int64 { return m.id }

//CreateSoftMaxClassifier will create a simple module with each of the convolution layers being in parallel.
func CreateSoftMaxClassifier(id int64, bldr *Builder, x, dx, y, target *Tensor) (m *ClassifierModule, err error) {
	m = new(ClassifierModule)
	m.b = bldr
	m.x = x
	m.y = y
	m.dx = dx
	m.dy = target
	m.l, err = loss.CreateSoftMax(bldr.h.Handler)
	if err != nil {
		return nil, err
	}

	return m, nil
	//return createOutputModule(bldr,batch,inputchannel,nsp)
}

//PerformError does the output and error calculation of the previous layer of the network
func (m *ClassifierModule) PerformError() error {
	return m.l.PerformError(m.x.Tensor, m.dx.Tensor, m.y.Tensor, m.dy.Tensor)
}

//Inference does a forward propagation without calculating errors
func (m *ClassifierModule) Inference() error {
	return m.l.Inference(m.x.Tensor, m.y.Tensor)
}

//TestForward does the testforward so that loss can be seen
func (m *ClassifierModule) TestForward() error {
	return m.l.TestForward(m.x.Tensor, m.y.Tensor, m.dy.Tensor)
}

//GetAverageBatchLoss gets the average batch loss
func (m *ClassifierModule) GetAverageBatchLoss() float32 {
	return m.l.GetAverageBatchLoss()
}

//CreateMSEClassifier sets the mean squared error classifier
func CreateMSEClassifier(id int64, bldr *Builder, x, dx, target *Tensor) (m *ClassifierModule, err error) {
	m = new(ClassifierModule)
	m.b = bldr
	m.l, err = loss.CreateMSE2(m.b.h.Handler, target.Tensor)
	return m, err
}

//GetTensorX returns set x tensor
func (m *ClassifierModule) GetTensorX() (x *Tensor) {
	return m.x
}

//GetTensorDX returns set dx tensor
func (m *ClassifierModule) GetTensorDX() (dx *Tensor) {
	return m.dx
}

//GetTensorY returns set y tensor
func (m *ClassifierModule) GetTensorY() (y *Tensor) {
	return m.y
}

//GetTensorDY returns set dy tensor
func (m *ClassifierModule) GetTensorDY() (dy *Tensor) {
	return m.dy
}

//SetTensorX sets x tensor
func (m *ClassifierModule) SetTensorX(x *Tensor) {
	m.x = x
}

//SetTensorDX sets dx tensor
func (m *ClassifierModule) SetTensorDX(dx *Tensor) {
	m.dx = dx
}

//SetTensorY sets y tensor
func (m *ClassifierModule) SetTensorY(y *Tensor) {
	m.y = y
}

//SetTensorDY sets dy tensor
func (m *ClassifierModule) SetTensorDY(dy *Tensor) {
	m.dy = dy
}
