package gocunets

import (
	act "github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/activation"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//DataType struct wrapper for gocudnn.Datatype.  Look up methods in gocudnn.
type DataType struct {
	gocudnn.DataType
}

//Float sets and returns the Float flag
func (d *DataType) Float() DataType {
	d.DataType.Float()
	return *d
}

//Half sets and returns the Half flag
func (d *DataType) Half() DataType {
	d.DataType.Half()
	return *d
}

//Double sets and returns the Double flag
func (d *DataType) Double() DataType {
	d.DataType.Double()
	return *d
}

//UInt8 sets and returns the Uint8 flag
func (d *DataType) UInt8() DataType {
	d.DataType.UInt8()
	return *d
}

//Int8 sets and returns the Int8 flag
func (d *DataType) Int8() DataType {
	d.DataType.Int8()
	return *d
}

//Int8x32 sets and returns the Int32 flag
func (d *DataType) Int8x32() DataType {
	d.DataType.Int8x32()
	return *d
}

//Int8x4 sets and returns the Int32 flag
func (d *DataType) Int8x4() DataType {
	d.DataType.Int8x4()
	return *d
}

//UInt8x4 sets and returns the Int32 flag
func (d *DataType) UInt8x4() DataType {
	d.DataType.UInt8x4()
	return *d
}

//Int32 sets and returns the Int32 flag
func (d *DataType) Int32() DataType {
	d.DataType.Int32()
	return *d
}

//TensorFormat struct wrapper for gocudnn.TensorFormat.  Look up methods in gocudnn.
type TensorFormat struct {
	gocudnn.TensorFormat
}

//NCHW sets and returns the NCHW flag
func (t *TensorFormat) NCHW() TensorFormat {
	t.TensorFormat.NCHW()
	return *t
}

//NHWC sets and returns the NHWC flag
func (t *TensorFormat) NHWC() TensorFormat {
	t.TensorFormat.NHWC()
	return *t
}

//NCHWvectC sets and returns the NHWC flag
func (t *TensorFormat) NCHWvectC() TensorFormat {
	t.TensorFormat.NCHWvectC()
	return *t
}

//Strided sets and returns the NHWC flag
func (t *TensorFormat) Strided() TensorFormat {
	t.TensorFormat.Strided()
	return *t
}

//ConvolutionMode struct wrapper for gocudnn.ConvolutionMode.  Look up methods in gocudnn.
type ConvolutionMode struct {
	gocudnn.ConvolutionMode
}

//Convolution sets and returns the Convolution flag
func (c *ConvolutionMode) Convolution() ConvolutionMode {
	c.ConvolutionMode.Convolution()
	return *c
}

//CrossCorrelation sets and returns the CrossCorrelation flag
func (c *ConvolutionMode) CrossCorrelation() ConvolutionMode {
	c.ConvolutionMode.CrossCorrelation()
	return *c
}

//NanProp struct wrapper for gocudnn.NanProp.  Look up methods in gocudnn.
type NanProp struct {
	gocudnn.NANProp
}

//Propigate sets and returns the Propigate flag
func (n *NanProp) Propigate() NanProp {
	n.NANProp.Propigate()
	return *n
}

//NotPropigate sets and returns the NotPropigate flag
func (n *NanProp) NotPropigate() NanProp {
	n.NANProp.NotPropigate()
	return *n
}

//BatchNormMode struct wrapper for gocudnn.BatchNormMode.  Look up methods in gocudnn.
type BatchNormMode struct {
	gocudnn.BatchNormMode
}

//PerActivation sets and returns the PerActivation flag
func (b *BatchNormMode) PerActivation() BatchNormMode {
	b.BatchNormMode.PerActivation()
	return *b
}

//Spatial sets and returns the Spatial flag
func (b *BatchNormMode) Spatial() BatchNormMode {
	b.BatchNormMode.Spatial()
	return *b
}

//SpatialPersistent sets and returns the SpatialPersistent flag
func (b *BatchNormMode) SpatialPersistent() BatchNormMode {
	b.BatchNormMode.SpatialPersistent()
	return *b
}

//BatchNormOps struct wrapper for gocudnn.BatchNormOps.  Look up methods in gocudnn.
type BatchNormOps struct {
	gocudnn.BatchNormOps
}

//Activation sets and returns the Activation flag
func (b *BatchNormOps) Activation() BatchNormOps {
	b.BatchNormOps.Activation()
	return *b
}

//Normal sets and returns the Normal flag
func (b *BatchNormOps) Normal() BatchNormOps {
	b.BatchNormOps.Normal()
	return *b
}

//AddActivation sets and returns the AddActivation flag
func (b *BatchNormOps) AddActivation() BatchNormOps {
	b.BatchNormOps.AddActivation()
	return *b
}

//PoolingMode struct wrapper for gocudnn.PoolingMode.  Look up methods in gocudnn.
type PoolingMode struct {
	gocudnn.PoolingMode
}

//AverageCountExcludePadding sets and returns the AverageCountExcludePadding flag
func (p *PoolingMode) AverageCountExcludePadding() PoolingMode {
	p.PoolingMode.AverageCountExcludePadding()
	return *p
}

//AverageCountIncludePadding sets and returns the AverageCountIncludePadding flag
func (p *PoolingMode) AverageCountIncludePadding() PoolingMode {
	p.PoolingMode.AverageCountIncludePadding()
	return *p
}

//Max sets and returns the Max flag
func (p *PoolingMode) Max() PoolingMode {
	p.PoolingMode.Max()
	return *p
}

//MaxDeterministic sets and returns the MaxDeterministic flag
func (p *PoolingMode) MaxDeterministic() PoolingMode {
	p.PoolingMode.MaxDeterministic()
	return *p
}

//ActivationMode struct wrapper for gocudnn.ActivationMode.  Look up methods in gocudnn.
type ActivationMode struct {
	act.Mode
}

//ClippedRelu sets and returns the ClippedRelu flag
func (a *ActivationMode) ClippedRelu() ActivationMode {
	a.Mode.ClippedRelu()
	return *a
}

//Elu sets and returns the Elu flag
func (a *ActivationMode) Elu() ActivationMode {
	a.Mode.Elu()
	return *a
}

//Identity sets and returns the Identity flag
func (a *ActivationMode) Identity() ActivationMode {
	a.Mode.Identity()
	return *a
}

//Leaky sets and returns the Leaky flag
func (a *ActivationMode) Leaky() ActivationMode {
	a.Mode.Leaky()
	return *a
}

//PRelu sets and returns the PRelu flag
func (a *ActivationMode) PRelu() ActivationMode {
	a.Mode.PRelu()
	return *a
}

//Relu sets and returns the Relu flag
func (a *ActivationMode) Relu() ActivationMode {
	a.Mode.Relu()
	return *a
}

//Sigmoid sets and returns the Sigmoid flag
func (a *ActivationMode) Sigmoid() ActivationMode {
	a.Mode.Sigmoid()
	return *a
}

//Tanh sets and returns the Tanh flag
func (a *ActivationMode) Tanh() ActivationMode {
	a.Mode.Tanh()
	return *a
}

//Threshhold sets and returns the Threshhold flag
func (a *ActivationMode) Threshhold() ActivationMode {
	a.Mode.Threshhold()
	return *a
}

//SoftmaxAlgo determins what algo to use for softmax
type SoftmaxAlgo struct {
	gocudnn.SoftMaxAlgorithm
}

//Accurate sets and returns the Accurate flag
func (s *SoftmaxAlgo) Accurate() SoftmaxAlgo {
	s.SoftMaxAlgorithm.Accurate()
	return *s
}

//Fast sets and returns the Fast flag
func (s *SoftmaxAlgo) Fast() SoftmaxAlgo {
	s.SoftMaxAlgorithm.Fast()
	return *s
}

//Log sets and returns the Log flag
func (s *SoftmaxAlgo) Log() SoftmaxAlgo {
	s.SoftMaxAlgorithm.Log()
	return *s
}

//SoftmaxMode determins what mode to use for softmax
type SoftmaxMode struct {
	gocudnn.SoftMaxMode
}

//Channel sets and returns the Channel flag
func (s *SoftmaxMode) Channel() SoftmaxMode {
	s.SoftMaxMode.Channel()
	return *s
}

//Instance sets and returns the Instance flag
func (s *SoftmaxMode) Instance() SoftmaxMode {
	s.SoftMaxMode.Instance()
	return *s
}

//MathType is math type for tensor cores
type MathType struct {
	gocudnn.MathType
}

//AllowConversion sets and returns the AllowConversion flag
func (m *MathType) AllowConversion() MathType {
	m.MathType.AllowConversion()
	return *m
}

//Default sets and returns the Default flag
func (m *MathType) Default() MathType {
	m.MathType.Default()
	return *m
}

//TensorOpMath sets and returns the TensorOpMath flag
func (m *MathType) TensorOpMath() MathType {
	m.MathType.TensorOpMath()
	return *m
}

//Flags is a struct that should only be used for passing flags.
var Flags struct {
	Format TensorFormat
	Dtype  DataType
	Nan    NanProp
	CMode  ConvolutionMode
	BNMode BatchNormMode
	BNOps  BatchNormOps
	PMode  PoolingMode
	AMode  ActivationMode
	SMMode SoftmaxMode
	SMAlgo SoftmaxAlgo
	MType  MathType
	//EleOp  ElementwiseOp
}
var pflags struct {
	Format gocudnn.TensorFormat
	Dtype  gocudnn.DataType
	Nan    gocudnn.NANProp
	CMode  gocudnn.ConvolutionMode
	BNMode gocudnn.BatchNormMode
	BNOps  gocudnn.BatchNormOps
	PMode  gocudnn.PoolingMode
	AMode  act.Mode
	SMmode gocudnn.SoftMaxMode
	SMAlgo gocudnn.SoftMaxAlgorithm
}
