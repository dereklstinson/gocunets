package xactivation

import (
	"github.com/dereklstinson/GoCuNets/gocudnn/tensor"
	"github.com/dereklstinson/GoCuNets/gocudnn/tensor/xactivation"
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Layer is an activation layer
type Layer struct {
	act        *xactivation.Ops
	memmanaged bool
	alphas     *layers.IO
	gsum       *gocudnn.Malloced
	xsum       *gocudnn.Malloced
	t          gocudnn.TrainingParams
	batchsize  int
}

const defaultadambeta1 = 0.9
const defaultadambeta2 = 0.999
const defaultadameps = float32(1e-8)
const defaultadamrate = .001

//Setup stages the layer it also needs input put layer to Build the output layer
func Setup(
	h *gocudnn.XHandle,
	input *layers.IO,
	blocksize uint32,
	amode gocudnn.XActivationMode,
	tmode gocudnn.TrainingMode,
	coef float64,
	managedmem bool,
	batchsize int) (*Layer, *layers.IO, error) {
	frmt, dtype, dims, err := input.Properties()
	length := gocudnn.FindLength(input.T().Memer().ByteSize(), dtype)
	if err != nil {
		return nil, nil, err
	}
	op, err := xactivation.Stage(h, blocksize, amode, tmode, dtype, coef)
	if err != nil {
		return nil, nil, err
	}
	var alphas *layers.IO
	var gsum *gocudnn.Malloced
	var xsum *gocudnn.Malloced
	var tmodeflg gocudnn.TrainingModeFlag
	var xactmodeflg gocudnn.XActivationModeFlag
	output, err := layers.BuildIO(frmt, dtype, dims, managedmem)
	if err != nil {
		return nil, nil, err
	}
	if amode == xactmodeflg.Leaky() {
		return &Layer{
			act: op,
		}, output, nil

	}

	var trp gocudnn.TrainingParams

	trp.SetBeta1(defaultadambeta1)
	trp.SetBeta2(defaultadambeta2)
	trp.SetEps(defaultadameps)
	trp.SetRate(defaultadamrate)
	alphas, err = layers.BuildIO(frmt, dtype, dims, managedmem)
	if err != nil {
		return nil, nil, err
	}

	alphas.T().SetRandom(.01, .001, float64(length))

	if err != nil {
		return nil, nil, err
	}
	if tmode == tmodeflg.Adam() || tmode == tmodeflg.AdaDelta() {

		if managedmem == true {
			var gsum *gocudnn.Malloced
			var xsum *gocudnn.Malloced
			xsum, err = gocudnn.MallocManaged(input.T().Memer().ByteSize(), gocudnn.ManagedMemFlag{}.Global())
			if err != nil {
				return nil, nil, err
			}
			gsum, err = gocudnn.MallocManaged(input.T().Memer().ByteSize(), gocudnn.ManagedMemFlag{}.Global())
			if err != nil {
				xsum.Free()
				return nil, nil, err
			}
			xsum.Set(0)
			gsum.Set(0)
			return &Layer{
				act:        op,
				alphas:     alphas,
				gsum:       gsum,
				xsum:       xsum,
				memmanaged: managedmem,
				batchsize:  batchsize,
				t:          trp,
			}, output, nil

		}
		xsum, err = gocudnn.Malloc(input.T().Memer().ByteSize())
		if err != nil {

			return nil, nil, err
		}
		gsum, err = gocudnn.Malloc(input.T().Memer().ByteSize())
		if err != nil {
			xsum.Free()
			return nil, nil, err
		}
		xsum.Set(0)
		gsum.Set(0)
		return &Layer{
			act:        op,
			alphas:     alphas,
			gsum:       gsum,
			xsum:       xsum,
			memmanaged: managedmem,
			batchsize:  batchsize,
			t:          trp,
		}, output, nil
	}

	if managedmem == true {
		gsum, err = gocudnn.MallocManaged(input.T().Memer().ByteSize(), gocudnn.ManagedMemFlag{}.Global())
		if err != nil {
			return nil, nil, err
		}
		gsum.Set(0)
		return &Layer{
			act:        op,
			alphas:     alphas,
			gsum:       gsum,
			memmanaged: managedmem,
			batchsize:  batchsize,
			t:          trp,
		}, output, nil
	} else {
		gsum, err = gocudnn.Malloc(input.T().Memer().ByteSize())
		if err != nil {
			xsum.Free()
			return nil, nil, err
		}
		gsum.Set(0)
		return &Layer{
			act:        op,
			alphas:     alphas,
			gsum:       gsum,
			memmanaged: managedmem,
			batchsize:  batchsize,
			t:          trp,
		}, output, nil
	}

}

//ForwardProp does the forward prop
func (l *Layer) ForwardProp(handle *gocudnn.XHandle, x, y *layers.IO) error {
	var alpha *tensor.Volume
	if l.alphas == nil {
		alpha = nil
	} else {
		alpha = l.alphas.T()
	}
	return l.act.FwdProp(handle, x.T(), y.T(), alpha)
}

//BackProp does the backprop operation
func (l *Layer) BackProp(handle *gocudnn.XHandle, x, y *layers.IO) error {

	if l.alphas == nil {
		return l.act.BwdProp(handle, x.DeltaT(), y.DeltaT(), nil, nil)
	}
	return l.act.BwdProp(handle, x.DeltaT(), y.DeltaT(), l.alphas.T(), l.alphas.DeltaT())

}

//UpdateParams updates the params as long as it is set up that way
func (l *Layer) UpdateParams(handle *gocudnn.XHandle) error {

	return l.act.UpdateParams(handle, l.batchsize, l.alphas.T(), l.alphas.DeltaT(), l.gsum, l.xsum, l.t)
}
