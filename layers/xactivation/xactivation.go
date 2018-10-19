package xactivation

import (
	"github.com/dereklstinson/GoCuNets/gocudnn/xactivation"
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
}

const defaultadambeta1 = 0.9
const defaultadambeta2 = 0.999
const defaultadameps = float32(1e-8)
const defaultadamrate = .001

//SetupStatic stages the layer it also needs input put layer to Build the output layer
func SetupStatic(
	h *gocudnn.XHandle,
	input *layers.IO,
	amode gocudnn.XActivationMode,
	tmode gocudnn.TrainingMode,
	coef float64,
	managedmem bool) (*Layer, *layers.IO, error) {
	frmt, dtype, dims, err := input.Properties()
	length := gocudnn.FindLength(input.T().Memer().ByteSize(), dtype)
	if err != nil {
		return nil, nil, err
	}
	op, err := xactivation.Stage(h, amode, tmode, dtype, coef)
	if err != nil {
		return nil, nil, err
	}

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
	var tmodeflg gocudnn.TrainingModeFlag
	var trp gocudnn.TrainingParams

	trp.SetBeta1(defaultadambeta1)
	trp.SetBeta2(defaultadambeta2)
	trp.SetEps(defaultadameps)
	trp.SetRate(defaultadamrate)
	dims[0] = 1
	alphas, err := layers.BuildIO(frmt, dtype, dims, managedmem)
	if err != nil {
		return nil, nil, err
	}

	alphas.T().SetRandom(.1, .05, float64(length))

	if err != nil {
		return nil, nil, err
	}
	if tmode == tmodeflg.Adam() || tmode == tmodeflg.AdaDelta() {

		if managedmem == true {

			xsum, err := gocudnn.MallocManaged(alphas.T().Memer().ByteSize(), gocudnn.ManagedMemFlag{}.Global())
			if err != nil {
				return nil, nil, err
			}
			gsum, err := gocudnn.MallocManaged(alphas.T().Memer().ByteSize(), gocudnn.ManagedMemFlag{}.Global())
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
				t:          trp,
			}, output, nil

		}

		xsum, err := gocudnn.Malloc(alphas.T().Memer().ByteSize())
		if err != nil {

			return nil, nil, err
		}
		gsum, err := gocudnn.Malloc(alphas.T().Memer().ByteSize())
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
			t:          trp,
		}, output, nil
	}

	if managedmem == true {

		gsum, err := gocudnn.MallocManaged(alphas.T().Memer().ByteSize(), gocudnn.ManagedMemFlag{}.Global())
		if err != nil {
			return nil, nil, err
		}
		gsum.Set(0)
		return &Layer{
			act:        op,
			alphas:     alphas,
			gsum:       gsum,
			memmanaged: managedmem,
			t:          trp,
		}, output, nil
	} else {

		gsum, err := gocudnn.Malloc(alphas.T().Memer().ByteSize())
		if err != nil {
			return nil, nil, err
		}
		gsum.Set(0)
		return &Layer{
			act:        op,
			alphas:     alphas,
			gsum:       gsum,
			memmanaged: managedmem,
			t:          trp,
		}, output, nil
	}

}
func findlengthfromdims(dims []int32) uint32 {
	mult := int32(1)
	for i := 1; i < len(dims); i++ {
		mult *= dims[i]
	}
	return uint32(mult)
}

//SetupNoOutDynamic does a setup with no output
func SetupNoOutDynamic(
	h *gocudnn.XHandle,
	input *layers.IO,
	amode gocudnn.XActivationMode,
	tmode gocudnn.TrainingMode,
	coef float64,
	managedmem bool) (*Layer, error) {
	frmt, dtype, dims, err := input.Properties()
	length := findlengthfromdims(dims)
	if err != nil {
		return nil, err
	}
	op, err := xactivation.Stage(h, amode, tmode, dtype, coef)
	if err != nil {
		return nil, err
	}

	var xactmodeflg gocudnn.XActivationModeFlag

	if amode == xactmodeflg.Leaky() {
		return &Layer{
			act: op,
		}, nil

	}
	var tmodeflg gocudnn.TrainingModeFlag
	var trp gocudnn.TrainingParams

	trp.SetBeta1(defaultadambeta1)
	trp.SetBeta2(defaultadambeta2)
	trp.SetEps(defaultadameps)
	trp.SetRate(defaultadamrate)
	dims[0] = 1
	alphas, err := layers.BuildIO(frmt, dtype, dims, managedmem)
	if err != nil {
		return nil, err
	}

	alphas.T().SetRandom(.1, .05, float64(length))

	if err != nil {
		return nil, err
	}
	if tmode == tmodeflg.Adam() || tmode == tmodeflg.AdaDelta() {

		if managedmem == true {

			xsum, err := gocudnn.MallocManaged(alphas.T().Memer().ByteSize(), gocudnn.ManagedMemFlag{}.Global())
			if err != nil {
				return nil, err
			}
			gsum, err := gocudnn.MallocManaged(alphas.T().Memer().ByteSize(), gocudnn.ManagedMemFlag{}.Global())
			if err != nil {
				xsum.Free()
				return nil, err
			}
			xsum.Set(0)
			gsum.Set(0)
			return &Layer{
				act:        op,
				alphas:     alphas,
				gsum:       gsum,
				xsum:       xsum,
				memmanaged: managedmem,
				t:          trp,
			}, nil

		}

		xsum, err := gocudnn.Malloc(alphas.T().Memer().ByteSize())
		if err != nil {

			return nil, err
		}
		gsum, err := gocudnn.Malloc(alphas.T().Memer().ByteSize())
		if err != nil {
			xsum.Free()
			return nil, err
		}
		xsum.Set(0)
		gsum.Set(0)
		return &Layer{
			act:        op,
			alphas:     alphas,
			gsum:       gsum,
			xsum:       xsum,
			memmanaged: managedmem,
			t:          trp,
		}, nil
	}

	if managedmem == true {

		gsum, err := gocudnn.MallocManaged(alphas.T().Memer().ByteSize(), gocudnn.ManagedMemFlag{}.Global())
		if err != nil {
			return nil, err
		}
		gsum.Set(0)
		return &Layer{
			act:        op,
			alphas:     alphas,
			gsum:       gsum,
			memmanaged: managedmem,
			t:          trp,
		}, nil
	} else {

		gsum, err := gocudnn.Malloc(alphas.T().Memer().ByteSize())
		if err != nil {
			return nil, err
		}
		gsum.Set(0)
		return &Layer{
			act:        op,
			alphas:     alphas,
			gsum:       gsum,
			memmanaged: managedmem,
			t:          trp,
		}, nil
	}

}

//ForwardProp does the forward prop
func (l *Layer) ForwardProp(handle *gocudnn.XHandle, x, y *layers.IO) error {

	if l.alphas == nil {
		return l.act.FwdProp(handle, x.T(), y.T(), nil)
	}
	return l.act.FwdProp(handle, x.T(), y.T(), l.alphas.T())
}

//BackProp does the backprop operation
func (l *Layer) BackProp(handle *gocudnn.XHandle, x, y *layers.IO) error {

	if l.alphas == nil {
		return l.act.BwdProp(handle, x.T(), x.DeltaT(), y.DeltaT(), nil, nil)
	}
	return l.act.BwdProp(handle, x.T(), x.DeltaT(), y.DeltaT(), l.alphas.T(), l.alphas.DeltaT())

}

//UpdateParams updates the params as long as it is set up that way
func (l *Layer) UpdateParams(handle *gocudnn.XHandle, batchsize int) error {

	return l.act.UpdateParams(handle, batchsize, l.alphas.T(), l.alphas.DeltaT(), l.gsum, l.xsum, l.t)
}
