package cnndots

import (
	"github.com/dereklstinson/GoCuNets/arrays"
	"github.com/dereklstinson/GoCuNets/gpu"
	"github.com/dereklstinson/cuda"
)

type CULayer3d struct {
	device           gpu.Compute
	neurons          []arrays.CUArray3d
	gradadds         []arrays.CUArray3d
	biases           arrays.CUArray1d
	bgradadds        arrays.CUArray1d
	x, y             dependents
	dropped          bool
	gdim             cuda.Dims
	bdim             cuda.Dims
	shared           uint
	inputlength      int
	offset           int
	forwardfuncname  string
	backwardfuncname string
}
type Attributes struct {
	forwardfuncname  string
	backwardfuncname string
	kernelfolder     string
	kernelname       string
	neuronnum        int
	x                int
	y                int
	z                int
	dropped          bool
	padx             int
	pady             int
	slidex           int
	slidey           int
}

//Build builds a layer
func (layer *CULayer3d) Build(device *cuda.Device, buffersize int, x Attributes) error {
	var err error
	layer.device, err = gpu.BuildCompute(device, buffersize)
	if err != nil {
		return err
	}
	err = layer.device.LoadPTXinfo(x.kernelfolder, x.kernelname)
	if err != nil {
		return err
	}
	layer.neurons = make([]arrays.CUArray3d, x.neuronnum)
	layer.gradadds = make([]arrays.CUArray3d, x.neuronnum)
	for i := 0; i < x.neuronnum; i++ {
		layer.neurons[i], err = arrays.CreateCUArray3d(arrays.NewHArray3d(x.x, x.y, x.z))
		if err != nil {
			return err
		}
		layer.gradadds[i], err = arrays.CreateCUArray3d(arrays.NewHArray3d(x.x, x.y, x.z))
		if err != nil {
			return err
		}
	}

	return nil
}

//Forward is the forward function of the cnn layer
func (layer *CULayer3d) Forward(input *arrays.CUArray3d, output *arrays.CUArray3d) error {
	ix, iy, iz := input.XYZ()
	px := layer.x.pad
	py := layer.y.pad
	sx := layer.x.slide
	sy := layer.y.slide
	err := <-layer.device.Context.Run(func() error {
		module, err := cuda.NewModule(layer.device.Context, layer.device.Ptx)
		if err != nil {
			return err
		}
		stream, err := cuda.NewStream(true)
		if err != nil {
			return err
		}
		for i := 0; i < len(layer.neurons); i++ {
			for offset := 0; offset < layer.offset; offset++ {
				module.LaunchDims(layer.forwardfuncname, layer.gdim, layer.bdim, layer.shared, stream, input.DPTR(), layer.neurons[i].DPTR(), output.DPTR(), ix, iy, iz, px, py, sx, sy, offset, i)
			}

		}
		return nil
	})
	if err != nil {
		return err

	}
	return nil
}
