package cnndots

import (
	"github.com/dereklstinson/GoCuNets/arrays"
	"github.com/dereklstinson/cuda"
)

type CULayer3d struct {
	neurons   []arrays.CUArray3d
	gradadds  []arrays.CUArray3d
	biases    []float32
	bgradadds []float32
	x, y      dependents
	dropped   bool
	ptx       string
	module    *cuda.Module
	gdim      cuda.Dims
	bdim      cuda.Dims
	shared    uint
	stream    *cuda.Stream
	offset    int
}

func (layer *CULayer3d) Forward(input *arrays.CUArray3d, output *arrays.CUArray3d) {
	ix, iy, iz := input.XYZ()
	px := layer.x.pad
	py := layer.y.pad
	sx := layer.x.slide
	sy := layer.y.slide
	for i := 0; i < len(layer.neurons); i++ {
		for offset := 0; offset < layer.offset; offset++ {
			layer.module.LaunchDims(layer.ptx, layer.gdim, layer.bdim, layer.shared, layer.stream, input.DPTR(), layer.neurons[i].DPTR(), output.DPTR(), ix, iy, iz, px, py, sx, sy, offset, i)
		}
	}

}
