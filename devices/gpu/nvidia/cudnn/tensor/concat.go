package tensor

import (
	"errors"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCudnn/cudart"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

//CreateOutputConcatVolume creates a volume out of the concat values
func CreateOutputConcatVolume(h *cudnn.Handler, src []*Volume) (dest *Volume, err error) {
	var tfp gocudnn.TensorFormat

	var tfflag gocudnn.TensorFormat
	for i := range src {
		if i == 0 {
			tfp = src[i].TD().Format()

		}
		if tfp != src[i].TD().Format() {
			return nil, errors.New("CreateOutputConcatVolume: src tensor descriptors not all matching")
		}

	}

	switch tfp {
	case tfflag.NHWC():
		return createconcatNHWC(h, src)
	case tfflag.NCHW():
		return createconcatNCHW(h, src)

	}
	return nil, errors.New("Not supported Tensor Format")
}

func createconcatNCHW(h *cudnn.Handler, src []*Volume) (dst *Volume, err error) {
	var dims []int32
	var fmtf gocudnn.TensorFormat
	var channels []int32
	for i := range src {
		if i == 0 {
			dims = src[i].TD().Dims()
		}

		cdims := src[i].Dims()
		if cdims[0] != dims[0] {
			return nil, errors.New("Concat batchsize needs to be the same")
		}
		for j := 2; j < len(cdims); j++ {
			if dims[j] != cdims[j] {
				return nil, errors.New("Concat all dim sizes(except for channel) need to be the same")
			}
		}

		channels = append(channels, cdims[1])
	}
	var sum int32
	for i := range channels {
		sum += channels[i]
	}
	dims[1] = sum

	return Build(h, fmtf.NCHW(), src[0].DataType(), dims)
}
func createconcatNHWC(h *cudnn.Handler, src []*Volume) (dst *Volume, err error) {
	var dims []int32
	var fmtf gocudnn.TensorFormat
	var channels []int32
	for i := range src {
		if i == 0 {
			dims = src[i].TD().Dims()
		}

		cdims := src[i].Dims()
		if cdims[0] != dims[0] {
			return nil, errors.New("Concat batchsize needs to be the same")
		}
		for j := 1; j < len(cdims)-1; j++ {
			if dims[j] != cdims[j] {
				return nil, errors.New("Concat all dim sizes(except for channel) need to be the same")
			}
		}

		channels = append(channels, cdims[len(cdims)-1])
	}
	var sum int32
	for i := range channels {
		sum += channels[i]
	}
	dims[len(dims)-1] = sum

	return Build(h, fmtf.NHWC(), src[0].DataType(), dims)
}

//Concat does a concat run CreateOutputConcatVolume before using.
func Concat(srcs []*Volume, dest *Volume) error {
	destdims := dest.Dims()
	destbatches := destdims[0]
	destbatchoffset := gocudnn.FindSizeTfromVol(destdims[1:], dest.dtype)
	var mckf cudart.MemcpyKind

	var tff gocudnn.TensorFormat
	switch dest.Format() {
	case tff.NCHW():
		for j := (int32)(0); j < destbatches; j++ {
			var uisrcoffset uint
			for i := range srcs {
				srcdims := srcs[i].Dims()
				srcbatchoffset := gocudnn.FindSizeTfromVol(srcdims[1:], dest.dtype)
				srcchanoffset := gocudnn.FindSizeTfromVol(srcdims[2:], dest.dtype)
				uj := uint(j)
				ui := uint(i)
				uisrcoffset += srcchanoffset
				cudart.MemCpy(dest.memgpu.OffSet((uj*destbatchoffset)+uisrcoffset), srcs[i].memgpu.OffSet((uj*srcbatchoffset)+(ui*srcchanoffset)), srcchanoffset, mckf.Default())

			}

		}

	case tff.NHWC():
		return errors.New("NHWC not supported")
	}
	return nil
}

//ReverseConcat does it reverse style run CreateOutputConcatVolume before using.
func ReverseConcat(src *Volume, dests []*Volume) error {
	srcs := dests
	dest := src
	destdims := dest.Dims()
	destbatches := destdims[0]
	destbatchoffset := gocudnn.FindSizeTfromVol(destdims[1:], dest.dtype)
	var mckf cudart.MemcpyKind

	var tff gocudnn.TensorFormat
	switch dest.Format() {
	case tff.NCHW():
		for j := (int32)(0); j < destbatches; j++ {
			var uisrcoffset uint
			for i := range srcs {
				srcdims := srcs[i].Dims()
				srcbatchoffset := gocudnn.FindSizeTfromVol(srcdims[1:], dest.dtype)
				srcchanoffset := gocudnn.FindSizeTfromVol(srcdims[2:], dest.dtype)
				uj := uint(j)
				ui := uint(i)
				uisrcoffset += srcchanoffset
				cudart.MemCpy(srcs[i].memgpu.OffSet((uj*srcbatchoffset)+(ui*srcchanoffset)), dest.memgpu.OffSet((uj*destbatchoffset)+uisrcoffset), srcchanoffset, mckf.Default())

			}

		}

	case tff.NHWC():
		return errors.New("NHWC not supported")
	}
	return nil
}
