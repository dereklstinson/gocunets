//Package nvutil are functions that use the other nvidia packages and allows them to be used with each other
package nvutil

import (
	"errors"
	"github.com/dereklstinson/GoCuNets/utils"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/tensor"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/jpeg"

	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/GoCudnn/npp"
)

//ImageToNppi Converts an jpeg.Image to a npp.type with its npp.Size
func ImageToNppi(img *jpeg.Image) (planar []*npp.Uint8, sizes []npp.Size) {
	chans := img.GetChannels()
	planar = make([]*npp.Uint8, len(chans))
	sizes = make([]npp.Size, len(chans))
	for i := range chans {
		planar[i] = (*npp.Uint8)(chans[i].Ptr.Ptr())
		sizes[i].Set(chans[i].Height, chans[i].Pitch)
	}
	return planar, sizes

}

type Handle struct {
	ctx      *npp.StreamContext
	polation npp.InterpolationMode
}

func CreateHandle(ctx *npp.StreamContext, polation npp.InterpolationMode) *Handle {
	return &Handle{
		ctx:      ctx,
		polation: polation,
	}
}
func ResizeNppi(h *Handle, src, dest []*npp.Uint8, srcSize, destSize npp.Size) error {
	var srcROI npp.Rect
	var destROI npp.Rect

	switch len(src) {
	case 1:
		return npp.Resize8uC1R(src[0], srcSize, 0, srcROI, dest[0], destSize, 0, destROI, h.polation, h.ctx)
	case 3:
		return npp.Resize8uP3R(src, srcSize, 0, srcROI, dest, destSize, 0, destROI, h.polation, h.ctx)
	case 4:
		return npp.Resize8uP4R(src, srcSize, 0, srcROI, dest, destSize, 0, destROI, h.polation, h.ctx)
	}
	return errors.New("Unsupported src,dest size")
}

func cudnnbatchchannelsize(x *tensor.Volume) (batch int, channel int, err error) {
	frmt, _, dims, err := x.Properties()
	if err != nil {
		return 0, 0, err
	}

	fflg := frmt
	switch frmt {
	case fflg.NCHW():
		return int(dims[0]), int(dims[1]), nil
	case fflg.NHWC():
		return int(dims[0]), int(dims[len(dims)-1]), nil

	default:
		return -1, -1, errors.New("Unsupported Format")

	}
}

//Mirror - flips images according to axis. If dest is nil then function is done in place
func Mirror(h *Handle, src, dest []*npp.Uint8, sizes npp.Size, flip npp.Axis) error {
	var err error
	if dest == nil {
		for i := range src {
			err = npp.Mirror8uC1IR(src[i], 0, sizes, flip, h.ctx)
			if err != nil {
				return err
			}
		}

	}
	for i := range src {
		err = npp.Mirror8uC1R(src[i], 0, dest[i], 0, sizes, flip, h.ctx)
		if err != nil {
			return err
		}
	}

	return nil
}

//CreateBatchMaker is NCHW. And works for stand alone images
func CreateBatchBuffer(dims []int32) *BatchBuffer {
	strides := utils.FindStridesInt32(dims)
	head := npp.Malloc8u(utils.FindVolumeInt32(dims, nil))

	return &BatchBuffer{
		dims:    dims,
		strides: strides,
		head:    head,
		nchw:    true,
	}
}
func (b *BatchBuffer) batchchannel(batch, channel int32) gocu.Mem {
	if b.nchw {
		return gocu.Offset(b.head, uint(batch*b.strides[0]+channel*b.strides[1]))
	}
	return gocu.Offset(b.head, uint(batch*b.strides[0]))

}

type BatchBuffer struct {
	currentimages   []*jpeg.Image
	rois            []npp.Rect
	head            *npp.Uint8
	batchchansuint8 [][]*npp.Uint8
	strides         []int32
	nchw            bool
	dims            []int32
}

func convertNppitoNppsCHW(channel []*npp.Uint8, sizes []npp.Size, mem *npp.Uint8) (n uint, err error) {

	var destoffset, srcsize uint

	for i := range sizes {
		h, w := sizes[i].Get()
		srcsize = uint(h * w)
		destoffset = srcsize * uint(i)

		err = nvidia.Memcpy(gocu.Offset(mem, (destoffset)), channel[i], srcsize)
		if err != nil {
			return n, err
		}
		n += srcsize
	}
	return n, nil
}
func convertsCHWstoNCHW(srcs []*npp.Uint8, srcsSIBs []uint, dest *npp.Uint8) (n uint, err error) {
	var destoffset uint
	for i := range srcs {
		destoffset = srcsSIBs[i] * uint(i)
		err = nvidia.Memcpy(gocu.Offset(dest, destoffset), srcs[i], srcsSIBs[i])
		if err != nil {
			return n, err
		}
		n += srcsSIBs[i]
	}
	return n, nil
}

func convertNppitoNppsNCHW(channels [][]*npp.Uint8, sizes [][]npp.Size, mem *npp.Uint8) error {
	coffsets := make([][]int, 0)
	boffsets := make([]int, 0)
	var err error
	for i := range sizes {
		var adder int
		coffsets[i] = make([]int, len(sizes[i]))
		for j := range sizes[i] {
			adder += chanoffset(sizes[i][j])
			coffsets[i] = append(coffsets[i], adder)
		}
		boffsets = append(boffsets, adder)
	}

	for i := range channels {

		for j := range channels[i] {
			err = nvidia.Memcpy(gocu.Offset(mem, uint((i*boffsets[i])+(j*coffsets[i][j]))), channels[i][j], (uint)(coffsets[i][j]))
			if err != nil {
				return err
			}
		}

	}
	return nil
}

func nppuint8tonppfloat32(src *npp.Uint8, dst *npp.Float32, length int32) error {
	return npp.Convert8u32f(src, dst, length)
}

func chanoffset(size npp.Size) int {
	w, h := size.Get()
	return int(w * h)
}

func totalVol(sizes [][]npp.Size) int {

	var h, w, adder int32
	for i := range sizes {
		for j := range sizes[i] {
			w, h = (sizes[i][j].Get())
			adder += (int32)(w * h)
		}
	}

	return (int)(adder)
}
