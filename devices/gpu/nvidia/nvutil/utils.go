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
func resizenpp(h *Handle, src, dest []*npp.Uint8, srcSize, destSize npp.Size, srcROI, destROI npp.Rect) error {
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
func mirror(h *Handle, src, dest []*npp.Uint8, sizes npp.Size, flip npp.Axis) error {
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

//CreateBatchBuffer is NCHW. And works for stand alone images
func CreateBatchBuffer(dims []int32, averagepixel float32) *BatchBuffer {
	strides := utils.FindStridesInt32(dims)
	volume := utils.FindVolumeInt32(dims, nil)
	head := npp.Malloc8u(volume)
	var size npp.Size
	size.Set(dims[3], dims[2])
	return &BatchBuffer{
		dims:    dims,
		strides: strides,
		head:    head,
		nchw:    true,
		size:    size,
		average: (npp.Float32)(averagepixel),
		volume:  volume,
	}
}

func (b *BatchBuffer) GetProperties() (dims []int32, nchw bool) {
	return b.dims, b.nchw
}

//LoadImages will take the images and resize them considering the srcROIs,and destROIs to the buffer.
//
//if srcROIS and/or destROIs are nil, then it will make the ROIs that are null the size of the src and dest space.
//
//if src or dest is not nil. Then the length needs to be the same as imgs.
func (b *BatchBuffer) LoadImages(h *Handle, imgs []*jpeg.Image, srcROIs, destROIs []npp.Rect) (err error) {

	if srcROIs == nil {
		//do stuff
		srcROIs = make([]npp.Rect, len(imgs))
		for i := range imgs {
			chans := imgs[i].GetChannels()
			srcROIs[i].Set(0, 0, chans[0].Pitch, chans[0].Height)
		}
		//ok i did the stuff
	}
	if destROIs == nil {
		//do stuff
		destROIs = make([]npp.Rect, len(imgs))
		for i := range imgs {
			if b.nchw {
				destROIs[i].Set(0, 0, b.dims[3], b.dims[2])
			} else {
				destROIs[i].Set(0, 0, b.dims[2], b.dims[1])
			}

		}
		//ok I did the stuff
	}
	if len(srcROIs) != len(imgs) || len(srcROIs) != len(imgs) {
		return errors.New("srcROIs and destROIs and imgs need to be the same length")
	}
	imgchanptrs := make([][]*npp.Uint8, len(imgs))
	sizes := make([][]npp.Size, len(imgs))
	for i := range imgs {
		imgchannels, size := ImageToNppi(imgs[i])
		imgchanptrs[i] = append(imgchanptrs[i], imgchannels...)
		sizes[i] = append(sizes[i], size...)
	}
	for i, channels := range imgchanptrs {
		err = resizenpp(h, channels, b.getbatcheschannelsptrs(int32(i)), sizes[i][0], b.size, srcROIs[i], destROIs[i])
		if err != nil {
			return err
		}
	}
	return nil
}

//Mirror will mirror the batches indicated in batches.  len(onaxis)==len(batches) else it will cause an error
func (b *BatchBuffer) Mirror(h *Handle, batches []int32, onaxis []npp.Axis) (err error) {
	if len(batches) != len(onaxis) {
		return errors.New("(b *BatchBuffer) Mirror -- len(batches)!=len(onxis)")
	}

	for i, axis := range onaxis {
		if batches[i] > b.dims[0]-1 {
			return errors.New("(b *BatchBuffer) Mirror -- batches[i]>b.dims[0]-1")
		}
		chans := b.getbatcheschannelsptrs(batches[i])
		err = mirror(h, chans, nil, b.size, axis)
		if err != nil {
			return err
		}
	}
	return nil
}
func (b *BatchBuffer) FillTensor(h *Handle, t *tensor.Volume) error {
	dtype := t.DataType()
	dflg := dtype
	switch dtype {
	case dflg.Float():
		err := npp.Convert8u32f(b.head, (*npp.Float32)(t.Memer().Ptr()), b.volume, h.ctx)
		if err != nil {
			return err
		}
		return npp.DivC32fI(b.average, (*npp.Float32)(t.Memer().Ptr()), b.volume, h.ctx)
	default:
		return errors.New("Only Float supported")

	}
}

func findfittingpad(src, dst, stride int32) int32 {

	sections := intceiling(src-dst, stride) + 1
	return (1 - src%sections) * sections

}

func intceiling(a, b int32) int32 {
	return ((a - int32(1)) / b) + int32(1)
}

/*
func findSrcROIandDstROI(srcH, srcW, strideH, strideW, dstH, dstW int32) (srcROI, dstROI []npp.Rect) {
	srcROI := make([]npp.Rect, 0)
	dstROI := make([]npp.Rect, 0)
	for i := 0; i < srcH; strideH {
		for j := 0; j < srcW; j += strideW {

		}
	}
}

//NCSHW==numbatches,channels,sections,height,width
func imgto4dsignalNCSHW(img jpeg.Image, window, stride []int32, imagesignal *npp.Uint8) error {
	imagechan := img.GetChannels()
	for i, im := range imagechan {
		h := im.Height
		w := im.Pitch
		ptr := im.Ptr
	}
}
*/
//this will only work if in nchw
func (b *BatchBuffer) getbatcheschannelsptrs(batch int32) (channels []*npp.Uint8) {
	batchchan := make([]int32, len(b.dims))

	batchchan[0] = batch

	for i := range batchchan {
		batchchan[i] = 1
	}
	if !b.nchw {
		channels := make([]*npp.Uint8, 1)

		channels[0] = b.getptrat(batchchan)
	}
	chans := b.dims[1]
	channels = make([]*npp.Uint8, chans)
	for i := range channels {
		batchchan[1] = int32(i)
		channels[i] = b.getptrat(batchchan)
	}
	return channels
}
func (b *BatchBuffer) getptrat(location []int32) *npp.Uint8 {

	loc := int32(0)
	for i := range location {
		loc += location[i] * b.strides[i]
	}
	gomem := gocu.Offset(b.head, uint(loc))
	return (*npp.Uint8)(gomem.Ptr())

}

type BatchBuffer struct {
	head            *npp.Uint8
	batchchansuint8 [][]*npp.Uint8
	size            npp.Size
	strides         []int32
	nchw            bool
	dims            []int32
	average         npp.Float32
	volume          int32
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

func nppuint8tonppfloat32(h *Handle, src *npp.Uint8, dst *npp.Float32, length int32) error {
	return npp.Convert8u32f(src, dst, length, h.ctx)
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
