package nvutil

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/tensor"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/jpeg"
	"github.com/dereklstinson/GoCuNets/utils"
	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/GoCudnn/npp"
)

type BatchBuffer struct {
	head            *npp.Uint8
	batchchansuint8 [][]*npp.Uint8
	size            npp.Size
	strides         []int32
	nchw            bool
	dims            []int32
	average         float32
	volume          int32
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
		average: (averagepixel),
		volume:  volume,
	}
}

//this will only work if in nchw
func (b *BatchBuffer) getbatcheschannelsptrs(batch int32) (channels []*npp.Uint8) {
	batchchan := make([]int32, len(b.dims))

	batchchan[0] = batch

	for i := range batchchan {
		batchchan[i] = 1
	}
	if !b.nchw {
		channels := make([]*npp.Uint8, 1)

		channels[0] = b.getpointerat(batchchan)
	}
	chans := b.dims[1]
	channels = make([]*npp.Uint8, chans)
	for i := range channels {
		batchchan[1] = int32(i)
		channels[i] = b.getpointerat(batchchan)
	}
	return channels
}
func (b *BatchBuffer) getpointerat(location []int32) *npp.Uint8 {

	loc := int32(0)
	for i := range location {
		loc += location[i] * b.strides[i]
	}
	gomem := gocu.Offset(b.head, uint(loc))
	return (*npp.Uint8)(gomem.Ptr())

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
			//chans := imgs[i].GetChannels()
			w, h := imgs[i].Size()
			srcROIs[i].Set(0, 0, w, h)
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
		sizes[i] = append(sizes[i], size)
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
