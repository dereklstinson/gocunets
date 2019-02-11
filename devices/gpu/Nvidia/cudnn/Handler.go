/*
Package cudnn takes the descriptors from GoCudnn which is from cudnn and seperates them into seperate packages.
In the hopes to eventually move away from all those pesky flags.  Some flags will have to exist still, but where I can get rid of them I will.
Like DataType and NanProp and TensorFormat, and even algorithm for convolution.
*/
package cudnn

import (
	"github.com/dereklstinson/GoCuNets/utils"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Handler contains the handles used in gocudnn and also the xtra kernals.
type Handler struct {
	cudnn    *gocudnn.Handle
	xtra     *gocudnn.XHandle
	stream   *gocudnn.Stream
	unified  bool
	maxbatch int32
}

//FindMaxVol will find the max vol for tensor.  This is going to hold two functions
//pmax can be either the previous maxvol, or it could be
func (h *Handler) FindMaxVol(outputdims []int32) int32 {
	if h.maxbatch <= 0 {
		utils.FindVolumeInt32(outputdims, nil)
	}
	return utils.FindMaxVolThroughMaxBatch(h.maxbatch, outputdims)
}

//SetMaxBatch sets the max batch used behind the scenes to allow dynamic resizing of tensors.
func (h *Handler) SetMaxBatch(maxbatchsize int32) {
	h.maxbatch = maxbatchsize
}

//GetMaxBatch returns the max batch
func (h *Handler) GetMaxBatch() int32 {
	if h.maxbatch < 1 {
		return 1
	}
	return h.maxbatch

}

//FindMaxSizeT returns the max sizeT
func (h *Handler) FindMaxSizeT(outputdims []int32) SizeT {
	if h.maxbatch <= 0 {
		return SizeT(utils.FindVolumeInt32(outputdims, nil) * 4)
	}
	return SizeT(utils.FindMaxVolThroughMaxBatch(h.maxbatch, outputdims) * 4)
}

//Unified returns if the device the handler is using uses unified memory
func (h *Handler) Unified() bool {
	return h.unified
}

//MakeNotUnified sets unified flag for device the handler is using to false.
func (h *Handler) MakeNotUnified() {
	h.unified = false
}

//Cudnn returns a pointer to the cudnn handle
func (h *Handler) Cudnn() *gocudnn.Handle {
	return h.cudnn
}

//Stream returns the stream
func (h *Handler) Stream() *gocudnn.Stream {
	return h.stream
}

//XHandle returns a pointer to the XHandle
func (h *Handler) XHandle() *gocudnn.XHandle {
	return h.xtra
}

//Sync syncs the streams
func (h *Handler) Sync() error {
	if h.stream == nil {
		return gocudnn.Cuda{}.CtxSynchronize()
	}
	return h.stream.Sync()
}

//SyncContext will sync the contexts
func (h *Handler) SyncContext() error {
	return gocudnn.Cuda{}.CtxSynchronize()
}

//DeviceSync syncs the device
func (h *Handler) DeviceSync() error {
	return gocudnn.Cuda{}.DeviceSync()
}

//CreateHandler creates a the handlers
//The handler is used in managing memory for all the packages that use cudnn.Handler. This function will raise a flag that will tell the program
//to use unified memory management.  If that is not wanted call MakeNotUnified immediately to turn this off.
func CreateHandler(dev *gocudnn.Device, xtrakernsfolder string) *Handler {
	err := dev.Set()
	if err != nil {
		panic(err)
	}
	var unified bool
	if 6 < dev.Major() {
		unified = true
	}
	x := gocudnn.NewHandle()
	y, err := gocudnn.Xtra{}.MakeXHandle(xtrakernsfolder, dev)
	if err != nil {
		panic(err)
	}

	return &Handler{
		cudnn:    x,
		xtra:     y,
		unified:  unified,
		maxbatch: 0,
	}
}

//SetStream sets the stream for the handles
func (h *Handler) SetStream(stream *gocudnn.Stream) error {
	if h.stream != nil {
		h.stream.Destroy()
	}
	err := h.cudnn.SetStream(stream)
	if err != nil {
		return err
	}
	err = h.xtra.SetStream(stream)
	if err != nil {
		return err
	}
	h.stream = stream
	return nil
}
