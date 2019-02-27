/*
Package cudnn takes the descriptors from GoCudnn which is from cudnn and seperates them into seperate packages.
In the hopes to eventually move away from all those pesky flags.  Some flags will have to exist still, but where I can get rid of them I will.
Like DataType and NanProp and TensorFormat, and even algorithm for convolution.
*/
package cudnn

import (
	"github.com/dereklstinson/GoCuNets/utils"
	gocudnn "github.com/dereklstinson/GoCudnn"
	"github.com/dereklstinson/GoCudnn/cuda"
	"github.com/dereklstinson/GoCudnn/cudart"
	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/GoCudnn/xtra"
)

//Handler contains the handles used in gocudnn and also the xtra kernals.
type Handler struct {
	cudnn    *gocudnn.Handle
	xtra     *xtra.Handle
	stream   gocu.Streamer
	unified  bool
	maxbatch int32
	device   cudart.Device
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

//FindMaxUint returns the max sizeT
func (h *Handler) FindMaxUint(outputdims []int32) uint {
	if h.maxbatch <= 0 {
		return uint(utils.FindVolumeInt32(outputdims, nil) * 4)
	}
	return uint(utils.FindMaxVolThroughMaxBatch(h.maxbatch, outputdims) * 4)
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
func (h *Handler) Stream() gocu.Streamer {
	return h.stream
}

//XHandle returns a pointer to the XHandle
func (h *Handler) XHandle() *xtra.Handle {
	return h.xtra
}

//Sync syncs the streams
func (h *Handler) Sync() error {
	if h.stream == nil {
		return cuda.CtxSynchronize()
	}
	return h.stream.Sync()
}

//SyncContext will sync the contexts
func (h *Handler) SyncContext() error {
	return cuda.CtxSynchronize()
}

//DeviceSync syncs the device
func (h *Handler) DeviceSync() error {
	return h.device.DeviceSync()
}

//CreateHandler creates a the handlers
//The handler is used in managing memory for all the packages that use cudnn.Handler. This function will raise a flag that will tell the program
//to use unified memory management.  If that is not wanted call MakeNotUnified immediately to turn this off.
func CreateHandler(dev cudart.Device, xtrakernsfolder string) *Handler {
	err := dev.Set()
	if err != nil {
		panic(err)
	}
	var unified bool
	major, err := dev.Major()
	if err != nil {
		return nil
	}
	if 6 < major {
		unified = true
	}
	x := gocudnn.NewHandle()
	y, err := xtra.MakeHandle(xtrakernsfolder, dev, unified)
	if err != nil {
		panic(err)
	}

	return &Handler{
		cudnn:    x,
		xtra:     y,
		unified:  unified,
		maxbatch: 0,
		device:   dev,
	}
}

//SetStream sets the stream for the handles
func (h *Handler) SetStream(stream gocu.Streamer) error {
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
