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
	"github.com/dereklstinson/GoCudnn/curand"
	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/GoCudnn/xtra"
)

//Handler contains the handles used in gocudnn and also the xtra kernals.
type Handler struct {
	*gocu.Worker
	cudnn   *gocudnn.Handle
	xtra    *xtra.Handle
	stream  gocu.Streamer
	unified bool
	device  cudart.Device
	rngtype curand.RngType
	curng   *curand.Generator
	seed    uint
}

//FindVol will find the max vol for tensor.  This is going to hold two functions
//pmax can be either the previous maxvol, or it could be
func (h *Handler) FindVol(outputdims []int32) int32 {
	return utils.FindVolumeInt32(outputdims, nil)
}

/*
//FindMaxUint returns the max sizeT
func (h *Handler) FindSIB(outputdims []int32) uint {

	return uint(utils.FindVolumeInt32(outputdims, nil) * 4)

}
*/

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
	if h.stream == nil {
		var err error
		h.stream, err = cudart.CreateNonBlockingStream()
		if err != nil {
			panic(err)
		}
	}
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

//SetDevice sets the device the handler is using
func (h *Handler) SetDevice() error {
	return h.device.Set()
}

//GetCuRNG returns the curand.Generator Handler is holding
func (h *Handler) GetCuRNG() *curand.Generator {
	return h.curng
}

//CreateHandler creates a the handlers
//The handler is used in managing memory for all the packages that use cudnn.Handler. This function will raise a flag that will tell the program
//to use unified memory management.  If that is not wanted call MakeNotUnified immediately to turn this off.
func CreateHandler(w *gocu.Worker, dev cudart.Device, seed uint64) (h *Handler) {

	h = new(Handler)
	h.rngtype.PseudoDefault()
	h.Worker = w
	h.device = dev
	var unified bool
	major, err := dev.Major()
	if err != nil {
		panic(err)
	}
	if 6 < major {
		unified = true
	}
	h.unified = unified
	if w != nil {
		h.cudnn = gocudnn.CreateHandleEX(h.Worker, true)
		h.curng = curand.CreateGeneratorEx(h.Worker, h.rngtype)
		h.xtra, err = xtra.MakeHandleEx(h.Worker, unified)
		if err != nil {
			panic(err)
		}
		err = h.curng.SetPsuedoSeed(seed)
		if err != nil {
			panic(err)
		}
	} else {
		h.cudnn = gocudnn.CreateHandle(true)
		h.curng = curand.CreateGenerator(h.rngtype)
		h.xtra, err = xtra.MakeHandle(dev, unified)
		if err != nil {
			panic(err)
		}
		err = h.curng.SetPsuedoSeed(seed)
		if err != nil {
			panic(err)
		}
	}

	return h
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
	err = h.curng.SetStream(stream)
	if err != nil {
		return err
	}
	h.stream = stream
	return nil
}
