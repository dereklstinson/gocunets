package cudnn

import (
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Handler contains the handles used in gocudnn and also the xtra kernals.
type Handler struct {
	cudnn  *gocudnn.Handle
	xtra   *gocudnn.XHandle
	stream *gocudnn.Stream
}

//Cudnn returns a pointer to the cudnn handle
func (h *Handler) Cudnn() *gocudnn.Handle {
	return h.cudnn
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
func CreateHandler(dev *gocudnn.Device, xtrakernsfolder string) *Handler {
	err := dev.Set()
	if err != nil {
		panic(err)
	}
	x := gocudnn.NewHandle()
	y, err := gocudnn.Xtra{}.MakeXHandle(xtrakernsfolder, dev)
	if err != nil {
		panic(err)
	}

	return &Handler{
		cudnn: x,
		xtra:  y,
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
