package cudnn

import (
	"errors"

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
		return errors.New("handlers stream is not set")
	}
	return h.stream.Sync()
}

//CreateHandler creates a the handlers
func CreateHandler(dev *gocudnn.Device, xtrakernsfolder string, stream *gocudnn.Stream) *Handler {

	x := gocudnn.NewHandle()
	y, err := gocudnn.Xtra{}.MakeXHandle(xtrakernsfolder, dev)
	if err != nil {
		panic(err)
	}
	x.SetStream(stream)
	if err != nil {
		panic(err)
	}
	return &Handler{
		cudnn:  x,
		xtra:   y,
		stream: stream,
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
