package nvidia

import (
	"errors"
	"reflect"
	"unsafe"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"

	"github.com/dereklstinson/GoCudnn/cudart"
)

//FillSlice will fill a slice that was passed in the input
func (m *Malloced) FillSlice(handle *cudnn.Handler, input interface{}) error {
	if handle.Worker == nil {
		val := reflect.ValueOf(input)
		ptr := unsafe.Pointer(val.Pointer())
		if ptr == nil {
			return errors.New("Nil sent as input for FillSlice")
		}

		return cudart.MemcpyUS(ptr, m.Ptr(), m.numbytes, defaultmemcopykind)
	}
	return handle.Work(func() error {
		val := reflect.ValueOf(input)
		ptr := unsafe.Pointer(val.Pointer())
		if ptr == nil {
			return errors.New("Nil sent as input for FillSlice")
		}

		return cudart.MemcpyUS(ptr, m.Ptr(), m.numbytes, defaultmemcopykind)
	})
}
