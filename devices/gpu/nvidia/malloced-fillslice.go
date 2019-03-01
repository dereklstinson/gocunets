package nvidia

import (
	"errors"
	"reflect"
	"unsafe"

	"github.com/dereklstinson/GoCudnn/cudart"
)

//FillSlice will fill a slice that was passed in the input
func (m *Malloced) FillSlice(input interface{}) error {
	val := reflect.ValueOf(input)

	ptr := unsafe.Pointer(val.Pointer())
	if ptr == nil {
		return errors.New("Nil sent as input for FillSlice")
	}

	return cudart.MemcpyUnsafe(ptr, m.Ptr(), m.TotalBytes(), defaultmemcopykind)

}
