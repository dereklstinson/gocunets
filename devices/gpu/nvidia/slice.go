package nvidia

import (
	"github.com/dereklstinson/GoCudnn/gocu"
)

//Slice is a slice of memory on an nvidia gpu
type Slice struct {
	ptr      gocu.Mem
	len      int
	unitsize int
}

func (s *Slice) Mem() gocu.Mem {
	return s.ptr
}
