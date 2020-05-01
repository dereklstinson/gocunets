package nvidia

import (
	"unsafe"

	"github.com/dereklstinson/gocudnn/cudart"
	"github.com/dereklstinson/gocudnn/cudart/crtutil"
	"github.com/dereklstinson/gocudnn/gocu"
	"github.com/dereklstinson/cutil"
)

//Malloced is a pointer to some nvidia memory
type Malloced struct {
	ptr      unsafe.Pointer
	numbytes uint
	host     bool
}
type reader struct {
	m       *Malloced
	counter uint
}
type Worker interface {
	Work(func() error) error
}

const defaultmemcopykind = cudart.MemcpyKind(4) //enum of 4 is the default memcopy kind

//Ptr is an unsafe pointer to nvidia memory
func (m *Malloced) Ptr() unsafe.Pointer {
	if m == nil {
		return nil
	}
	return m.ptr
}

//DPtr is a double pointer to nvidia device memory
func (m *Malloced) DPtr() *unsafe.Pointer {
	if m == nil {
		return nil
	}
	return &m.ptr
}

//OffSet returns the offset of the nvidia memory
func (m *Malloced) OffSet(bybytes uint) *Malloced {
	if m.numbytes-bybytes < 1 {
		return nil
	}
	offset := unsafe.Pointer(uintptr(m.ptr) + uintptr(bybytes))

	return &Malloced{
		ptr:      offset,
		numbytes: m.numbytes - bybytes,
		host:     m.host,
	}
}

//SIB returns the size in bytes
func (m *Malloced) SIB() uint {
	if m == nil {
		return 0
	}
	return m.numbytes
}

//NewReadWriter creates a devio.Buffer for the malloced memory.
//If s is nil then copies will be synced if not then copies will be async
func (m *Malloced) NewReadWriter(s gocu.Streamer) *crtutil.ReadWriter {
	return crtutil.NewReadWriter(m, m.numbytes, s)
}

//MallocHost allocates memory onto the host used by nvidia devices.
//Handler will set the device it is allocating to. Besure to set back if wanting to use another device
func MallocHost(w Worker, sizebytes uint) (x *Malloced, err error) {
	x = new(Malloced)
	x.numbytes = sizebytes
	x.host = true
	if w == nil {
		return nil, cudart.MallocManagedGlobal(x, sizebytes)
	}
	err = w.Work(func() error {
		return cudart.MallocManagedHost(x, sizebytes)
	})
	if err != nil {
		return nil, err
	}

	return x, nil

}

type copier struct {
	async bool
	s     gocu.Streamer
}

func (c copier) CopyHostToDevice(dest, src cutil.Pointer, sib uint) error {
	if c.s != nil {
		return cudart.MemcpyAsync(dest, src, sib, defaultmemcopykind, c.s)
	}
	return cudart.Memcpy(dest, src, sib, defaultmemcopykind)
}
func (c copier) CopyDeviceToHost(dest, src cutil.Pointer, sib uint) error {
	if c.s != nil {
		return cudart.MemcpyAsync(dest, src, sib, defaultmemcopykind, c.s)
	}
	return cudart.Memcpy(dest, src, sib, defaultmemcopykind)
}
func (c copier) Sync() error {
	if c.s != nil {
		return c.s.Sync()
	}
	return nil
}

//Memcpy is like cudart.Memcpy but it is using the cudart.Memcpykind{}.Default() flag
func Memcpy(dest, src cutil.Pointer, sizeinbytes uint) error {
	//	if w != nil {
	//		return w.Work(func() error {
	//			return cudart.MemCpy(dest, src, sizeinbytes, defaultmemcopykind)
	//		})
	//	}
	return cudart.Memcpy(dest, src, sizeinbytes, defaultmemcopykind)
}

//SetAll sets the memory to whatever integer value passed
func (m *Malloced) SetAll(val int32) error {
	//if w != nil {
	//	return w.Work(func() error {
	//		return cudart.Memset(m, val, m.numbytes)
	//	})
	//}
	return cudart.Memset(m, val, m.numbytes)
}

//MallocGlobal allocates memory to the nvidia gpu
//Handler will set the device it is allocating to. Besure to set back if wanting to use another device
func MallocGlobal(w Worker, sizebytes uint) (x *Malloced, err error) {
	if w == nil {
		x = new(Malloced)
		x.numbytes = sizebytes
		err = cudart.MallocManagedGlobal(x, sizebytes)
		return x, err
	}
	err = w.Work(func() error {
		x = new(Malloced)
		x.numbytes = sizebytes
		return cudart.MallocManagedGlobal(x, sizebytes)
	})
	if err != nil {
		return nil, err
	}

	return x, nil
}
