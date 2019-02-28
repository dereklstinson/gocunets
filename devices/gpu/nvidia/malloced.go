package nvidia

import (
	"fmt"
	"io"
	"unsafe"

	"github.com/dereklstinson/GoCuNets/devices"
	"github.com/dereklstinson/GoCudnn/cudart"
	"github.com/dereklstinson/GoCudnn/gocu"
)

//Handler is a gpu device handler that can set the device and tell if the memory is under the nvidia unified memory management
type Handler interface {
	SetDevice() error //
}

//Malloced is a pointer to some nvidia memory
type Malloced struct {
	ptr       unsafe.Pointer
	unitbytes uint
	unitlen   uint
	host      bool
}

const defaultmemcopykind = cudart.MemcpyKind(4) //enum of 4 is the default memcopy kind

//Ptr is an unsafe pointer to nvidia memory
func (m *Malloced) Ptr() unsafe.Pointer {
	return m.ptr
}

//DPtr is a double pointer to nvidia device memory
func (m *Malloced) DPtr() *unsafe.Pointer {
	return &m.ptr
}

//OffSet returns the offset of the nvidia memory
func (m *Malloced) OffSet(byunits uint) *Malloced {
	offset := unsafe.Pointer(uintptr(m.ptr) + uintptr(byunits*m.unitbytes))
	return &Malloced{
		ptr:       offset,
		unitlen:   m.unitlen - byunits,
		unitbytes: m.unitbytes,
	}
}

//TotalBytes returns the total bytes the malloced has
func (m *Malloced) TotalBytes() uint {
	return m.unitbytes * m.unitlen
}

//MallocHost allocates memory onto the host used by nvidia devices.
//Handler will set the device it is allocating to. Besure to set back if wanting to use another device
func MallocHost(h Handler, sizebytes uint, dt devices.Type) (*Malloced, error) {
	err := h.SetDevice()
	if err != nil {
		return nil, err
	}

	x := new(Malloced)
	x.unitbytes = dt.SizeOf().Uint()
	x.unitlen = sizebytes / dt.SizeOf().Uint()
	x.host = true
	err = cudart.MallocManagedHost(x, sizebytes)
	if err != nil {
		return nil, err
	}

	return x, nil

}

//Memcpy is like cudart.Memcpy but it is using the cudart.Memcpykind{}.Default() flag
func Memcpy(dest, src gocu.Mem, sizeinbytes uint) error {
	return cudart.MemCpy(dest, src, sizeinbytes, defaultmemcopykind)
}

//MallocGlobal allocates memory to the nvidia gpu
//Handler will set the device it is allocating to. Besure to set back if wanting to use another device
func MallocGlobal(h Handler, sizebytes uint, dt devices.Type) (*Malloced, error) {
	err := h.SetDevice()
	if err != nil {
		return nil, err
	}

	x := new(Malloced)
	x.unitbytes = dt.SizeOf().Uint()
	x.unitlen = sizebytes / dt.SizeOf().Uint()
	x.host = true
	err = cudart.MallocManagedGlobal(x, sizebytes)
	if err != nil {
		return nil, err
	}

	return x, nil
}

func (m *Malloced) Write(p []byte) (int, error) {
	gomem, err := gocu.MakeGoMem(p)
	if err != nil {
		return 0, err
	}
	var errholder error
	size := uint(len(p))
	if (m.TotalBytes()) < size {
		size = m.TotalBytes()
		errholder = fmt.Errorf("Total Size of Malloced is %d < bytes passed %d.  Still wrote to Nvidia Device", size, len(p))
	}
	err = cudart.MemCpy(m, gomem, size, defaultmemcopykind)
	if err != nil {
		return 0, err
	}
	return int(size), errholder

}

func (m *Malloced) Read(p []byte) (n int, err error) {
	gomem, err := gocu.MakeGoMem(p)
	if err != nil {
		return 0, err
	}
	var errholder error
	size := uint(len(p))
	if (m.TotalBytes()) < size {
		size = m.TotalBytes()
		errholder = fmt.Errorf("Total Size of Malloced is %d < bytes passed %d.  Still read from Nvidia Device", size, len(p))
	}
	err = cudart.MemCpy(gomem, m, size, defaultmemcopykind)
	if err != nil {
		return 0, err
	}
	return int(size), errholder
}

//WriteTo writes data to w until there is no more data to write or when an error occures. If w is a *Malloced then it will do a Memcpy
//The return value n is the number of bytes written. Any error encountered during the write is also returned
func (m *Malloced) WriteTo(w io.Writer) (n int64, err error) {
	var size = m.TotalBytes()
	var errholder error
	wmalloced, ok := w.(*Malloced)
	if ok {

		if wmalloced.TotalBytes() < size {
			size = wmalloced.TotalBytes()
			errholder = fmt.Errorf("Total Size of Malloced is %d. Than MallocedWriter's passed bytes.  Still read from Nvidia Device", size, wmalloced.TotalBytes())
		}
		err = Memcpy(wmalloced, m, size)
		if err != nil {
			return 0, err
		}
		err = errholder
		return (int64)(size), err
	}
	somebytes := make([]byte, m.TotalBytes())
	bytesread, err := m.Read(somebytes)
	if err != nil {
		return 0, errholder
	}
	totalwritten, err := w.Write(somebytes[:bytesread])
	return (int64)(totalwritten), err

}
