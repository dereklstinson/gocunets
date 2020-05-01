package cudart

import (
	"errors"
	"fmt"

	"github.com/dereklstinson/gocudnn/gocu"
	"github.com/dereklstinson/cutil"
	"github.com/dereklstinson/half"

	"github.com/dereklstinson/gocunets/devices"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia"
)

//CudaSlice is cuda memory.
type CudaSlice struct {
	mem       *nvidia.Malloced
	dtype     devices.Type
	device    bool
	length    uint
	capacity  uint
	memcpyflg cutil.Mem
}

//MallocDevice allocates memory to set device
func mallocdevice(h nvidia.Handler, dtype devices.Type, length, capacity uint) (*CudaSlice, error) {

	mem, err := nvidia.MallocGlobal(h, capacity*dtype.SizeOf())
	if err != nil {
		return nil, err
	}
	return &CudaSlice{
		length:   length,
		capacity: capacity,
		mem:      mem,
		dtype:    dtype,
		device:   true,
	}, nil

}

//MallocCudaHost allocates paged host memory that is usable by cuda
func mallochost(h nvidia.Handler, dtype devices.Type, length, capacity uint) (*CudaSlice, error) {
	mem, err := nvidia.MallocHost(h, capacity*dtype.SizeOf())
	if err != nil {
		return nil, err
	}
	return &CudaSlice{
		length:   length,
		capacity: capacity,
		dtype:    dtype,
		mem:      mem,
	}, nil

}

//Append appends the slice
func (c *CudaSlice) Append(h nvidia.Handler, val interface{}) (*CudaSlice, error) {
	if devices.Len(val) <= c.capacity-c.length {
		offset := c.length
		c.length += devices.Len(val)
		c.Set(val, offset)
		return c, nil
	}
	newsize := c.length + devices.Len(val)
	newcudaslice, err := mallocdevice(h, c.dtype, c.length, newsize)
	if err != nil {
		return c, err
	}
	byteswritten, err := c.mem.WriteTo(newcudaslice.mem)
	//err = gocudnn.CudaMemCopyUnsafe(newcudaslice.mem.Ptr(), c.mem.Ptr(), gocudnn.SizeT(c.length*c.dtype.SizeOf().Uint()), c.memcpyflg.Default())
	if err != nil {
		fmt.Println("BytesWritten and NewSlice Size", byteswritten, newsize)
		return c, err
	}
	c = newcudaslice
	return newcudaslice.Append(h, val)

}

//Get will get values of a cudaslice and fill the slice
func (c *CudaSlice) Get(val interface{}, offset uint) error {
	length := devices.Len(val)
	if offset+length > c.length {

		return fmt.Errorf("Illegal Access SliceLength: %d, Offset: %d, ValLength: %d", c.length, offset, offset+length)

	}

	gptr, err := gocu.MakeGoMem(val)
	if err != nil {
		return err
	}
	destloc := c.mem.OffSet(offset * c.dtype.SizeOf())
	return nvidia.Memcpy(gptr, destloc, length*c.dtype.SizeOf())

}

//Type returns the devices.Type the cudaslice contains
func (c *CudaSlice) Type() devices.Type {
	return c.dtype
}

//Length returns the length of elements in the CudaSlice
func (c *CudaSlice) Length() uint {
	return c.length
}

//Set is an experimental function that will set a value or slice of values from host into cuda mem
func (c *CudaSlice) Set(val interface{}, offset uint) error {
	length := devices.Len(val)
	if offset+length > c.length {
		return fmt.Errorf("Illegal Access SliceLength: %d, Offset: %d, ValLength: %d", c.length, offset, offset+length)

	}

	gptr, err := gocu.MakeGoMem(val)
	if err != nil {
		return err
	}

	destloc := c.mem.OffSet(offset * c.dtype.SizeOf())
	return nvidia.Memcpy(destloc, gptr, length*c.dtype.SizeOf())

}

//Make is like the make function of golang pass []type{}, gotypes with the exception of device.FLoat16
//Then you define the length and the capacity.
func Make(h nvidia.Handler, x interface{}, args ...uint) (*CudaSlice, error) {

	switch len(args) {
	case 1:
		return make1(h, x, args...)
	case 2:
		return make2(h, x, args...)

	}

	return nil, errors.New("Unsupported Length of args")
}

func make1(h nvidia.Handler, x interface{}, args ...uint) (*CudaSlice, error) {

	switch x.(type) {
	case []uint8:
		return mallocdevice(h, devices.Uint8, args[0], args[0])
	case []int8:
		return mallocdevice(h, devices.Int8, args[0], args[0])
	case []uint16:
		return mallocdevice(h, devices.Uint16, args[0], args[0])
	case []int16:
		return mallocdevice(h, devices.Int16, args[0], args[0])
	case []uint32:
		return mallocdevice(h, devices.Uint32, args[0], args[0])
	case []int32:
		return mallocdevice(h, devices.Int32, args[0], args[0])
	case []uint64:
		return mallocdevice(h, devices.Uint64, args[0], args[0])
	case []int64:
		return mallocdevice(h, devices.Int64, args[0], args[0])
	case []uint:
		return mallocdevice(h, devices.Uint, args[0], args[0])
	case []int:
		return mallocdevice(h, devices.Int, args[0], args[0])

	case []half.Float16:
		return mallocdevice(h, devices.Float16H, args[0], args[0])
	case []float32:
		return mallocdevice(h, devices.Float32, args[0], args[0])
	case []float64:
		return mallocdevice(h, devices.Float64, args[0], args[0])

	}

	return nil, errors.New("Unsupported Type")
}

func make2(h nvidia.Handler, x interface{}, args ...uint) (*CudaSlice, error) {
	if args[1] < args[0] {
		return nil, errors.New("Length has to be less than Capacity")
	}
	switch x.(type) {
	case []uint8:
		return mallocdevice(h, devices.Uint8, args[0], args[1])
	case []int8:
		return mallocdevice(h, devices.Int8, args[0], args[1])
	case []uint16:
		return mallocdevice(h, devices.Uint16, args[0], args[1])
	case []int16:
		return mallocdevice(h, devices.Int16, args[0], args[1])
	case []uint32:
		return mallocdevice(h, devices.Uint32, args[0], args[1])
	case []int32:
		return mallocdevice(h, devices.Int32, args[0], args[1])
	case []uint64:
		return mallocdevice(h, devices.Uint64, args[0], args[1])
	case []int64:
		return mallocdevice(h, devices.Int64, args[0], args[1])
	case []uint:
		return mallocdevice(h, devices.Uint, args[0], args[1])
	case []int:
		return mallocdevice(h, devices.Int, args[0], args[1])

	case []half.Float16:
		return mallocdevice(h, devices.Float16H, args[0], args[1])
	case []float32:
		return mallocdevice(h, devices.Float32, args[0], args[1])
	case []float64:
		return mallocdevice(h, devices.Float64, args[0], args[1])

	}

	return nil, errors.New("Unsupported Type")
}
