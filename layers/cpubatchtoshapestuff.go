package layers

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/cpu"
	"github.com/dereklstinson/GoCuNets/cudnn"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//ShapetoBatchIOBWDCPU takes the batched IO that was created from the fwd process and replaces the delta Tensor values
func (i *IO) ShapetoBatchIOBWDCPU(handle *cudnn.Handler, batched *IO, stride []int32) error {
	var fmtflag cudnn.TensorFormatFlag
	frmt, _, dimsy, err := batched.Properties()

	if err != nil {
		return err
	}
	_, _, dimsx, err := i.Properties()
	if err != nil {
		return err
	}
	blength, err := batched.GetLength()
	if err != nil {
		return err
	}
	ilength, err := i.GetLength()
	if err != nil {
		return err
	}

	dx := make([]float32, ilength)

	dy := make([]float32, blength)
	ptrdx, err := gocudnn.MakeGoPointer(dx)
	if err != nil {
		return err
	}
	ptrdy, err := gocudnn.MakeGoPointer(dy)
	if err != nil {
		return err
	}
	sizetdy := batched.DeltaT().CurrentSizeT()

	if handle.Unified() == true {
		gocudnn.CudaMemCopy(ptrdy, batched.dx.Memer(), sizetdy.Cu(), gocudnn.MemcpyKindFlag{}.Default())
	} else {
		gocudnn.CudaMemCopy(ptrdy, batched.dx.Memer(), sizetdy.Cu(), gocudnn.MemcpyKindFlag{}.DeviceToHost())
	}

	if frmt == fmtflag.NCHW() {
		err = cpu.ShapeToBatchNCHW4DBackward(dx, dimsx, dy, dimsy, stride)
		if err != nil {
			return err
		}
		return i.LoadDeltaTValues(handle, ptrdx)

	} else if frmt == fmtflag.NHWC() {
		err = cpu.ShapeToBatchNHWC4DBackward(dx, dimsx, dy, dimsy, stride)
		if err != nil {
			return err
		}
		return i.LoadDeltaTValues(handle, ptrdx)
	}
	return errors.New("Unsupported Format")
}

/*
//ShapetoBatchIOCopyFWDCPU reshapes the makes a reshaped copy of the IO
func (i *IO) ShapetoBatchIOCopyFWDCPU(handle *cudnn.Handler,window []int32, stride []int32) (*IO, error) {
	frmt, dtype, dims, err := i.Properties()
	if err != nil {
		return nil, err
	}
	length, err := i.GetLength()
	if err != nil {
		return nil, err
	}
	slice := make([]float32, length)
	i.T().Memer().FillSlice(slice)
	var fmtflag cudnn.TensorFormatFlag
	if frmt == fmtflag.NCHW() {
		reshapedslice, rashapeddims, err := cpu.ShapeToBatchNCHW4DForward(slice, dims, window, stride)
		if err != nil {
			return nil, err
		}
		newIO, err := BuildIO(frmt, dtype, rashapeddims, i.MemIsManaged())
		if err != nil {
			return nil, err
		}
		goptr, err := gocudnn.MakeGoPointer(reshapedslice)
		if err != nil {
			return nil, err
		}
		err = newIO.LoadTValues(goptr)
		return newIO, err
	}

	reshapedslice, rashapeddims, err := cpu.ShapeToBatchNHWC4DForward(slice, dims, window, stride)
	if err != nil {
		return nil, err
	}
	newIO, err := BuildIO(frmt, dtype, rashapeddims, i.MemIsManaged())
	if err != nil {
		return nil, err
	}
	goptr, err := gocudnn.MakeGoPointer(reshapedslice)
	if err != nil {
		return nil, err
	}
	err = newIO.LoadTValues(goptr)
	return newIO, err
}
*/
/*
//ShapetoBatchIOCopyCPUWithSliceFloat32 reshapes the makes a reshaped copy of the IO
func (i *IO) ShapetoBatchIOCopyCPUWithSliceFloat32(window, stride []int32) (*IO, []float32, error) {
	frmt, dtype, dims, err := i.Properties()
	if err != nil {
		return nil, nil, err
	}
	length, err := i.GetLength()
	if err != nil {
		return nil, nil, err
	}
	slice := make([]float32, length)
	i.T().Memer().FillSlice(slice)
	var fmtflag cudnn.TensorFormatFlag
	if frmt == fmtflag.NCHW() {
		reshapedslice, rashapeddims, err := cpu.ShapeToBatchNCHW4DForward(slice, dims, window, stride)
		if err != nil {
			return nil, nil, err
		}
		newIO, err := BuildIO(frmt, dtype, rashapeddims, i.MemIsManaged())
		if err != nil {
			return nil, nil, err
		}
		goptr, err := gocudnn.MakeGoPointer(reshapedslice)
		if err != nil {
			return nil, nil, err
		}
		err = newIO.LoadTValues(goptr)
		return newIO, reshapedslice, err
	}

	reshapedslice, rashapeddims, err := cpu.ShapeToBatchNHWC4DForward(slice, dims, window, stride)
	if err != nil {
		return nil, nil, err
	}
	newIO, err := BuildIO(frmt, dtype, rashapeddims, i.MemIsManaged())
	if err != nil {
		return nil, nil, err
	}
	goptr, err := gocudnn.MakeGoPointer(reshapedslice)
	if err != nil {
		return nil, nil, err
	}
	err = newIO.LoadTValues(goptr)
	return newIO, reshapedslice, err
}
*/

/*
//LoadMem Replaces The memory on the device.
func (i *IO) LoadMem(mem gocudnn.Memer, kind gocudnn.MemcpyKind) error {
	size, err := i.desc.GetSizeInBytes()
	if err != nil {
		return err
	}
	if size != mem.ByteSize() {
		return errors.New("Memory Size doesn't Match Descriptor")
	}
	gocudnn.CudaMemCopy(i.mem, mem, size, kind)
	return nil
}
*/
