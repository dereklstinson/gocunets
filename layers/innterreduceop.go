package layers

import (
	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/reduce"
	"github.com/dereklstinson/GoCuNets/cudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

type reduceop struct {
	op          *reduce.Ops
	mem         *tensor.Volume
	gptr        *gocudnn.GoPointer
	indicies    *gocudnn.Malloced
	wspace      *gocudnn.Malloced
	val         []float32
	alpha, beta float64
	unified     bool
}

func buildminreduce(handle *cudnn.Handler, iomem *tensor.Volume, batches bool) (*reduceop, error) {
	rflg := reduce.Flags
	return genericbuildreduceop(handle, rflg.ReduceMode.Min(), iomem, batches)
}
func buildmaxreduce(handle *cudnn.Handler, iomem *tensor.Volume, batches bool) (*reduceop, error) {
	rflg := reduce.Flags
	return genericbuildreduceop(handle, rflg.ReduceMode.Max(), iomem, batches)
}
func buildavgreduce(handle *cudnn.Handler, iomem *tensor.Volume, batches bool) (*reduceop, error) {
	rflg := reduce.Flags
	return genericbuildreduceop(handle, rflg.ReduceMode.Avg(), iomem, batches)
}
func buildnorm1reduce(handle *cudnn.Handler, iomem *tensor.Volume, batches bool) (*reduceop, error) {
	rflg := reduce.Flags
	return genericbuildreduceop(handle, rflg.ReduceMode.Norm1(), iomem, batches)
}
func buildnorm2reduce(handle *cudnn.Handler, iomem *tensor.Volume, batches bool) (*reduceop, error) {
	rflg := reduce.Flags
	return genericbuildreduceop(handle, rflg.ReduceMode.Norm2(), iomem, batches)
}

func genericbuildreduceop(handle *cudnn.Handler, mode reduce.OpMode, iomem *tensor.Volume, batches bool) (*reduceop, error) {
	rflg := reduce.Flags
	frmt, dtype, dims, err := iomem.Properties()
	managed := iomem.Unified()
	if err != nil {
		return nil, err
	}
	reducedims := make([]int32, len(dims))

	for i := 0; i < len(reducedims); i++ {
		reducedims[i] = 1
	}
	if batches == true {
		reducedims[0] = dims[0]
	}
	op, err := reduce.Stage(mode, dtype, rflg.NanProp.NoPropNAN(), rflg.IndFlag.NoIndices(), rflg.IndType.Type32Bit())
	if err != nil {
		return nil, err
	}
	mem, err := tensor.Build(frmt, dtype, reducedims, managed)
	if err != nil {
		return nil, err
	}
	wspacesize, err := op.GetWorkSpaceSize(handle, iomem, mem)
	if err != nil {
		return nil, err
	}
	val := make([]float32, dims[0])
	gpr, err := gocudnn.MakeGoPointer(val)
	if err != nil {
		return nil, err
	}
	if managed {
		wspace, err := gocudnn.UnifiedMangedGlobal(wspacesize)
		if err != nil {
			return nil, err
		}
		return &reduceop{
			op:     op,
			mem:    mem,
			wspace: wspace,
			alpha:  1,
			beta:   0,
			val:    val,
			gptr:   gpr,
		}, nil
	}
	wspace, err := gocudnn.Malloc(wspacesize)
	if err != nil {
		return nil, err
	}

	return &reduceop{
		op:     op,
		mem:    mem,
		wspace: wspace,
		alpha:  1,
		beta:   0,
		val:    val,
		gptr:   gpr,
	}, nil
}
func buildreduceop(handle *cudnn.Handler, min bool, iomem *tensor.Volume, batches bool) (*reduceop, error) {
	frmt, dtype, dims, err := iomem.Properties()
	managed := iomem.Unified()
	if err != nil {
		return nil, err
	}
	reducedims := make([]int32, len(dims))

	for i := 0; i < len(reducedims); i++ {
		reducedims[i] = 1
	}
	if batches == true {
		reducedims[0] = dims[0]
	}

	rflg := reduce.Flags
	var opmode reduce.OpMode
	if min == true {
		opmode = rflg.ReduceMode.Min()
	} else {
		opmode = rflg.ReduceMode.Max()
	}
	op, err := reduce.Stage(opmode, dtype, rflg.NanProp.NoPropNAN(), rflg.IndFlag.NoIndices(), rflg.IndType.Type32Bit())
	if err != nil {
		return nil, err
	}
	mem, err := tensor.Build(frmt, dtype, reducedims, managed)
	if err != nil {
		return nil, err
	}
	wspacesize, err := op.GetWorkSpaceSize(handle, iomem, mem)
	if err != nil {
		return nil, err
	}
	val := make([]float32, dims[0])
	gpr, err := gocudnn.MakeGoPointer(val)
	if err != nil {
		return nil, err
	}
	if managed {
		wspace, err := gocudnn.UnifiedMangedGlobal(wspacesize)
		if err != nil {
			return nil, err
		}
		return &reduceop{
			op:     op,
			mem:    mem,
			wspace: wspace,
			alpha:  1,
			beta:   0,
			val:    val,
			gptr:   gpr,
		}, nil
	}
	wspace, err := gocudnn.Malloc(wspacesize)
	if err != nil {
		return nil, err
	}

	return &reduceop{
		op:     op,
		mem:    mem,
		wspace: wspace,
		alpha:  1,
		beta:   0,
		val:    val,
		gptr:   gpr,
	}, nil
}
func (r *reduceop) Reduce(handle *cudnn.Handler, x *tensor.Volume) ([]float32, error) {
	err := r.reduce(handle, x)
	if err != nil {
		return nil, err
	}
	if r.unified == true {
		err = gocudnn.UnifiedMemCopy(r.gptr, r.mem.Memer())
		if err != nil {
			return nil, err
		}
		return r.val, nil
	}
	bsize := r.mem.Memer().ByteSize()
	err = gocudnn.CudaMemCopy(r.gptr, r.mem.Memer(), bsize, gocudnn.MemcpyKindFlag{}.DeviceToHost())
	if err != nil {
		return nil, err
	}
	return r.val, nil
}
func (r *reduceop) reduce(handle *cudnn.Handler, x *tensor.Volume) error {
	return r.op.Reduce(handle, r.indicies, r.wspace, r.alpha, x, r.beta, r.mem)
}
