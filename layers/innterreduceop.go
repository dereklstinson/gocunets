package layers

import (
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn/reduce"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn/tensor"
	gocudnn "github.com/dereklstinson/gocudnn"
	"github.com/dereklstinson/gocudnn/gocu"
)

type reduceop struct {
	op          *reduce.Ops
	mem         *tensor.Volume
	gptr        *gocu.Wrapper
	indicies    *nvidia.Malloced
	wspace      *nvidia.Malloced
	val         []float32
	alpha, beta float64
	unified     bool
}

func buildminreduce(handle *cudnn.Handler, iomem *tensor.Volume) (*reduceop, error) {
	rflg := reduce.Flags
	return genericbuildreduceop(handle, rflg.ReduceMode.Min(), iomem)
}
func buildmaxreduce(handle *cudnn.Handler, iomem *tensor.Volume) (*reduceop, error) {
	rflg := reduce.Flags
	return genericbuildreduceop(handle, rflg.ReduceMode.Max(), iomem)
}
func buildavgreduce(handle *cudnn.Handler, iomem *tensor.Volume) (*reduceop, error) {
	rflg := reduce.Flags
	return genericbuildreduceop(handle, rflg.ReduceMode.Avg(), iomem)
}
func buildnorm1reduce(handle *cudnn.Handler, iomem *tensor.Volume) (*reduceop, error) {
	rflg := reduce.Flags
	return genericbuildreduceop(handle, rflg.ReduceMode.Norm1(), iomem)
}
func buildnorm2reduce(handle *cudnn.Handler, iomem *tensor.Volume) (*reduceop, error) {
	rflg := reduce.Flags
	return genericbuildreduceop(handle, rflg.ReduceMode.Norm2(), iomem)
}

func genericbuildreduceop(handle *cudnn.Handler, mode gocudnn.ReduceTensorOp, iomem *tensor.Volume) (*reduceop, error) {
	rflg := reduce.Flags
	frmt, dtype, dims, err := iomem.Properties()
	if err != nil {
		return nil, err
	}

	reducedims := make([]int32, len(dims))

	for i := 0; i < len(reducedims); i++ {
		reducedims[i] = 1
	}

	op, err := reduce.Stage(mode, dtype, rflg.NanProp.NotPropigate(), rflg.IndFlag.NoIndices(), rflg.IndType.Type32Bit())
	if err != nil {
		return nil, err
	}
	mem, err := tensor.Build(handle, frmt, dtype, reducedims)
	if err != nil {
		return nil, err
	}
	wspacesize, err := op.GetWorkSpaceSize(handle, iomem, mem)
	if err != nil {
		return nil, err
	}
	val := make([]float32, dims[0])
	gpr, err := gocu.MakeGoMem(val)
	if err != nil {
		return nil, err
	}

	wspace, err := nvidia.MallocGlobal(handle, wspacesize)
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

/*
func buildreduceop(handle *cudnn.Handler, min bool, iomem *tensor.Volume) (*reduceop, error) {
	frmt, dtype, dims, err := iomem.Properties()
	managed := handle.Unified()
	if err != nil {
		return nil, err
	}
	reducedims := make([]int32, len(dims))

	for i := 0; i < len(reducedims); i++ {
		reducedims[i] = 1
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
	mem, err := tensor.Build(handle, frmt, dtype, reducedims)
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
*/
func checkifones(x []int32) bool {
	for i := range x {
		if x[i] != 1 {
			return false
		}
	}
	return true
}
func (r *reduceop) Reduce(handle *cudnn.Handler, x *tensor.Volume) (float32, error) {
	if checkifones(x.Dims()) {

		bsize := x.SIB()
		err := nvidia.Memcpy(r.gptr, x, bsize)
		if err != nil {
			return 0, err
		}
		return r.val[0], nil

	}

	err := r.reduce(handle, x)
	if err != nil {
		return 0, err
	}

	bsize := x.SIB()
	err = nvidia.Memcpy(r.gptr, x, bsize)
	if err != nil {
		return 0, err
	}
	return r.val[0], nil
}
func (r *reduceop) reduce(handle *cudnn.Handler, x *tensor.Volume) error {
	return r.op.Reduce(handle, r.indicies, r.wspace, r.alpha, x, r.beta, r.mem)
}
