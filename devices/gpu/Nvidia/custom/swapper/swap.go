package swapper

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/Nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/Nvidia/cudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Swapper holds kernels that do swap functions
type Swapper struct {
	swap *gocudnn.Swapper
}

//CreateSwapper returns a Swapper
func CreateSwapper(handle *cudnn.Handler) (*Swapper, error) {
	swap, err := gocudnn.Xtra{}.NewBatchSwapper(handle.XHandle())
	if err != nil {
		return nil, err
	}
	return &Swapper{
		swap: swap,
	}, nil
}

//EveryOther swaps either the even or odd of every other batch between two tensors supports float and int8
func (s *Swapper) EveryOther(handle *cudnn.Handler, A, B *tensor.Volume, start, stride int32) error {

	return s.swap.EveryOther(handle.XHandle(), A.TD(), A.Memer(), B.TD(), B.Memer(), start, stride)
}

//UpperLower swaps either the upper or lower half of the batches between to tensors  supports float and int8
func (s *Swapper) UpperLower(handle *cudnn.Handler, A, B *tensor.Volume, Aupper, Bupper, inverse bool) error {
	return s.swap.UpperLower(handle.XHandle(), A.TD(), A.Memer(), B.TD(), B.Memer(), Aupper, Bupper, inverse)
}

/*
//InnerUpperLower swaps either the upper and lower batches of a single tensor inverse will start at top and bottom to do a swap.
//not inverse it will start at top and middle of batches to do the sap
func (s *Swapper) InnerUpperLower(handle *cudnn.Handler, A *tensor.Volume, inverse bool) error {
	return s.swap.InnerUpperLower(handle.XHandle(), A.TD(), A.Memer(), inverse)
}

//InnerBatch swill swap two batches of the inside of a tensor
func (s *Swapper) InnerBatch(handle *cudnn.Handler, A *tensor.Volume, batcha, batchb int32) error {
	return s.swap.InnerBatch(handle.XHandle(), A.TD(), A.Memer(), batcha, batchb)
}
*/
