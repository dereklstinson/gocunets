package gomem

import "errors"

type HArrayNd struct {
	Data []float32
	Dims []int
}

func NewHArrayND(dims []int) (*HArrayNd, error) {
	if len(dims) > 8 {
		return nil, errors.New("Too many Dims")
	}
	mult := 1
	for i := 0; i < len(dims); i++ {
		mult *= dims[i]
	}
	data := make([]float32, mult)
	return &HArrayNd{
		Data: data,
		Dims: dims,
	}, nil
}
func (h *HArrayNd) NumberofDims() int {
	return len(h.Dims)
}
