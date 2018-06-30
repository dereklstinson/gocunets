package tensor

import (
	"errors"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Tensor is a struct that holds a the necessary Tensors and Memory.
type Tensor struct {
	tD  *gocudnn.TensorD
	fD  *gocudnn.TensorD
	mem gocudnn.Memer
}

func (t *Tensor) ZeroClone() (*Tensor, error) {
	if t.tD == nil || t.fD == nil || t.mem == nil {
		return nil, errors.New("Tensor is nil")
	}
	t.tD.GetDescrptor()
	return &Tensor{}, nil
}
func (t *Tensor) SetAll(input float64) {

}

//func (t *Tensor) AddAll()
