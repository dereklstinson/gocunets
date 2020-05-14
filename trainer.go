package gocunets

import (
	"errors"
	"fmt"

	"github.com/dereklstinson/gocunets/trainer"
)

//TrainerHandler handles creating trainers and the updating and weight update, and probably some other things down the road.
type TrainerHandler struct {
	t     []trainer.Trainer
	b     *Builder
	w     []*Tensor
	dw    []*Tensor
	batch int32
}

func CreateTrainerHandler(b *Builder) (t *TrainerHandler) {

	t = new(TrainerHandler)
	t.b = b
	return t
}
func (t *TrainerHandler) GetWeights() []*Tensor {
	return t.w

}
func (t *TrainerHandler) GetDeltaWeights() []*Tensor {
	return t.dw
}

//SetupAdamTrainers sets adam trainers in thr trainerhandler and returns a slice of adam trainers.  This can be used to fine tune the trainers for each layer.
//Otherwise, it can be ignored.
func (t *TrainerHandler) SetupAdamTrainers(weights, deltaweights []*Tensor, learningrate, decay1, decay2 float32, batch int32) (a []*trainer.Adam, err error) {
	if len(weights) == 0 {
		return nil, errors.New("Len of weights is 0")
	}
	if len(weights) != len(deltaweights) {
		return nil, fmt.Errorf("(b *Builder)CreateAdamTrainerHandler: %s", "len(weights!=len(deltaweights)")
	}
	t.dw = deltaweights
	t.w = weights
	a = make([]*trainer.Adam, len(weights))
	t.t = make([]trainer.Trainer, len(weights))
	t.batch = batch
	for i := range weights {
		a[i], err = trainer.SetupAdam(t.b.h.Handler.XHandle(), decay1, decay2, batch)

		if err != nil {
			return nil, err
		}
		a[i].SetRates(learningrate, 0)
		err = a[i].SetTrainingMem(t.b.h.Handler, weights[i].Tensor)
		if err != nil {
			return nil, err
		}

		t.t[i] = a[i]

	}

	return a, nil
}
func (t *TrainerHandler) ChangeBatchSize(batchsize int32) {
	t.batch = batchsize
}
func (t *TrainerHandler) UpdateWeights() (err error) {
	for i := range t.t {
		err = t.t[i].UpdateWeights(t.b.h.Handler, t.dw[i].Tensor, t.w[i].Tensor, t.batch)
		if err != nil {
			return err
		}
	}

	return t.b.h.Sync()
}
