package activation

import (
	"errors"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/trainer"
)

//LoadTrainer loads the trainer and sets up the training memory
func (l *Layer) LoadTrainer(handle *cudnn.Handler, trainers []trainer.Trainer) error {
	var err error
	var addedtrainer int
	if l.negCoefs != nil {
		if len(trainers) > addedtrainer {
			l.negcotrain = trainers[addedtrainer]
			err = trainer.CreateTrainingMem(handle, l.negcotrain, l.negCoefs)
			if err != nil {
				return err
			}
			addedtrainer++
		} else {
			return errors.New("Not enough trainers")
		}

	}
	if l.posCoefs != nil {
		if len(trainers) > addedtrainer {
			l.poscotrain = trainers[addedtrainer]
			err = trainer.CreateTrainingMem(handle, l.poscotrain, l.posCoefs)
			if err != nil {
				return err
			}
			addedtrainer++
		} else {
			return errors.New("Not enough trainers")
		}
	}
	if l.threshold != nil {
		if len(trainers) > addedtrainer {
			l.thresholdtrain = trainers[addedtrainer]
			err = trainer.CreateTrainingMem(handle, l.thresholdtrain, l.threshold)
			if err != nil {
				return err
			}
			addedtrainer++
		} else {
			return errors.New("Not enough trainers")
		}
	}
	return nil
}

//UpdateWeights does the weight update
func (l *Layer) UpdateWeights(handle *cudnn.Handler, batch int) error {
	var err error
	if l.negCoefs != nil {
		err = l.negcotrain.UpdateWeights(handle, l.negCoefs, batch)
		if err != nil {
			return nil
		}
	}
	if l.posCoefs != nil {
		err = l.poscotrain.UpdateWeights(handle, l.posCoefs, batch)
		if err != nil {
			return nil
		}
	}
	if l.threshold != nil {
		err = l.poscotrain.UpdateWeights(handle, l.threshold, batch)
		if err != nil {
			return nil
		}
	}
	err = handle.Sync()
	if err != nil {
		return err
	}
	l.l1n, l.l2n = l.negcotrain.L1L2Loss()
	l.l1p, l.l2p = l.poscotrain.L1L2Loss()
	l.l1t, l.l2t = l.thresholdtrain.L1L2Loss()
	return nil
}

//L1L2Loss will return the L1 loss and L2 loss for the layer
func (l *Layer) L1L2Loss() (L1 float32, L2 float32) {
	return l.l1n + l.l1p, l.l2n + l.l2p + l.l1t + l.l2t
}
