package activation

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/activation"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/utils"
)

//Threshhold returns an activation layer set to AdvancedThreshRandRelu
func Threshhold(handle *cudnn.Handler, dtype cudnn.DataType, minneg, maxneg, minthresh, maxthresh, minpos, maxpos float32, managedmem bool) (*Layer, error) {

	var flg activation.ModeFlag

	layer, err := setup(handle, flg.Threshhold(), defaultnanprop, 0, 0, 0, 0, defaultcoef)
	if err != nil {
		return nil, err
	}
	layer.threshmax = maxthresh
	layer.threshmin = minthresh
	layer.negmin = minneg
	layer.negmax = minneg
	layer.posmax = maxpos
	layer.posmin = minpos
	layer.updatable = true
	return layer, nil
}

//MakeOutputTensor returns the output tensor, and it also builds the layer
func (l *Layer) MakeOutputTensor(input *layers.IO) (*layers.IO, error) {
	memmanaged := input.MemIsManaged()
	frmt, dtype, dims, err := input.Properties()
	if err != nil {
		return nil, err
	}
	var flg activation.ModeFlag

	switch l.act.Mode() {
	case flg.Threshhold():

		if l.negCoefs == nil {
			adims := make([]int32, len(dims))
			copy(adims, dims)
			adims[0] = 1
			l.threshandpreludims = adims
			l.negCoefs, err = layers.BuildIO(frmt, dtype, adims, memmanaged)
			if err != nil {
				return nil, err
			}
			l.negCoefs.T().SetRandomNormal(l.negmin, l.negmax)
			l.posCoefs, err = layers.BuildIO(frmt, dtype, adims, memmanaged)
			if err != nil {
				return nil, err
			}
			l.posCoefs.T().SetRandomNormal(l.posmin, l.posmax)
			l.threshold, err = layers.BuildNetworkInputIO(frmt, dtype, adims, memmanaged)
			if err != nil {
				return nil, err
			}
			l.threshold.T().SetRandomNormal(l.threshmin, l.threshmax)
		} else if utils.FindVolumeInt32(dims[1:], nil) != utils.FindVolumeInt32(l.threshandpreludims[1:], nil) {
			return nil, errors.New("Threshhold and Prelu Function have set number of weights.  Not able to change have dynamic sizing input")
		}

	case flg.PRelu():
		adims := make([]int32, len(dims))
		copy(adims, dims)
		adims[0] = 1
		if l.negCoefs == nil {
			l.negCoefs, err = layers.BuildIO(frmt, dtype, adims, memmanaged)
			if err != nil {
				return nil, err
			}
		} else if utils.FindVolumeInt32(dims[1:], nil) != utils.FindVolumeInt32(l.threshandpreludims[1:], nil) {
			return nil, errors.New("Threshhold and Prelu Function have set number of weights.  Not able to change have dynamic sizing input")
		}
	}
	return layers.BuildIO(frmt, dtype, dims, memmanaged)

}
