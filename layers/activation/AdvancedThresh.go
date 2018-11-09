package activation

import (
	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/activation"
	"github.com/dereklstinson/GoCuNets/layers"
)

//AdvancedThreshRandRelu returns an activation layer set to AdvancedThreshRandRelu
func AdvancedThreshRandRelu(handle *cudnn.Handler, dtype cudnn.DataType, inputdims []int32, managedmem bool) (*Layer, error) {
	var tff cudnn.TensorFormatFlag
	var flg activation.ModeFlag

	layer, err := setup(handle, flg.AdvancedThreshRandRelu(), defaultnanprop, defaultalpha, defaultbeta, defaultalpha, defaultbeta, defaultcoef)
	if err != nil {
		return nil, err
	}
	dims := []int32{int32(1), inputdims[1], inputdims[2], inputdims[3]}
	abs, err := layers.BuildIO(tff.NCHW(), dtype, dims, managedmem)
	if err != nil {
		return nil, err
	}

	err = abs.T().SetRandomNormal(.01, .1)
	if err != nil {
		return nil, err
	}
	err = abs.DeltaT().SetRandomNormal(0, .07)
	if err != nil {
		return nil, err
	}
	layer.alphasbetas = abs
	return layer, nil
}

//AdvancedThreshRandReluADVANCED returns an activation layer set to AdvancedThreshRandRelu, but gives more options
func AdvancedThreshRandReluADVANCED(handle *cudnn.Handler, dtype cudnn.DataType, nanprop cudnn.NanMode, inputdims []int32, alphaforward, alphabackward, betaforward, betabackward float64, managedmem bool) (*Layer, error) {
	var tff cudnn.TensorFormatFlag
	var flg activation.ModeFlag

	layer, err := setup(handle, flg.AdvancedThreshRandRelu(), nanprop, alphaforward, alphabackward, betaforward, betabackward, defaultcoef)
	if err != nil {
		return nil, err
	}
	dims := []int32{int32(1), inputdims[1], inputdims[2], inputdims[3]}
	abs, err := layers.BuildIO(tff.NCHW(), dtype, dims, managedmem)
	if err != nil {
		return nil, err
	}

	err = abs.T().SetRandomNormal(.01, .1)
	if err != nil {
		return nil, err
	}
	err = abs.DeltaT().SetRandomNormal(0, .07)
	if err != nil {
		return nil, err
	}
	layer.alphasbetas = abs
	return layer, nil
}
