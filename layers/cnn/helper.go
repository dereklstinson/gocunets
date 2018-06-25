package cnn

import (
	"errors"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Helper Struct is used to help build the layer
type Helper struct {
	nd         bool
	datatype   gocudnn.DataType
	mode       gocudnn.ConvolutionMode
	format     gocudnn.TensorFormat
	dim        int
	convdesc   *gocudnn.ConvolutionD
	inputdesc  *gocudnn.TensorD
	filter     *gocudnn.FilterD
	outputdesc *gocudnn.TensorD
}

func ConvolutionHelper() (Helper, gocudnn.Convolution, gocudnn.Tensor, gocudnn.Filter) {

	return Helper{}, gocudnn.Convolution{}, gocudnn.Tensor{}, gocudnn.Filter{}
}

func (h *Helper) CoreSettings(
	nd bool,
	datatype gocudnn.DataType,
	mode gocudnn.ConvolutionMode,
	format gocudnn.TensorFormat) {
	h.nd = nd
	h.datatype = datatype
	h.mode = mode
	h.format = format
}
func (h *Helper) InputInsert(inputdesc *gocudnn.TensorD) error {
	dtype, dim, _, err := inputdesc.GetDescrptor()
	h.dim = len(dim)
	if h.nd == true && len(dim) == 4 {
		return errors.New("input dims not matching nd/4d format")
	}
	if err != nil {
		return err
	}
	if dtype != h.datatype {
		return errors.New("inputdatatype doesn't match datatype of helper")
	}
	h.inputdesc = inputdesc
	return nil
}

func (h *Helper) InputSetup(data gocudnn.DataType, format gocudnn.TensorFormat, shape []int32) error {
	nd := false
	h.dim = len(shape)
	if len(shape) != 4 {
		nd = true
	}
	h.nd = nd
	var t gocudnn.Tensor

	if nd == false {
		desc, err := t.NewTensor4dDescriptor(data, format, shape)
		h.inputdesc = desc
		return err
	}

	desc, err := t.NewTensorNdDescriptorEx(format, data, shape)
	h.inputdesc = desc

	return err
}

func (h *Helper) FilterSetup(shape, pad, stride, dialation []int32) error {
	var f gocudnn.Filter
	var c gocudnn.Convolution
	var t gocudnn.Tensor
	if len(shape) != h.dim {
		return errors.New("Shape length doesn't match dims")
	}
	if len(shape) == 4 {
		Filtdesc, err := f.NewFilter4dDescriptor(h.datatype, h.format, shape)
		if err != nil {
			return err
		}
		h.filter = Filtdesc
		ConvDesc, err := c.NewConvolution2dDescriptor(h.mode, h.datatype, pad, stride, dialation)
		if err != nil {
			return err
		}
		h.convdesc = ConvDesc
		outputdims, err := ConvDesc.GetConvolution2dForwardOutputDim(h.inputdesc, Filtdesc)
		if err != nil {
			return err
		}
		h.outputdesc, err = t.NewTensor4dDescriptor(h.datatype, h.format, outputdims)
		if err != nil {
			return err
		}
	}
	Filtdesc, err := f.NewFilterNdDescriptor(h.datatype, h.format, shape)
	if err != nil {
		return err
	}
	h.filter = Filtdesc
	ConvDesc, err := c.NewConvolutionNdDescriptor(h.mode, h.datatype, pad, stride, dialation)
	if err != nil {
		return err
	}
	h.convdesc = ConvDesc
	outputdims, err := ConvDesc.GetConvolutionNdForwardOutputDim(h.inputdesc, Filtdesc)
	if err != nil {
		return err
	}
	h.outputdesc, err = t.NewTensorNdDescriptorEx(h.format, h.datatype, outputdims)
	if err != nil {
		return err
	}
	return nil
}

func (h *Helper) GetAlgosLists(handle *gocudnn.Handle, maxlist int32) ([]gocudnn.ConvFwdAlgoPerformance, []gocudnn.ConvBwdDataAlgoPerf, []gocudnn.ConvBwdFiltAlgoPerf, error) {
	var c gocudnn.Convolution
	//	size, err:=c.Funcs.Fwd.GetConvolutionForwardAlgorithmMaxCount(handle)
	fwdlist, err := c.Funcs.Fwd.FindConvolutionForwardAlgorithm(handle, h.inputdesc, h.filter, h.convdesc, h.outputdesc, maxlist)
	if err != nil {
		return nil, nil, nil, err
	}
	bwddatalist, err := c.Funcs.Bwd.FindConvolutionBackwardDataAlgorithm(handle, h.filter, h.outputdesc, h.convdesc, h.inputdesc, maxlist)
	if err != nil {
		return nil, nil, nil, err
	}
	bwdfiltlist, err := c.Funcs.Bwd.FindConvolutionBackwardFilterAlgorithm(handle, h.inputdesc, h.outputdesc, h.convdesc, h.filter, maxlist)
	if err != nil {
		return nil, nil, nil, err
	}
	return fwdlist, bwddatalist, bwdfiltlist, nil
}

// GetBestAlgoConsidering will give you the fasted algo considering the conditions that are sent.
//If speedpreffastest is true then wspacesize is ignored.
//If speedpreffastest is false, and wspacesize has a value.
//Then it will give pass algos considering that value.
//If speedpreffastest is false and wspacesize is zero then it will consider no noworkspace and give the fastest algo with that condition.
func (h *Helper) GetBestAlgoConsidering(handle *gocudnn.Handle, wspacesize gocudnn.SizeT, speedpreffastest bool) (*gocudnn.ConvFwdAlgo, *gocudnn.ConvBwdDataAlgo, *gocudnn.ConvBwdFiltAlgo, gocudnn.SizeT, error) {
	var c gocudnn.Convolution
	if speedpreffastest == true {
		fastest := c.Flgs.Fwd.Pref.PreferFastest()
		fastestfwd, err := c.Funcs.Fwd.GetConvolutionForwardAlgorithm(handle, h.inputdesc, h.filter, h.convdesc, h.outputdesc, fastest, 0)
		if err != nil {
			return nil, nil, nil, 0, err
		}
		fastestflagdataback := c.Flgs.Bwd.DataPref.PreferFastest()
		fastestbwddata, err := c.Funcs.Bwd.GetConvolutionBackwardDataAlgorithm(handle, h.filter, h.outputdesc, h.convdesc, h.inputdesc, fastestflagdataback, 0)
		if err != nil {
			return nil, nil, nil, 0, err
		}
		fastestflagfiltback := c.Flgs.Bwd.FltrPref.PrefFastest()
		fastestbwdfilt, err := c.Funcs.Bwd.GetConvolutionBackwardFilterAlgorithm(handle, h.inputdesc, h.outputdesc, h.convdesc, h.filter, fastestflagfiltback, 0)
		if err != nil {
			return nil, nil, nil, 0, err
		}
		fwdsize, err := c.Funcs.Fwd.GetConvolutionForwardWorkspaceSize(handle, h.inputdesc, h.filter, h.convdesc, h.outputdesc, fastestfwd)
		if err != nil {
			return nil, nil, nil, 0, err
		}
		largest := fwdsize
		bwddatasize, err := c.Funcs.Bwd.GetConvolutionBackwardDataWorkspaceSize(handle, h.filter, h.outputdesc, h.convdesc, h.inputdesc, fastestbwddata)
		if err != nil {
			return nil, nil, nil, 0, err
		}
		if bwddatasize > largest {
			largest = bwddatasize
		}
		bwdfiltsize, err := c.Funcs.Bwd.GetConvolutionBackwardFilterWorkspaceSize(handle, h.inputdesc, h.outputdesc, h.convdesc, h.filter, fastestbwdfilt)
		if err != nil {
			return nil, nil, nil, 0, err
		}
		if bwdfiltsize > largest {
			largest = bwdfiltsize
		}
		return &fastestfwd, &fastestbwddata, &fastestbwdfilt, largest, nil
	}
	if speedpreffastest == false && wspacesize == 0 {
		//I copied and pasted this so don't pay attention to the vars
		fastest := c.Flgs.Fwd.Pref.NoWorkSpace()
		fastestfwd, err := c.Funcs.Fwd.GetConvolutionForwardAlgorithm(handle, h.inputdesc, h.filter, h.convdesc, h.outputdesc, fastest, 0)
		if err != nil {
			return nil, nil, nil, 0, err
		}
		fastestflagdataback := c.Flgs.Bwd.DataPref.NoWorkSpace()
		fastestbwddata, err := c.Funcs.Bwd.GetConvolutionBackwardDataAlgorithm(handle, h.filter, h.outputdesc, h.convdesc, h.inputdesc, fastestflagdataback, 0)
		if err != nil {
			return nil, nil, nil, 0, err
		}
		fastestflagfiltback := c.Flgs.Bwd.FltrPref.NoWorkSpace()
		fastestbwdfilt, err := c.Funcs.Bwd.GetConvolutionBackwardFilterAlgorithm(handle, h.inputdesc, h.outputdesc, h.convdesc, h.filter, fastestflagfiltback, 0)
		if err != nil {
			return nil, nil, nil, 0, err
		}
		return &fastestfwd, &fastestbwddata, &fastestbwdfilt, 0, nil

	}
	//I copied and pasted this so don't pay attention to the vars
	fastest := c.Flgs.Fwd.Pref.SpecifyWorkSpaceLimit()
	fastestfwd, err := c.Funcs.Fwd.GetConvolutionForwardAlgorithm(handle, h.inputdesc, h.filter, h.convdesc, h.outputdesc, fastest, wspacesize)
	if err != nil {
		return nil, nil, nil, 0, err
	}
	fastestflagdataback := c.Flgs.Bwd.DataPref.SpecifyWorkSpaceLimit()
	fastestbwddata, err := c.Funcs.Bwd.GetConvolutionBackwardDataAlgorithm(handle, h.filter, h.outputdesc, h.convdesc, h.inputdesc, fastestflagdataback, wspacesize)
	if err != nil {
		return nil, nil, nil, 0, err
	}
	fastestflagfiltback := c.Flgs.Bwd.FltrPref.SpecifyWorkSpaceLimit()
	fastestbwdfilt, err := c.Funcs.Bwd.GetConvolutionBackwardFilterAlgorithm(handle, h.inputdesc, h.outputdesc, h.convdesc, h.filter, fastestflagfiltback, wspacesize)
	if err != nil {
		return nil, nil, nil, 0, err
	}
	fwdsize, err := c.Funcs.Fwd.GetConvolutionForwardWorkspaceSize(handle, h.inputdesc, h.filter, h.convdesc, h.outputdesc, fastestfwd)
	if err != nil {
		return nil, nil, nil, 0, err
	}
	largest := fwdsize
	bwddatasize, err := c.Funcs.Bwd.GetConvolutionBackwardDataWorkspaceSize(handle, h.filter, h.outputdesc, h.convdesc, h.inputdesc, fastestbwddata)
	if err != nil {
		return nil, nil, nil, 0, err
	}
	if bwddatasize > largest {
		largest = bwddatasize
	}
	bwdfiltsize, err := c.Funcs.Bwd.GetConvolutionBackwardFilterWorkspaceSize(handle, h.inputdesc, h.outputdesc, h.convdesc, h.filter, fastestbwdfilt)
	if err != nil {
		return nil, nil, nil, 0, err
	}
	if bwdfiltsize > largest {
		largest = bwdfiltsize
	}
	return &fastestfwd, &fastestbwddata, &fastestbwdfilt, largest, nil
}
func (helper *Helper) ReturnDescriptors() (*gocudnn.TensorD, *gocudnn.FilterD, *gocudnn.ConvolutionD, *gocudnn.TensorD) {
	return helper.inputdesc, helper.filter, helper.convdesc, helper.outputdesc
}
func (helper *Helper) DestroyDescriptors() (error, error, error, error) {
	err := helper.outputdesc.DestroyDescriptor()
	err1 := helper.filter.DestroyDescriptor()
	err2 := helper.outputdesc.DestroyDescriptor()
	err3 := helper.convdesc.DestroyDescriptor()
	return err, err1, err2, err3
}
