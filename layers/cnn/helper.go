package cnn

/*
import (
	"errors"

	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Helper Struct is used to help build the layer
type Helper struct {
	datatype gocudnn.DataType
	mode     gocudnn.ConvolutionMode
	format   gocudnn.TensorFormat
	dim      int
	convdesc *gocudnn.ConvolutionD
	x        *layers.IO
	w        *layers.IO
	y        *layers.IO
}

//CreateLayerHelper Creates and returns a helper for the convolutional layer
func CreateLayerHelper() Helper {

	return Helper{}
}

//CreateTensorHelper creates and returns a gocudnn.Tensor
func CreateTensorHelper() gocudnn.Tensor {
	return gocudnn.Tensor{}
}

//CreateConvolutionHelper creates and returns a gocudnn.Convolution
func CreateConvolutionHelper() gocudnn.Convolution {
	return gocudnn.Convolution{}
}

//CreateFilterHelper creates and returns a gocudnn.Filter
func CreateFilterHelper() gocudnn.Filter {
	return gocudnn.Filter{}
}

//CoreSettings sets up the core settigns
func (h *Helper) CoreSettings(
	datatype gocudnn.DataType,
	mode gocudnn.ConvolutionMode,
	format gocudnn.TensorFormat) {
	h.datatype = datatype
	h.mode = mode
	h.format = format
}

func (h *Helper) DeriveCoreSettings(x *layers.IO) error {
	fmt, dtype, dims, err := x.Properties()
	if err != nil {
		return err
	}
	h.dim = len(dims)
	h.datatype = dtype
	h.format = fmt
	h.x = x
	return nil
}

func (h *Helper) InputSetup(data gocudnn.DataType, format gocudnn.TensorFormat, shape []int32, managedmem bool) error {

	desc, err := layers.BuildIO(format, data, shape, managedmem)
	h.x = desc
	return err

}

func (h *Helper) FilterSetup(shape, pad, stride, dialation []int32, managedmem bool) error {
	var c gocudnn.Convolution
	var err error
	if len(shape) != h.dim {
		return errors.New("Shape length doesn't match dims")
	}
	if len(shape) < 4 {
		return errors.New("Shape should be an array of at least 4 dims not used should be marked as 1")
	}
	if len(shape) == 4 {
		h.w, err = layers.BuildIO(h.format, h.datatype, shape, managedmem)
		if err != nil {
			h.w.Destroy()
			return err
		}

		h.convdesc, err = c.NewConvolution2dDescriptor(h.mode, h.datatype, pad, stride, dialation)
		if err != nil {
			h.w.Destroy()
			h.convdesc.DestroyDescriptor()
			return err
		}

		outputdims, err := h.convdesc.GetConvolution2dForwardOutputDim(h.x.T().TD(), h.w.T().FD())
		if err != nil {
			h.w.Destroy()
			h.convdesc.DestroyDescriptor()
			return err
		}
		h.y, err = layers.BuildIO(h.format, h.datatype, outputdims, managedmem)
		if err != nil {
			h.w.Destroy()
			h.convdesc.DestroyDescriptor()
			h.y.Destroy()
			return err
		}

	}
	h.w, err = layers.BuildIO(h.format, h.datatype, shape, managedmem)
	if err != nil {
		h.w.Destroy()
		return err
	}

	h.convdesc, err = c.NewConvolutionNdDescriptor(h.mode, h.datatype, pad, stride, dialation)
	if err != nil {
		return err
	}

	outputdims, err := h.convdesc.GetConvolutionNdForwardOutputDim(h.x.T().TD(), h.w.T().FD())
	if err != nil {
		h.w.Destroy()
		h.convdesc.DestroyDescriptor()
		return err
	}
	h.y, err = layers.BuildIO(h.format, h.datatype, outputdims, managedmem)
	if err != nil {
		h.w.Destroy()
		h.convdesc.DestroyDescriptor()
		h.y.Destroy()
		return err
	}
	return nil
}

func (h *Helper) GetAlgosLists(handle *gocudnn.Handle, maxlist int32) ([]gocudnn.ConvFwdAlgoPerformance, []gocudnn.ConvBwdDataAlgoPerformance, []gocudnn.ConvBwdFiltAlgoPerformance, error) {
	var c gocudnn.Convolution
	//	size, err:=c.Funcs.Fwd.GetConvolutionForwardAlgorithmMaxCount(handle)
	fwdlist, err := c.Funcs.Fwd.FindConvolutionForwardAlgorithm(handle, h.x.T().TD(), h.w.T().FD(), h.convdesc, h.y.T().TD(), maxlist)
	if err != nil {
		return nil, nil, nil, err
	}
	bwddatalist, err := c.Funcs.Bwd.FindConvolutionBackwardDataAlgorithm(handle, h.w.DeltaT().FD(), h.y.DeltaT().TD(), h.convdesc, h.x.DeltaT().TD(), maxlist)
	if err != nil {
		return nil, nil, nil, err
	}
	bwdfiltlist, err := c.Funcs.Bwd.FindConvolutionBackwardFilterAlgorithm(handle, h.x.DeltaT().TD(), h.y.DeltaT().TD(), h.convdesc, h.w.T().FD(), maxlist)
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
		fastestfwd, err := c.Funcs.Fwd.GetConvolutionForwardAlgorithm(handle, h.x.T().TD(), h.w.T().FD(), h.convdesc, h.y.T().TD(), fastest, 0)
		if err != nil {
			return nil, nil, nil, 0, err
		}
		fastestflagdataback := c.Flgs.Bwd.DataPref.PreferFastest()
		fastestbwddata, err := c.Funcs.Bwd.GetConvolutionBackwardDataAlgorithm(handle, h.w.T().FD(), h.y.DeltaT().TD(), h.convdesc, h.x.DeltaT().TD(), fastestflagdataback, 0)
		if err != nil {
			return nil, nil, nil, 0, err
		}
		fastestflagfiltback := c.Flgs.Bwd.FltrPref.PrefFastest()
		fastestbwdfilt, err := c.Funcs.Bwd.GetConvolutionBackwardFilterAlgorithm(handle, h.x.DeltaT().TD(), h.y.T().TD(), h.convdesc, h.w.T().FD(), fastestflagfiltback, 0)
		if err != nil {
			return nil, nil, nil, 0, err
		}
		fwdsize, err := c.Funcs.Fwd.GetConvolutionForwardWorkspaceSize(handle, h.x.T().TD(), h.w.T().FD(), h.convdesc, h.y.T().TD(), fastestfwd)
		if err != nil {
			return nil, nil, nil, 0, err
		}
		largest := fwdsize
		bwddatasize, err := c.Funcs.Bwd.GetConvolutionBackwardDataWorkspaceSize(handle, h.w.T().FD(), h.y.T().TD(), h.convdesc, h.x.T().TD(), fastestbwddata)
		if err != nil {
			return nil, nil, nil, 0, err
		}
		if bwddatasize > largest {
			largest = bwddatasize
		}
		bwdfiltsize, err := c.Funcs.Bwd.GetConvolutionBackwardFilterWorkspaceSize(handle, h.x.T().TD(), h.y.T().TD(), h.convdesc, h.w.T().FD(), fastestbwdfilt)
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
		fastestfwd, err := c.Funcs.Fwd.GetConvolutionForwardAlgorithm(handle, h.x.T().TD(), h.w.T().FD(), h.convdesc, h.y.T().TD(), fastest, 0)
		if err != nil {
			return nil, nil, nil, 0, err
		}
		fastestflagdataback := c.Flgs.Bwd.DataPref.NoWorkSpace()
		fastestbwddata, err := c.Funcs.Bwd.GetConvolutionBackwardDataAlgorithm(handle, h.w.T().FD(), h.y.T().TD(), h.convdesc, h.x.T().TD(), fastestflagdataback, 0)
		if err != nil {
			return nil, nil, nil, 0, err
		}
		fastestflagfiltback := c.Flgs.Bwd.FltrPref.NoWorkSpace()
		fastestbwdfilt, err := c.Funcs.Bwd.GetConvolutionBackwardFilterAlgorithm(handle, h.x.T().TD(), h.y.T().TD(), h.convdesc, h.w.T().FD(), fastestflagfiltback, 0)
		if err != nil {
			return nil, nil, nil, 0, err
		}
		return &fastestfwd, &fastestbwddata, &fastestbwdfilt, 0, nil

	}
	//I copied and pasted this so don't pay attention to the vars
	fastest := c.Flgs.Fwd.Pref.SpecifyWorkSpaceLimit()
	fastestfwd, err := c.Funcs.Fwd.GetConvolutionForwardAlgorithm(handle, h.x.T().TD(), h.w.T().FD(), h.convdesc, h.y.T().TD(), fastest, wspacesize)
	if err != nil {
		return nil, nil, nil, 0, err
	}
	fastestflagdataback := c.Flgs.Bwd.DataPref.SpecifyWorkSpaceLimit()
	fastestbwddata, err := c.Funcs.Bwd.GetConvolutionBackwardDataAlgorithm(handle, h.w.T().FD(), h.y.T().TD(), h.convdesc, h.x.T().TD(), fastestflagdataback, wspacesize)
	if err != nil {
		return nil, nil, nil, 0, err
	}
	fastestflagfiltback := c.Flgs.Bwd.FltrPref.SpecifyWorkSpaceLimit()
	fastestbwdfilt, err := c.Funcs.Bwd.GetConvolutionBackwardFilterAlgorithm(handle, h.x.T().TD(), h.y.T().TD(), h.convdesc, h.w.T().FD(), fastestflagfiltback, wspacesize)
	if err != nil {
		return nil, nil, nil, 0, err
	}
	fwdsize, err := c.Funcs.Fwd.GetConvolutionForwardWorkspaceSize(handle, h.x.T().TD(), h.w.T().FD(), h.convdesc, h.y.T().TD(), fastestfwd)
	if err != nil {
		return nil, nil, nil, 0, err
	}
	largest := fwdsize
	bwddatasize, err := c.Funcs.Bwd.GetConvolutionBackwardDataWorkspaceSize(handle, h.w.T().FD(), h.y.T().TD(), h.convdesc, h.x.T().TD(), fastestbwddata)
	if err != nil {
		return nil, nil, nil, 0, err
	}
	if bwddatasize > largest {
		largest = bwddatasize
	}
	bwdfiltsize, err := c.Funcs.Bwd.GetConvolutionBackwardFilterWorkspaceSize(handle, h.x.T().TD(), h.y.T().TD(), h.convdesc, h.w.T().FD(), fastestbwdfilt)
	if err != nil {
		return nil, nil, nil, 0, err
	}
	if bwdfiltsize > largest {
		largest = bwdfiltsize
	}
	return &fastestfwd, &fastestbwddata, &fastestbwdfilt, largest, nil
}

//ReturnDescriptors will return the helper desriptors.  x, w, convdesc, y
func (h *Helper) ReturnDescriptors() (*layers.IO, *layers.IO, *gocudnn.ConvolutionD, *layers.IO) {
	return h.x, h.w, h.convdesc, h.y
}

/*
//Don't run this
func (helper *Helper) Destroy() error {
	var some string
	err := helper.outputdesc.DestroyDescriptor()

	if err != nil {
		some = some + err.Error() + ","
	}
	err = helper.filter.DestroyDescriptor()
	if err != nil {
		some = some + err.Error() + ","
	}
	err = helper.outputdesc.DestroyDescriptor()
	if err != nil {
		some = some + err.Error() + ","
	}
	err = helper.convdesc.DestroyDescriptor()
	if err != nil {
		some = some + err.Error() + ","
	}
	if some == "" {
		return nil
	}
	return errors.New(some)
}
*/
