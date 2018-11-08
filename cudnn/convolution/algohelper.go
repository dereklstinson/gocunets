package convolution

import (
	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//SetBestAlgosConsidering this method will set the best algos for the fwd, bwddata, and bwdfilter algos. and return the workspace size along with an error
//if an error is found the function will not set any values,
//Here are some simple rules to the function
//if fastest is marked true. Then it will find the fastest algo no mater what worksize is.
//if fastest is set to false. It will check if wspace is greater than zero then it will set the algos to the fastest algo considering the workspace size, and return the largest wspacesize in all the algos
//else it will find and set the fastest algos with no workspace size and return 0
func (c *Ops) SetBestAlgosConsidering(
	handle *cudnn.Handler,
	x *tensor.Volume,
	y *tensor.Volume,
	w *tensor.Volume,
	wspacesize int,
	fastest bool) (
	gocudnn.SizeT,
	error) {
	if fastest == true {
		return c.setfastest(handle, x, y, w)
	}
	if wspacesize > 0 {
		return c.setwspacelimit(handle, x, y, w, wspacesize)
	}
	return gocudnn.SizeT(wspacesize), c.setnowspace(handle, x, y, w)
}

//SetBestAlgosConsideringDims4d is the same as SetBestAlgosConsidering as long as the dims are 4d
func (c *Ops) SetBestAlgosConsideringDims4d(
	handle *cudnn.Handler,
	xd []int32,
	yd []int32,
	wd []int32,
	wspacesize int,
	fastest bool,
	dtype gocudnn.DataType,
	frmt gocudnn.TensorFormat,
) (
	gocudnn.SizeT,
	error) {

	x, err := gocudnn.Tensor{}.NewTensor4dDescriptor(dtype, frmt, xd)
	if err != nil {
		return 0, err
	}
	defer x.DestroyDescriptor()
	y, err := gocudnn.Tensor{}.NewTensor4dDescriptor(dtype, frmt, yd)
	if err != nil {
		return 0, err
	}
	defer y.DestroyDescriptor()
	w, err := gocudnn.Filter{}.NewFilter4dDescriptor(dtype, frmt, wd)
	if err != nil {
		return 0, err
	}
	defer w.DestroyDescriptor()
	if fastest == true {
		a, err := c.setfastestdescriptors(handle, x, y, w)
		return a, err
	}
	if wspacesize > 0 {
		a, err := c.setwspacelimitdescriptors(handle, x, y, w, wspacesize)
		return a, err
	}
	err = c.setnowspacedescriptors(handle, x, y, w)
	return 0, err
}

//this will set the fastest algos to to the struct and return the largest worksize for fwd, bwdd, bwdf. if an error is found nothing will be set.
func (c *Ops) setfastestdescriptors(handle *cudnn.Handler, x, y *gocudnn.TensorD, w *gocudnn.FilterD) (gocudnn.SizeT, error) {
	preff := c.helper.Flgs.Fwd.Pref.PreferFastest()
	prefbd := c.helper.Flgs.Bwd.DataPref.PreferFastest()
	prefbf := c.helper.Flgs.Bwd.FltrPref.PrefFastest()
	wspace := gocudnn.SizeT(0)
	fwdalgo, err := c.helper.Funcs.Fwd.GetConvolutionForwardAlgorithm(handle.Cudnn(), x, w, c.desc, y, preff, wspace)
	if err != nil {
		return 0, err
	}

	fwdwspace, err := c.helper.Funcs.Fwd.GetConvolutionForwardWorkspaceSize(handle.Cudnn(), x, w, c.desc, y, fwdalgo)
	if err != nil {
		return 0, err
	}
	bwddalgo, err := c.helper.Funcs.Bwd.GetConvolutionBackwardDataAlgorithm(handle.Cudnn(), w, y, c.desc, x, prefbd, wspace)
	if err != nil {
		return 0, err
	}

	bwddspace, err := c.helper.Funcs.Bwd.GetConvolutionBackwardDataWorkspaceSize(handle.Cudnn(), w, y, c.desc, x, bwddalgo)
	if err != nil {
		return 0, err
	}

	bwdfalgo, err := c.helper.Funcs.Bwd.GetConvolutionBackwardFilterAlgorithm(handle.Cudnn(), x, y, c.desc, w, prefbf, wspace)
	if err != nil {
		return 0, err
	}
	bwdfspace, err := c.helper.Funcs.Bwd.GetConvolutionBackwardFilterWorkspaceSize(handle.Cudnn(), x, y, c.desc, w, bwdfalgo)
	if err != nil {
		return 0, err
	}
	c.bwdfilt = bwdfalgo
	c.fwdalgo = fwdalgo
	c.bwddata = bwddalgo
	largest := fwdwspace
	if bwddspace > largest {
		largest = bwddspace
	}
	if bwdfspace > largest {
		largest = bwdfspace
	}
	return largest, nil
}

func (c *Ops) setnowspacedescriptors(handle *cudnn.Handler, x, y *gocudnn.TensorD, w *gocudnn.FilterD) error {
	preff := c.helper.Flgs.Fwd.Pref.NoWorkSpace()
	prefbd := c.helper.Flgs.Bwd.DataPref.NoWorkSpace()
	prefbf := c.helper.Flgs.Bwd.FltrPref.NoWorkSpace()
	wspace := gocudnn.SizeT(0)
	fwdalgo, err := c.helper.Funcs.Fwd.GetConvolutionForwardAlgorithm(handle.Cudnn(), x, w, c.desc, y, preff, wspace)
	if err != nil {
		return err
	}
	bwddalgo, err := c.helper.Funcs.Bwd.GetConvolutionBackwardDataAlgorithm(handle.Cudnn(), w, y, c.desc, x, prefbd, wspace)
	if err != nil {
		return err
	}

	bwdfalgo, err := c.helper.Funcs.Bwd.GetConvolutionBackwardFilterAlgorithm(handle.Cudnn(), x, y, c.desc, w, prefbf, wspace)
	if err != nil {
		return err
	}

	c.bwdfilt = bwdfalgo
	c.fwdalgo = fwdalgo
	c.bwddata = bwddalgo

	return nil
}

//this will set the fastest algo with the memory limit given if an error is returned no settings will be changed
func (c *Ops) setwspacelimitdescriptors(handle *cudnn.Handler, x, y *gocudnn.TensorD, w *gocudnn.FilterD, wspacesize int) (gocudnn.SizeT, error) {
	preff := c.helper.Flgs.Fwd.Pref.SpecifyWorkSpaceLimit()
	prefbd := c.helper.Flgs.Bwd.DataPref.SpecifyWorkSpaceLimit()
	prefbf := c.helper.Flgs.Bwd.FltrPref.SpecifyWorkSpaceLimit()
	wspace := gocudnn.SizeT(wspacesize)
	fwdalgo, err := c.helper.Funcs.Fwd.GetConvolutionForwardAlgorithm(handle.Cudnn(), x, w, c.desc, y, preff, wspace)
	if err != nil {
		return 0, err
	}

	bwddalgo, err := c.helper.Funcs.Bwd.GetConvolutionBackwardDataAlgorithm(handle.Cudnn(), w, y, c.desc, x, prefbd, wspace)
	if err != nil {
		return 0, err
	}

	bwdfalgo, err := c.helper.Funcs.Bwd.GetConvolutionBackwardFilterAlgorithm(handle.Cudnn(), x, y, c.desc, w, prefbf, wspace)
	if err != nil {
		return 0, err
	}
	fwdwspace, err := c.helper.Funcs.Fwd.GetConvolutionForwardWorkspaceSize(handle.Cudnn(), x, w, c.desc, y, fwdalgo)
	if err != nil {
		return 0, err
	}
	bwddspace, err := c.helper.Funcs.Bwd.GetConvolutionBackwardDataWorkspaceSize(handle.Cudnn(), w, y, c.desc, x, bwddalgo)
	if err != nil {
		return 0, err
	}
	bwdfspace, err := c.helper.Funcs.Bwd.GetConvolutionBackwardFilterWorkspaceSize(handle.Cudnn(), x, y, c.desc, w, bwdfalgo)
	if err != nil {
		return 0, err
	}
	c.bwdfilt = bwdfalgo
	c.fwdalgo = fwdalgo
	c.bwddata = bwddalgo
	largest := fwdwspace
	if bwddspace > largest {
		largest = bwddspace
	}
	if bwdfspace > largest {
		largest = bwdfspace
	}
	return largest, nil
}

//this will set the fastest algo with  no workspace if an error is returned then no settings will be changed
func (c *Ops) setnowspace(handle *cudnn.Handler, x, y, w *tensor.Volume) error {
	preff := c.helper.Flgs.Fwd.Pref.NoWorkSpace()
	prefbd := c.helper.Flgs.Bwd.DataPref.NoWorkSpace()
	prefbf := c.helper.Flgs.Bwd.FltrPref.NoWorkSpace()
	wspace := gocudnn.SizeT(0)
	fwdalgo, err := c.helper.Funcs.Fwd.GetConvolutionForwardAlgorithm(handle.Cudnn(), x.TD(), w.FD(), c.desc, y.TD(), preff, wspace)
	if err != nil {
		return err
	}
	bwddalgo, err := c.helper.Funcs.Bwd.GetConvolutionBackwardDataAlgorithm(handle.Cudnn(), w.FD(), y.TD(), c.desc, x.TD(), prefbd, wspace)
	if err != nil {
		return err
	}

	bwdfalgo, err := c.helper.Funcs.Bwd.GetConvolutionBackwardFilterAlgorithm(handle.Cudnn(), x.TD(), y.TD(), c.desc, w.FD(), prefbf, wspace)
	if err != nil {
		return err
	}

	c.bwdfilt = bwdfalgo
	c.fwdalgo = fwdalgo
	c.bwddata = bwddalgo

	return nil
}

//this will set the fastest algo with the memory limit given if an error is returned no settings will be changed
func (c *Ops) setwspacelimit(handle *cudnn.Handler, x, y, w *tensor.Volume, wspacesize int) (gocudnn.SizeT, error) {
	preff := c.helper.Flgs.Fwd.Pref.SpecifyWorkSpaceLimit()
	prefbd := c.helper.Flgs.Bwd.DataPref.SpecifyWorkSpaceLimit()
	prefbf := c.helper.Flgs.Bwd.FltrPref.SpecifyWorkSpaceLimit()
	wspace := gocudnn.SizeT(wspacesize)
	fwdalgo, err := c.helper.Funcs.Fwd.GetConvolutionForwardAlgorithm(handle.Cudnn(), x.TD(), w.FD(), c.desc, y.TD(), preff, wspace)
	if err != nil {
		return 0, err
	}

	bwddalgo, err := c.helper.Funcs.Bwd.GetConvolutionBackwardDataAlgorithm(handle.Cudnn(), w.FD(), y.TD(), c.desc, x.TD(), prefbd, wspace)
	if err != nil {
		return 0, err
	}

	bwdfalgo, err := c.helper.Funcs.Bwd.GetConvolutionBackwardFilterAlgorithm(handle.Cudnn(), x.TD(), y.TD(), c.desc, w.FD(), prefbf, wspace)
	if err != nil {
		return 0, err
	}
	fwdwspace, err := c.helper.Funcs.Fwd.GetConvolutionForwardWorkspaceSize(handle.Cudnn(), x.TD(), w.FD(), c.desc, y.TD(), fwdalgo)
	if err != nil {
		return 0, err
	}
	bwddspace, err := c.helper.Funcs.Bwd.GetConvolutionBackwardDataWorkspaceSize(handle.Cudnn(), w.FD(), y.TD(), c.desc, x.TD(), bwddalgo)
	if err != nil {
		return 0, err
	}
	bwdfspace, err := c.helper.Funcs.Bwd.GetConvolutionBackwardFilterWorkspaceSize(handle.Cudnn(), x.TD(), y.TD(), c.desc, w.FD(), bwdfalgo)
	if err != nil {
		return 0, err
	}
	c.bwdfilt = bwdfalgo
	c.fwdalgo = fwdalgo
	c.bwddata = bwddalgo
	largest := fwdwspace
	if bwddspace > largest {
		largest = bwddspace
	}
	if bwdfspace > largest {
		largest = bwdfspace
	}
	return largest, nil
}

//this will set the fastest algos to to the struct and return the largest worksize for fwd, bwdd, bwdf. if an error is found nothing will be set.
func (c *Ops) setfastest(handle *cudnn.Handler, x, y, w *tensor.Volume) (gocudnn.SizeT, error) {
	preff := c.helper.Flgs.Fwd.Pref.PreferFastest()
	prefbd := c.helper.Flgs.Bwd.DataPref.PreferFastest()
	prefbf := c.helper.Flgs.Bwd.FltrPref.PrefFastest()
	wspace := gocudnn.SizeT(0)
	fwdalgo, err := c.helper.Funcs.Fwd.GetConvolutionForwardAlgorithm(handle.Cudnn(), x.TD(), w.FD(), c.desc, y.TD(), preff, wspace)
	if err != nil {
		return 0, err
	}

	fwdwspace, err := c.helper.Funcs.Fwd.GetConvolutionForwardWorkspaceSize(handle.Cudnn(), x.TD(), w.FD(), c.desc, y.TD(), fwdalgo)
	if err != nil {
		return 0, err
	}
	bwddalgo, err := c.helper.Funcs.Bwd.GetConvolutionBackwardDataAlgorithm(handle.Cudnn(), w.FD(), y.TD(), c.desc, x.TD(), prefbd, wspace)
	if err != nil {
		return 0, err
	}

	bwddspace, err := c.helper.Funcs.Bwd.GetConvolutionBackwardDataWorkspaceSize(handle.Cudnn(), w.FD(), y.TD(), c.desc, x.TD(), bwddalgo)
	if err != nil {
		return 0, err
	}

	bwdfalgo, err := c.helper.Funcs.Bwd.GetConvolutionBackwardFilterAlgorithm(handle.Cudnn(), x.TD(), y.TD(), c.desc, w.FD(), prefbf, wspace)
	if err != nil {
		return 0, err
	}
	bwdfspace, err := c.helper.Funcs.Bwd.GetConvolutionBackwardFilterWorkspaceSize(handle.Cudnn(), x.TD(), y.TD(), c.desc, w.FD(), bwdfalgo)
	if err != nil {
		return 0, err
	}
	c.bwdfilt = bwdfalgo
	c.fwdalgo = fwdalgo
	c.bwddata = bwddalgo
	largest := fwdwspace
	if bwddspace > largest {
		largest = bwddspace
	}
	if bwdfspace > largest {
		largest = bwdfspace
	}
	return largest, nil
}
