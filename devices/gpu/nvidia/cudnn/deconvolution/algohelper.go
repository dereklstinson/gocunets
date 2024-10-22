package deconvolution

import (
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn/tensor"
	gocudnn "github.com/dereklstinson/gocudnn"
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
	uint,
	error) {
	if fastest == true {
		return c.setfastest(handle, x, y, w)
	}
	if wspacesize > 0 {
		return c.setwspacelimit(handle, x, y, w, wspacesize)
	}
	return uint(wspacesize), c.setnowspace(handle, x, y, w)
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
	uint,
	error) {
	x, err := gocudnn.CreateTensorDescriptor()
	if err != nil {
		return 0, err
	}
	y, err := gocudnn.CreateTensorDescriptor()
	if err != nil {
		return 0, err
	}
	w, err := gocudnn.CreateFilterDescriptor()
	if err != nil {
		return 0, err
	}
	err = x.Set(frmt, dtype, xd, nil)
	if err != nil {
		return 0, err
	}
	err = y.Set(frmt, dtype, yd, nil)
	if err != nil {
		return 0, err
	}
	err = w.Set(dtype, frmt, wd)
	if err != nil {
		return 0, err
	}

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
func (c *Ops) setfastestdescriptors(handle *cudnn.Handler, x, y *gocudnn.TensorD, w *gocudnn.FilterD) (uint, error) {
	var preff gocudnn.DeConvolutionForwardPref
	var prefbd gocudnn.DeConvBwdDataPref
	var prefbf gocudnn.DeConvBwdFilterPref
	preff.PreferFastest()
	prefbd.PreferFastest()
	prefbf.PreferFastest()

	wspace := uint(0)
	fwdalgo, err := c.op.GetForwardAlgorithm(handle.Cudnn(), x, w, y, preff, wspace)
	if err != nil {
		return 0, err
	}

	fwdwspace, err := c.op.GetForwardWorkspaceSize(handle.Cudnn(), x, w, y, fwdalgo)
	if err != nil {
		return 0, err
	}
	bwddalgo, err := c.op.GetBackwardDataAlgorithm(handle.Cudnn(), w, y, x, prefbd, wspace)
	if err != nil {
		return 0, err
	}

	bwddspace, err := c.op.GetBackwardDataWorkspaceSize(handle.Cudnn(), w, y, x, bwddalgo)
	if err != nil {
		return 0, err
	}

	bwdfalgo, err := c.op.GetBackwardFilterAlgorithm(handle.Cudnn(), x, y, w, prefbf, wspace)
	if err != nil {
		return 0, err
	}
	bwdfspace, err := c.op.GetBackwardFilterWorkspaceSize(handle.Cudnn(), x, y, w, bwdfalgo)
	if err != nil {
		return 0, err
	}
	c.perfbackfilt.Algo = bwdfalgo
	c.perfforward.Algo = fwdalgo
	c.perfbackdata.Algo = bwddalgo
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
	var preff gocudnn.DeConvolutionForwardPref
	var prefbd gocudnn.DeConvBwdDataPref
	var prefbf gocudnn.DeConvBwdFilterPref
	preff.NoWorkSpace()
	prefbd.NoWorkSpace()
	prefbf.NoWorkSpace()

	wspace := uint(0)
	fwdalgo, err := c.op.GetForwardAlgorithm(handle.Cudnn(), x, w, y, preff, wspace)
	if err != nil {
		return err
	}

	bwddalgo, err := c.op.GetBackwardDataAlgorithm(handle.Cudnn(), w, y, x, prefbd, wspace)
	if err != nil {
		return err
	}

	bwdfalgo, err := c.op.GetBackwardFilterAlgorithm(handle.Cudnn(), x, y, w, prefbf, wspace)
	if err != nil {
		return err
	}

	c.perfbackfilt.Algo = bwdfalgo
	c.perfforward.Algo = fwdalgo
	c.perfbackdata.Algo = bwddalgo
	return nil
}

//this will set the fastest algo with the memory limit given if an error is returned no settings will be changed
func (c *Ops) setwspacelimitdescriptors(handle *cudnn.Handler, x, y *gocudnn.TensorD, w *gocudnn.FilterD, wspacesize int) (uint, error) {
	var preff gocudnn.DeConvolutionForwardPref
	var prefbd gocudnn.DeConvBwdDataPref
	var prefbf gocudnn.DeConvBwdFilterPref
	preff.SpecifyWorkSpaceLimit()
	prefbd.SpecifyWorkSpaceLimit()
	prefbf.SpecifyWorkSpaceLimit()

	wspace := uint(wspacesize)
	fwdalgo, err := c.op.GetForwardAlgorithm(handle.Cudnn(), x, w, y, preff, wspace)
	if err != nil {
		return 0, err
	}

	fwdwspace, err := c.op.GetForwardWorkspaceSize(handle.Cudnn(), x, w, y, fwdalgo)
	if err != nil {
		return 0, err
	}
	bwddalgo, err := c.op.GetBackwardDataAlgorithm(handle.Cudnn(), w, y, x, prefbd, wspace)
	if err != nil {
		return 0, err
	}

	bwddspace, err := c.op.GetBackwardDataWorkspaceSize(handle.Cudnn(), w, y, x, bwddalgo)
	if err != nil {
		return 0, err
	}

	bwdfalgo, err := c.op.GetBackwardFilterAlgorithm(handle.Cudnn(), x, y, w, prefbf, wspace)
	if err != nil {
		return 0, err
	}
	bwdfspace, err := c.op.GetBackwardFilterWorkspaceSize(handle.Cudnn(), x, y, w, bwdfalgo)
	if err != nil {
		return 0, err
	}
	c.perfbackfilt.Algo = bwdfalgo
	c.perfforward.Algo = fwdalgo
	c.perfbackdata.Algo = bwddalgo

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
	var preff gocudnn.DeConvolutionForwardPref
	var prefbd gocudnn.DeConvBwdDataPref
	var prefbf gocudnn.DeConvBwdFilterPref
	preff.NoWorkSpace()
	prefbd.NoWorkSpace()
	prefbf.NoWorkSpace()

	wspace := uint(0)
	fwdalgo, err := c.op.GetForwardAlgorithm(handle.Cudnn(), x.TD(), w.FD(), y.TD(), preff, wspace)
	if err != nil {
		return err
	}

	bwddalgo, err := c.op.GetBackwardDataAlgorithm(handle.Cudnn(), w.FD(), y.TD(), x.TD(), prefbd, wspace)
	if err != nil {
		return err
	}

	bwdfalgo, err := c.op.GetBackwardFilterAlgorithm(handle.Cudnn(), x.TD(), y.TD(), w.FD(), prefbf, wspace)
	if err != nil {
		return err
	}

	c.perfbackfilt.Algo = bwdfalgo
	c.perfforward.Algo = fwdalgo
	c.perfbackdata.Algo = bwddalgo

	return nil
}

//this will set the fastest algo with the memory limit given if an error is returned no settings will be changed
func (c *Ops) setwspacelimit(handle *cudnn.Handler, x, y, w *tensor.Volume, wspacesize int) (uint, error) {
	var preff gocudnn.DeConvolutionForwardPref
	var prefbd gocudnn.DeConvBwdDataPref
	var prefbf gocudnn.DeConvBwdFilterPref
	preff.SpecifyWorkSpaceLimit()
	prefbd.SpecifyWorkSpaceLimit()
	prefbf.SpecifyWorkSpaceLimit()

	wspace := uint(wspacesize)
	fwdalgo, err := c.op.GetForwardAlgorithm(handle.Cudnn(), x.TD(), w.FD(), y.TD(), preff, wspace)
	if err != nil {
		return 0, err
	}

	fwdwspace, err := c.op.GetForwardWorkspaceSize(handle.Cudnn(), x.TD(), w.FD(), y.TD(), fwdalgo)
	if err != nil {
		return 0, err
	}
	bwddalgo, err := c.op.GetBackwardDataAlgorithm(handle.Cudnn(), w.FD(), y.TD(), x.TD(), prefbd, wspace)
	if err != nil {
		return 0, err
	}

	bwddspace, err := c.op.GetBackwardDataWorkspaceSize(handle.Cudnn(), w.FD(), y.TD(), x.TD(), bwddalgo)
	if err != nil {
		return 0, err
	}

	bwdfalgo, err := c.op.GetBackwardFilterAlgorithm(handle.Cudnn(), x.TD(), y.TD(), w.FD(), prefbf, wspace)
	if err != nil {
		return 0, err
	}
	bwdfspace, err := c.op.GetBackwardFilterWorkspaceSize(handle.Cudnn(), x.TD(), y.TD(), w.FD(), bwdfalgo)
	if err != nil {
		return 0, err
	}
	c.perfbackfilt.Algo = bwdfalgo
	c.perfforward.Algo = fwdalgo
	c.perfbackdata.Algo = bwddalgo
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
func (c *Ops) setfastest(handle *cudnn.Handler, x, y, w *tensor.Volume) (uint, error) {
	var preff gocudnn.DeConvolutionForwardPref
	var prefbd gocudnn.DeConvBwdDataPref
	var prefbf gocudnn.DeConvBwdFilterPref
	preff.PreferFastest()
	prefbd.PreferFastest()
	prefbf.PreferFastest()

	wspace := uint(0)
	fwdalgo, err := c.op.GetForwardAlgorithm(handle.Cudnn(), x.TD(), w.FD(), y.TD(), preff, wspace)
	if err != nil {
		return 0, err
	}

	fwdwspace, err := c.op.GetForwardWorkspaceSize(handle.Cudnn(), x.TD(), w.FD(), y.TD(), fwdalgo)
	if err != nil {
		return 0, err
	}
	bwddalgo, err := c.op.GetBackwardDataAlgorithm(handle.Cudnn(), w.FD(), y.TD(), x.TD(), prefbd, wspace)
	if err != nil {
		return 0, err
	}

	bwddspace, err := c.op.GetBackwardDataWorkspaceSize(handle.Cudnn(), w.FD(), y.TD(), x.TD(), bwddalgo)
	if err != nil {
		return 0, err
	}

	bwdfalgo, err := c.op.GetBackwardFilterAlgorithm(handle.Cudnn(), x.TD(), y.TD(), w.FD(), prefbf, wspace)
	if err != nil {
		return 0, err
	}
	bwdfspace, err := c.op.GetBackwardFilterWorkspaceSize(handle.Cudnn(), x.TD(), y.TD(), w.FD(), bwdfalgo)
	if err != nil {
		return 0, err
	}
	c.perfbackfilt.Algo = bwdfalgo
	c.perfforward.Algo = fwdalgo
	c.perfbackdata.Algo = bwddalgo
	largest := fwdwspace
	if bwddspace > largest {
		largest = bwddspace
	}
	if bwdfspace > largest {
		largest = bwdfspace
	}
	return largest, nil
}
