package cnn

import (
	"github.com/dereklstinson/GoCuNets/cudnn"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

/*
//SaveImagesToFile saves images do file
func (c *Layer) SaveImagesToFile(dir string) error {
	return c.w.SaveImagesToFile(dir)
}

//WeightImgs returns 2d array of images
func (c *Layer) WeightImgs() ([][]image.Image, [][]image.Image, error) {
	return c.w.Images()
}
*/

//LoadWValues will load a slice into cuda memory for the Weights.
func (c *Layer) LoadWValues(slice interface{}) error {
	ptr, err := gocudnn.MakeGoPointer(slice)
	if err != nil {
		return err
	}
	return c.w.LoadTValues(ptr)
}

//LoadBiasValues will load a slice into cuda memory for the Weights.
func (c *Layer) LoadBiasValues(slice interface{}) error {
	ptr, err := gocudnn.MakeGoPointer(slice)
	if err != nil {
		return err
	}
	return c.bias.LoadTValues(ptr)
}

//LoaddWValues will load a slice into cuda memory for the delta Weights.
func (c *Layer) LoaddWValues(slice interface{}) error {
	ptr, err := gocudnn.MakeGoPointer(slice)
	if err != nil {
		return err
	}
	return c.w.LoadDeltaTValues(ptr)
}

/*
//BiasImgs returns 2d array of images
func (c *Layer) BiasImgs() ([][]image.Image, [][]image.Image, error) {
	return c.bias.Images()
}
*/

//WeightsFillSlice will fill a slice with the weight values
func (c *Layer) WeightsFillSlice(input interface{}) error {
	return c.w.T().Memer().FillSlice(input)

}

//DeltaWeightsFillSlice will fill the weights with values
func (c *Layer) DeltaWeightsFillSlice(input interface{}) error {
	return c.w.DeltaT().Memer().FillSlice(input)
}

//SetupMinMaxReducers builds the minmax reduces
func (c *Layer) SetupMinMaxReducers(handle *cudnn.Handler, batches bool) (err error) {
	err = c.w.SetMinMaxReducers(handle, batches)
	if err != nil {
		return err
	}
	err = c.bias.SetMinMaxReducers(handle, batches)
	if err != nil {
		return err
	}
	return nil
}

//WMax returns the Max weight value per neuron
func (c *Layer) WMax(handle *cudnn.Handler) ([]float32, error) {
	return c.w.MaxX(handle)
}

//DWMax returns the Max delta weight value per neuron
func (c *Layer) DWMax(handle *cudnn.Handler) ([]float32, error) {
	return c.w.MaxDX(handle)
}

//BMax returns the Max bias value per neuron
func (c *Layer) BMax(handle *cudnn.Handler) ([]float32, error) {
	return c.bias.MaxX(handle)
}

//DBMax returns the Max delta bias value per neuron
func (c *Layer) DBMax(handle *cudnn.Handler) ([]float32, error) {
	return c.bias.MaxDX(handle)
}

//WMin returns the Min weight value per neuron
func (c *Layer) WMin(handle *cudnn.Handler) ([]float32, error) {
	return c.w.MinX(handle)
}

//DWMin returns the Min delta weight value per neuron
func (c *Layer) DWMin(handle *cudnn.Handler) ([]float32, error) {
	return c.w.MinDX(handle)
}

//BMin returns the Min bias value per neuron
func (c *Layer) BMin(handle *cudnn.Handler) ([]float32, error) {
	return c.bias.MinX(handle)
}

//DBMin returns the Min delta bias value per neuron
func (c *Layer) DBMin(handle *cudnn.Handler) ([]float32, error) {
	return c.bias.MinDX(handle)
}
