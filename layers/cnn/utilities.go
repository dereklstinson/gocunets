package cnn

import (
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
