package cnn

import "github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"

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
func (c *Layer) LoadWValues(handle *cudnn.Handler, slice interface{}, length int) error {
	/*	ptr, err := gocudnn.MakeGoPointer(slice)
		if err != nil {
			return err
		}
	*/
	return c.w.LoadTValuesFromGoSlice(handle, slice, int32(length))
}

//LoadBiasValues will load a slice into cuda memory for the Weights.
func (c *Layer) LoadBiasValues(handle *cudnn.Handler, slice interface{}, length int) error {
	/*ptr, err := gocudnn.MakeGoPointer(slice)
	if err != nil {
		return err
	}*/
	return c.bias.LoadTValuesFromGoSlice(handle, slice, int32(length))
}

/*

//LoaddWValues will load a slice into cuda memory for the delta Weights.
func (c *Layer) LoaddWValues(handle *cudnn.Handler, slice interface{}) error {
	ptr, err := gocudnn.MakeGoPointer(slice)
	if err != nil {
		return err
	}
	return c.w.LoadDeltaTValues(handle, ptr)
}

/*
//BiasImgs returns 2d array of images
func (c *Layer) BiasImgs() ([][]image.Image, [][]image.Image, error) {
	return c.bias.Images()
}

*/

//WeightsFillSlice will fill a slice with the weight values
func (c *Layer) WeightsFillSlice(input interface{}, length int) error {
	return c.w.T().FillSlice(input, int32(length))
	//	return c.w.T().Memer().FillSlice(input)

}

//DeltaWeightsFillSlice will fill the weights with values
func (c *Layer) DeltaWeightsFillSlice(input interface{}, length int) error {
	return c.w.DeltaT().FillSlice(input, int32(length))
	//	return c.w.DeltaT().Memer().FillSlice(input)
}

//SetupWStatReducers builds the statistic reducers for the w part of the Weights and bias
func (c *Layer) SetupWStatReducers(handle *cudnn.Handler) (err error) {
	err = c.w.SetXStatReducers(handle)
	if err != nil {
		return err
	}
	err = c.bias.SetXStatReducers(handle)
	if err != nil {
		return err
	}

	return nil
}

//SetupDWStatReducers b builds the statistic reducers for the dw part of the Weights and bias
func (c *Layer) SetupDWStatReducers(handle *cudnn.Handler) (err error) {
	err = c.w.SetDXStatReducers(handle)
	if err != nil {
		return err
	}
	err = c.bias.SetDXStatReducers(handle)
	if err != nil {
		return err
	}

	return nil
}

/*

Weights

*/

//WMax returns the Max weight value for the layer.
func (c *Layer) WMax(handle *cudnn.Handler) (float32, error) {
	return c.w.MaxX(handle)
}

//WMin returns the Min weight value for the layer
func (c *Layer) WMin(handle *cudnn.Handler) (float32, error) {
	return c.w.MinX(handle)
}

// WAvg returns the avg weight value for the layer
func (c *Layer) WAvg(handle *cudnn.Handler) (float32, error) {
	return c.w.AvgX(handle)
}

// WNorm1 returns the norm1 weight value for the layer
func (c *Layer) WNorm1(handle *cudnn.Handler) (float32, error) {
	return c.w.Norm1X(handle)
}

// WNorm2 returns the norm2 weight value for the layer
func (c *Layer) WNorm2(handle *cudnn.Handler) (float32, error) {
	return c.w.Norm2X(handle)
}

/*

Bias

*/

//BMax returns the Max bias value for the layer
func (c *Layer) BMax(handle *cudnn.Handler) (float32, error) {
	return c.bias.MaxX(handle)
}

//BMin returns the Min bias value for the layer
func (c *Layer) BMin(handle *cudnn.Handler) (float32, error) {
	return c.bias.MinX(handle)
}

// BAvg returns the avg weight value for the layer
func (c *Layer) BAvg(handle *cudnn.Handler) (float32, error) {
	return c.bias.AvgX(handle)
}

// BNorm1 returns the norm1 bias value for the layer
func (c *Layer) BNorm1(handle *cudnn.Handler) (float32, error) {
	return c.bias.Norm1X(handle)
}

// BNorm2 returns the norm2 bias value for the layer
func (c *Layer) BNorm2(handle *cudnn.Handler) (float32, error) {
	return c.bias.Norm2X(handle)
}

/*

Delta Weights

*/

//DWMax returns the Max delta weight value for the layer
func (c *Layer) DWMax(handle *cudnn.Handler) (float32, error) {
	return c.w.MaxDX(handle)
}

//DWMin returns the Min delta weight value for the layer
func (c *Layer) DWMin(handle *cudnn.Handler) (float32, error) {
	return c.w.MinDX(handle)
}

// DWAvg returns the avg delta weight value for the layer
func (c *Layer) DWAvg(handle *cudnn.Handler) (float32, error) {
	return c.w.AvgDX(handle)
}

// DWNorm1 returns the norm1 delta weight value for the layer
func (c *Layer) DWNorm1(handle *cudnn.Handler) (float32, error) {
	return c.w.Norm1DX(handle)
}

// DWNorm2 returns the norm2 delta weight value for the layer
func (c *Layer) DWNorm2(handle *cudnn.Handler) (float32, error) {
	return c.w.Norm2DX(handle)
}

/*

Delta Bias

*/

//DBMax returns the Max delta bias value for the layer
func (c *Layer) DBMax(handle *cudnn.Handler) (float32, error) {
	return c.bias.MaxDX(handle)
}

//DBMin returns the Min delta bias value for the layer
func (c *Layer) DBMin(handle *cudnn.Handler) (float32, error) {
	return c.bias.MinDX(handle)
}

// DBAvg returns the avg delta bias value for the layer
func (c *Layer) DBAvg(handle *cudnn.Handler) (float32, error) {
	return c.bias.AvgDX(handle)
}

// DBNorm1 returns the norm1 delta bias value for the layer
func (c *Layer) DBNorm1(handle *cudnn.Handler) (float32, error) {
	return c.bias.Norm1DX(handle)
}

// DBNorm2 returns the norm2 delta bias value for the layer
func (c *Layer) DBNorm2(handle *cudnn.Handler) (float32, error) {
	return c.bias.Norm2DX(handle)
}
