package cnn

import (
	"io"

	"github.com/dereklstinson/gocudnn/cudart/crtutil"

	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
)

//LoadWValues will load a slice into cuda memory for the Weights.
func (c *Layer) LoadWValues(handle *cudnn.Handler, slice interface{}, length int) error {
	/*	ptr, err := gocudnn.MakeGoPointer(slice)
		if err != nil {
			return err
		}
	*/
	return c.w.LoadValuesFromSLice(handle, slice, int32(length))
}

//LoadWvaluesEX takes a reader and coppies the bytes over to the weights
func (c *Layer) LoadWvaluesEX(handle *cudnn.Handler, r io.Reader) error {
	rw := crtutil.NewReadWriter(c.w, c.w.SIB(), handle.Stream())
	_, err := io.Copy(rw, r)
	return err
}

//LoadBiasValues will load a slice into cuda memory for the Weights.
func (c *Layer) LoadBiasValues(handle *cudnn.Handler, slice interface{}, length int) error {
	/*ptr, err := gocudnn.MakeGoPointer(slice)
	if err != nil {
		return err
	}*/
	return c.bias.LoadValuesFromSLice(handle, slice, int32(length))
}

//LoadBiasValuesEX takes a reader and coppies the bytes over to the bias
func (c *Layer) LoadBiasValuesEX(handle *cudnn.Handler, r io.Reader) error {
	rw := crtutil.NewReadWriter(c.bias, c.bias.SIB(), handle.Stream())
	_, err := io.Copy(rw, r)
	return err
}

//WeightsFillSlice will fill a slice with the weight values
func (c *Layer) WeightsFillSlice(h *cudnn.Handler, input interface{}, length int) error {
	return c.w.FillSlice(h, input)
	//	return c.w.T().Memer().FillSlice(input)

}

//DeltaWeightsFillSlice will fill the weights with values
func (c *Layer) DeltaWeightsFillSlice(h *cudnn.Handler, input interface{}, length int) error {
	return c.w.FillSlice(h, input)
	//	return c.w.DeltaT().Memer().FillSlice(input)
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
	return c.dw.MaxX(handle)
}

//DWMin returns the Min delta weight value for the layer
func (c *Layer) DWMin(handle *cudnn.Handler) (float32, error) {
	return c.dw.MinX(handle)
}

// DWAvg returns the avg delta weight value for the layer
func (c *Layer) DWAvg(handle *cudnn.Handler) (float32, error) {
	return c.dw.AvgX(handle)
}

// DWNorm1 returns the norm1 delta weight value for the layer
func (c *Layer) DWNorm1(handle *cudnn.Handler) (float32, error) {
	return c.dw.Norm1X(handle)
}

// DWNorm2 returns the norm2 delta weight value for the layer
func (c *Layer) DWNorm2(handle *cudnn.Handler) (float32, error) {
	return c.dw.Norm2X(handle)
}

/*

Delta Bias

*/

//DBMax returns the Max delta bias value for the layer
func (c *Layer) DBMax(handle *cudnn.Handler) (float32, error) {
	return c.dbias.MaxX(handle)
}

//DBMin returns the Min delta bias value for the layer
func (c *Layer) DBMin(handle *cudnn.Handler) (float32, error) {
	return c.dbias.MinX(handle)
}

// DBAvg returns the avg delta bias value for the layer
func (c *Layer) DBAvg(handle *cudnn.Handler) (float32, error) {
	return c.dbias.AvgX(handle)
}

// DBNorm1 returns the norm1 delta bias value for the layer
func (c *Layer) DBNorm1(handle *cudnn.Handler) (float32, error) {
	return c.dbias.Norm1X(handle)
}

// DBNorm2 returns the norm2 delta bias value for the layer
func (c *Layer) DBNorm2(handle *cudnn.Handler) (float32, error) {
	return c.dbias.Norm2X(handle)
}
