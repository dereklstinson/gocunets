package gocunets

import (
	"encoding/json"
	"errors"
	"io"
	"strings"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/tensor"
	"github.com/dereklstinson/GoCuNets/utils"
)

//Weights are the weights that are used to save and load data to a layer
type Weights struct {
	Layer    string    `json:"layer,omitempty"`
	Format   string    `json:"format,omitempty"`
	Datatype string    `json:"datatype,omitempty"`
	Dims     []int32   `json:"dims,omitempty"`
	Stride   []int32   `json:"stride,omitempty"` //Stride is a holder for now
	Values   []float64 `json:"values,omitempty"`
}

//NetworkSavedWeights is a bunch of saved weights
type NetworkSavedWeights struct {
	Weights []Weights `json:"weights,omitempty"`
}

//GetWeightsJSON gets the weights from data
func GetWeightsJSON(data []byte) (*Weights, error) {
	x := new(Weights)
	err := json.Unmarshal(data, x)

	return x, err
}

//GetNetworkSavedWeightsJSON takes data and converts it to a NetworkSavedWeights
func GetNetworkSavedWeightsJSON(data []byte) (*NetworkSavedWeights, error) {
	x := new(NetworkSavedWeights)
	err := json.Unmarshal(data, x)

	return x, err
}

//WriteTo takes a writer and writes the NetworkSavedWeights in json format
func (val *NetworkSavedWeights) WriteTo(w io.Writer) (n int64, err error) {
	bytes, err := json.Marshal(val)
	if err != nil {
		return 0, err
	}
	x, err := w.Write(bytes)
	return int64(x), err
}

//WriteTo takes a writer and writes the weights in json format
func (val *Weights) WriteTo(w io.Writer) (n int64, err error) {
	bytes, err := json.Marshal(val)
	if err != nil {
		return 0, err
	}
	x, err := w.Write(bytes)
	return int64(x), err
}

//GetWeights gets the weight info from a tensor.Volume
func GetWeights(tensor *tensor.Volume, layer string) (Weights, error) {
	frmt, err := formattostring(tensor.Format())
	if err != nil {
		return Weights{}, err
	}
	dtype, err := datatypetostring(tensor.DataType())
	if err != nil {
		return Weights{}, err
	}
	dims := tensor.Dims()
	numofelements := utils.FindVolumeInt32(dims, nil)

	values := make([]float64, 0)
	var flg cudnn.DataTypeFlag
	switch tensor.DataType() {
	case flg.Double():
		x := make([]float64, numofelements)
		tensor.Memer().FillSlice(x)
		values = tofloat64(x)
	case flg.Float():
		x := make([]float32, numofelements)
		tensor.Memer().FillSlice(x)
		values = tofloat64(x)

	case flg.Int32():
		x := make([]int32, numofelements)
		tensor.Memer().FillSlice(x)
		values = tofloat64(x)
	case flg.Int8():
		x := make([]int8, numofelements)
		tensor.Memer().FillSlice(x)
		values = tofloat64(x)

	case flg.UInt8():
		x := make([]uint8, numofelements)
		tensor.Memer().FillSlice(x)
		values = tofloat64(x)
	}
	//	tensor.Memer().FillSlice()

	return Weights{
		Layer:    layer,
		Format:   frmt,
		Datatype: dtype,
		Dims:     dims,
		Values:   values,
	}, nil

}
func datatypetostring(dtype cudnn.DataType) (string, error) {
	var flg cudnn.DataTypeFlag
	switch dtype {
	case flg.Double():
		return "Double", nil
	case flg.Float():
		return "Float", nil
	case flg.Int32():
		return "Int32", nil
	case flg.Int8():
		return "Int8", nil
	case flg.UInt8():
		return "UInt8", nil

	}
	return "Unsupported", errors.New("Unsupported Datatype")
}
func stringtodatatype(dtype string) (cudnn.DataType, error) {
	dtype = strings.ToUpper(dtype)
	var flg cudnn.DataTypeFlag
	switch dtype {
	case "DOUBLE":
		return flg.Double(), nil
	case "FLOAT":
		return flg.Float(), nil
	case "INT32":
		return flg.Int32(), nil
	case "INT8":
		return flg.Int8(), nil
	case "UINT8":
		return flg.UInt8(), nil
	default:
		return cudnn.DataType(9999999), errors.New("Unsupported String")
	}
}
func stringtoformat(frmt string) (cudnn.TensorFormat, error) {
	var flgs cudnn.TensorFormatFlag
	frmt = strings.ToUpper(frmt)
	switch frmt {
	case "NCHW":
		return flgs.NCHW(), nil
	case "NHWC":
		return flgs.NHWC(), nil
	case "NCHWVECTC":
		return flgs.NCHWvectC(), nil
	}
	return cudnn.TensorFormat(999999), errors.New("Unsupported string name")
}
func formattostring(frmt cudnn.TensorFormat) (string, error) {
	var flgs cudnn.TensorFormatFlag
	switch frmt {
	case flgs.NCHW():
		return "NCHW", nil
	case flgs.NHWC():
		return "NHWC", nil
	case flgs.NCHWvectC():
		return "NCHWvectC", nil
	}
	return "Unsupported", errors.New("Unsupported Tensor Format")
}

func tofloat32(input interface{}) []float32 {
	return utils.ToFloat32Slice(input)
}
func tofloat64(input interface{}) []float64 {
	return utils.ToFLoat64Slice(input)
}
