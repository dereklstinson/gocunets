package gocunets

import (
	"encoding/json"
	"errors"
	"io"
	"strings"

	"github.com/dereklstinson/GoCudnn"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/tensor"
	"github.com/dereklstinson/GoCuNets/utils"
)

//Tensor are the Tensor that are used to save and load data to a layer
type Tensor struct {
	Format   string    `json:"format,omitempty"`
	Datatype string    `json:"datatype,omitempty"`
	Dims     []int32   `json:"dims,omitempty"`
	Stride   []int32   `json:"stride,omitempty"` //Stride is a holder for now
	Values   []float64 `json:"values,omitempty"`
}

//Params are a layers paramters or weights
type Params struct {
	Layer  string `json:"layer,omitempty"`
	Weight Tensor `json:"weight,omitempty"`
	Bias   Tensor `json:"bias,omitempty"`
	Xtra   Tensor `json:"xtra,omitempty"`
}

//NetworkSavedTensor is a bunch of saved Tensor
type NetworkSavedTensor struct {
	Tensor []*Params `json:"Tensor,omitempty"`
}

//GetTensorJSON gets the Tensor from data
func GetTensorJSON(data []byte) (*Tensor, error) {
	x := new(Tensor)
	err := json.Unmarshal(data, x)

	return x, err
}

/*
func (n *Network) SaveNetworkTensorParams(w io.writer) error {
	bnlayers := n.BatchNorms()

}
*/

//GetNetworkSavedTensorJSON takes data and converts it to a NetworkSavedTensor
func GetNetworkSavedTensorJSON(data []byte) (*NetworkSavedTensor, error) {
	x := new(NetworkSavedTensor)
	err := json.Unmarshal(data, x)

	return x, err
}

//WriteTo takes a writer and writes the NetworkSavedTensor in json format
func (val *NetworkSavedTensor) WriteTo(w io.Writer) (n int64, err error) {
	bytes, err := json.Marshal(val)
	if err != nil {
		return 0, err
	}
	x, err := w.Write(bytes)
	return int64(x), err
}

//WriteTo takes a writer and writes the Tensor in json format
func (val *Tensor) WriteTo(w io.Writer) (n int64, err error) {
	bytes, err := json.Marshal(val)
	if err != nil {
		return 0, err
	}
	x, err := w.Write(bytes)
	return int64(x), err
}

//GetTensor gets the weight info from a tensor.Volume
func getTensor(tensor *tensor.Volume) (Tensor, error) {
	if tensor == nil {
		return Tensor{}, errors.New("Tensor is ni")
	}
	frmt, err := formattostring(tensor.Format())
	if err != nil {
		return Tensor{}, err
	}
	dtype, err := datatypetostring(tensor.DataType())
	if err != nil {
		return Tensor{}, err
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

	return Tensor{
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

//LoadTensor will load the Tensor into a tensor.Volume passed.
// Dims don't need to be the same, but the volume does need to be the same
// Also Datatype Needs to be the same
func (val *Tensor) LoadTensor(t *tensor.Volume) error {
	var flg cudnn.DataTypeFlag
	tdtype, err := stringtodatatype(val.Datatype)
	if err != nil {
		return err
	}
	if tdtype != t.DataType() {
		return errors.New("Datatype Not the same")
	}
	if utils.FindVolumeInt32(t.Dims(), nil) != utils.FindVolumeInt32(val.Dims, nil) {
		return errors.New("LoadTensor-Volumes Don't Match")
	}
	switch tdtype {
	case flg.Double():
		x := utils.ToFLoat64Slice(val.Values)
		gptr, err := gocudnn.MakeGoPointer(x)
		if err != nil {
			return err
		}
		return t.LoadMem(gptr)
	case flg.Float():
		x := utils.ToFloat32Slice(val.Values)
		gptr, err := gocudnn.MakeGoPointer(x)
		if err != nil {
			return err
		}
		return t.LoadMem(gptr)
	case flg.Int32():
		x := utils.ToInt32Slice(val.Values)
		gptr, err := gocudnn.MakeGoPointer(x)
		if err != nil {
			return err
		}
		return t.LoadMem(gptr)
	case flg.Int8():
		x := utils.ToInt8Slice(val.Values)
		gptr, err := gocudnn.MakeGoPointer(x)
		if err != nil {
			return err
		}
		return t.LoadMem(gptr)
	case flg.UInt8():
		x := utils.ToUint8Slice(val.Values)
		gptr, err := gocudnn.MakeGoPointer(x)
		if err != nil {
			return err
		}
		return t.LoadMem(gptr)
	}
	return errors.New("Unsupported Type")
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
