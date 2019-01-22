package gocunets

import (
	"encoding/json"
	"errors"
	"io"
	"strings"

	gocudnn "github.com/dereklstinson/GoCudnn"

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
	TestLoss float32   `json:"test_loss,omitempty"`
	Layers   []*Params `json:"Layers,omitempty"`
}

//GetTensorJSON gets the Tensor from data
func GetTensorJSON(data []byte) (*Tensor, error) {
	x := new(Tensor)
	err := json.Unmarshal(data, x)

	return x, err
}

//LoadWeights loads the weights from file returns the previous Loss for the weights.  It none was saved or an error then it will return 999999999
func (n *Network) LoadWeights(handle *cudnn.Handler, r io.Reader, size int64) (float32, error) {
	var err error
	data := make([]byte, size)
	_, err = r.Read(data)
	if err != nil {
		return 999999999, err
	}
	netparams, err := GetNetworkSavedTensorJSON(data)
	if err != nil {
		return 999999999, err
	}
	return n.LoadNetworkTensorparams(handle, netparams)
}

//LoadNetworkTensorparams - Loads the weights from Networksavedtensor
func (n *Network) LoadNetworkTensorparams(handle *cudnn.Handler, netsavedparams *NetworkSavedTensor) (float32, error) {
	if netsavedparams == nil {
		return 99999999, errors.New("netsavedparams is nil")
	}
	paramcounter := 0

	var err error
	for i := range n.layer {
		if n.layer[i].hasweights() {
			err = n.layer[i].loadparams(handle, netsavedparams.Layers[paramcounter])
			if err != nil {
				return 0, err
			}
			paramcounter++

		}
	}
	if paramcounter == 0 {
		return 9999999, errors.New("LoadNetworkTensorparams loaded nothing because n.layer[i].hasweights() didn't return true on anything")
	}
	return netsavedparams.TestLoss, nil
}

//SaveNetworkTensorParams saves network params to the writer
func (n *Network) SaveNetworkTensorParams(w io.Writer) (int64, error) {
	layers := make([]*layer, 0)
	for i := range n.layer {
		if n.layer[i].hasweights() {
			layers = append(layers, n.layer[i])
		}
	}
	netparams := make([]*Params, len(layers))
	var err error
	for i := range layers {
		netparams[i], err = layers[i].params()
		if err != nil {
			return 0, err
		}
	}
	x := NetworkSavedTensor{Layers: netparams}
	return x.WriteTo(w)
}

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
func (val *Tensor) LoadTensor(handle *cudnn.Handler, t *tensor.Volume) error {
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
		return t.LoadMem(handle, gptr)
	case flg.Float():
		x := utils.ToFloat32Slice(val.Values)
		gptr, err := gocudnn.MakeGoPointer(x)
		if err != nil {
			return err
		}
		return t.LoadMem(handle, gptr)
	case flg.Int32():
		x := utils.ToInt32Slice(val.Values)
		gptr, err := gocudnn.MakeGoPointer(x)
		if err != nil {
			return err
		}
		return t.LoadMem(handle, gptr)
	case flg.Int8():
		x := utils.ToInt8Slice(val.Values)
		gptr, err := gocudnn.MakeGoPointer(x)
		if err != nil {
			return err
		}
		return t.LoadMem(handle, gptr)
	case flg.UInt8():
		x := utils.ToUint8Slice(val.Values)
		gptr, err := gocudnn.MakeGoPointer(x)
		if err != nil {
			return err
		}
		return t.LoadMem(handle, gptr)
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