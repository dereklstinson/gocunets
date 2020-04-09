package gocunets

//import (
//	"bufio"
//	"encoding/json"
//	"errors"
//	"fmt"
//	"io"
//	"io/ioutil"
//	"strings"
//
//	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
//	"github.com/dereklstinson/GoCuNets/layers"
//	"github.com/dereklstinson/GoCuNets/utils"
//	gocudnn "github.com/dereklstinson/GoCudnn"
//	"github.com/dereklstinson/GoCudnn/gocu"
//)
//
////TensorInfo are the Tensor that are used to save and load data to a layer
//type TensorInfo struct {
//	Format   string  `json:"format,omitempty"`
//	Datatype string  `json:"datatype,omitempty"`
//	Dims     []int32 `json:"dims,omitempty"`
//	Stride   []int32 `json:"stride,omitempty"` //Stride is a holder for now
//	Data     []byte  `json:"data,omitempty"`
//}
//
////Params are a layers paramters or weights
//type Params struct {
//	Layer  string     `json:"layer,omitempty"`
//	Weight TensorInfo `json:"weight,omitempty"`
//	Bias   TensorInfo `json:"bias,omitempty"`
//	Xtra   TensorInfo `json:"xtra,omitempty"`
//}
//
////NetworkSavedTensor is a bunch of saved Tensor
//type NetworkSavedTensor struct {
//	TestLoss float32   `json:"test_loss,omitempty"`
//	Layers   []*Params `json:"Layers,omitempty"`
//}
//
////GetTensorJSON gets the Tensor from data
//func GetTensorJSON(data []byte) (*Tensor, error) {
//	x := new(Tensor)
//	err := json.Unmarshal(data, x)
//
//	return x, err
//}
//
////LoadWeights loads the weights from file returns the previous Loss for the weights.  It none was saved or an error then it will return 999999999
//func (n *Network) LoadWeights(handle *cudnn.Handler, r io.Reader, size int64) (float32, error) {
//	var err error
//	data := make([]byte, size)
//	_, err = r.Read(data)
//	if err != nil {
//		return 999999999, err
//	}
//	n.savedparams, err = GetNetworkSavedTensorJSON(data)
//	if err != nil {
//		return 999999999, err
//	}
//	return n.savedparams.TestLoss, nil
//}
//
////LoadNetworkTensorparams - Loads the weights from Networksavedtensor
//func (n *Network) LoadNetworkTensorparams(handle *cudnn.Handler, netsavedparams *NetworkSavedTensor) error {
//	if netsavedparams == nil {
//		return errors.New("netsavedparams is nil")
//	}
//	paramcounter := 0
//
//	var err error
//	for i := range n.layers {
//		if n.layers[i].hasweights() {
//			err = n.layers[i].loadparams(handle, netsavedparams.Layers[paramcounter])
//			if err != nil {
//				return err
//			}
//			paramcounter++
//
//		}
//	}
//	if paramcounter == 0 {
//		return errors.New("LoadNetworkTensorparams loaded nothing because n.layers[i].hasweights() didn't return true on anything")
//	}
//	return nil
//}
//
////SaveNetworkTensorParams saves network params to the writer
//func (n *Network) SaveNetworkTensorParams(w io.Writer) (int64, error) {
//	layers := make([]*Layer, 0)
//	for i := range n.layers {
//		if n.layers[i].hasweights() {
//			layers = append(layers, n.layers[i])
//		}
//	}
//	netparams := make([]*Params, len(layers))
//	var err error
//	for i := range layers {
//		netparams[i], err = layers[i].params()
//		if err != nil {
//			return 0, err
//		}
//	}
//	x := NetworkSavedTensor{Layers: netparams}
//	return x.WriteTo(w)
//}
//
////GetNetworkSavedTensorJSON takes data and converts it to a NetworkSavedTensor
//func GetNetworkSavedTensorJSON(data []byte) (*NetworkSavedTensor, error) {
//	x := new(NetworkSavedTensor)
//	err := json.Unmarshal(data, x)
//
//	return x, err
//}
//
////WriteTo takes a writer and writes the NetworkSavedTensor in json format
//func (val *NetworkSavedTensor) WriteTo(w io.Writer) (n int64, err error) {
//	bytes, err := json.Marshal(val)
//	if err != nil {
//		return 0, err
//	}
//	x, err := w.Write(bytes)
//	return int64(x), err
//}
//
////WriteTo takes a writer and writes the Tensor in json format
//func (val *TensorInfo) WriteTo(w io.Writer) (n int64, err error) {
//	bytes, err := json.Marshal(val)
//	if err != nil {
//		return 0, err
//	}
//	x, err := w.Write(bytes)
//	return int64(x), err
//}
//
////GetTensor gets the weight info from a tensor.Volume
//func getTensor(tensor *layers.Tensor, s gocu.Streamer) (TensorInfo, error) {
//	if tensor == nil {
//		return TensorInfo{}, errors.New("Tensor is ni")
//	}
//	frmt, err := formattostring(tensor.Format())
//	if err != nil {
//		return TensorInfo{}, err
//	}
//	dtype, err := datatypetostring(tensor.DataType())
//	if err != nil {
//		return TensorInfo{}, err
//	}
//	dims := tensor.Dims()
//	buf := tensor.Malloced.NewReadWriter(s)
//
//	data, err := ioutil.ReadAll(buf)
//	if err != nil {
//		return TensorInfo{}, err
//	}
//
//	//	tensor.FillSlice()
//
//	return TensorInfo{
//		Format:   frmt,
//		Datatype: dtype,
//		Dims:     dims,
//		Data:     data,
//	}, nil
//
//}
//func datatypetostring(dtype gocudnn.DataType) (string, error) {
//	var flg gocudnn.DataType
//	switch dtype {
//	case flg.Double():
//		return "Double", nil
//	case flg.Float():
//		return "Float", nil
//	case flg.Int32():
//		return "Int32", nil
//	case flg.Int8():
//		return "Int8", nil
//	case flg.UInt8():
//		return "UInt8", nil
//
//	}
//	return "Unsupported", errors.New("Unsupported Datatype")
//}
//func stringtodatatype(dtype string) (gocudnn.DataType, error) {
//	dtype = strings.ToUpper(dtype)
//	var flg gocudnn.DataType
//	switch dtype {
//	case "DOUBLE":
//		return flg.Double(), nil
//	case "FLOAT":
//		return flg.Float(), nil
//	case "INT32":
//		return flg.Int32(), nil
//	case "INT8":
//		return flg.Int8(), nil
//	case "UINT8":
//		return flg.UInt8(), nil
//	default:
//		return gocudnn.DataType(9999999), errors.New("Unsupported String")
//	}
//}
//
////LoadTensor will load the Tensor into a tensor.Volume passed.
//// Dims don't need to be the same, but the volume does need to be the same
//// Also Datatype Needs to be the same
//func (val *TensorInfo) LoadTensor(handle *cudnn.Handler, t *layers.Tensor) error {
//
//	tdtype, err := stringtodatatype(val.Datatype)
//	if err != nil {
//		return err
//	}
//	if tdtype != t.DataType() {
//		return errors.New("Datatype Not the same")
//	}
//	if utils.FindVolumeInt32(t.Dims(), nil) != utils.FindVolumeInt32(val.Dims, nil) {
//		return errors.New("LoadTensor-Volumes Don't Match")
//	}
//	rw := t.NewReadWriter(handle.Stream())
//
//	bw := bufio.NewWriter(rw)
//	if err != nil {
//		return nil
//	}
//	written, err := bw.Write(val.Data)
//
//	if err != nil {
//		fmt.Println("Written amount and size of data", written, len(val.Data))
//	}
//	return err
//}
//func stringtoformat(frmt string) (gocudnn.TensorFormat, error) {
//	var flgs gocudnn.TensorFormat
//	frmt = strings.ToUpper(frmt)
//	switch frmt {
//	case "NCHW":
//		return flgs.NCHW(), nil
//	case "NHWC":
//		return flgs.NHWC(), nil
//	case "NCHWVECTC":
//		return flgs.NCHWvectC(), nil
//	}
//	return gocudnn.TensorFormat(999999), errors.New("Unsupported string name")
//}
//func formattostring(frmt gocudnn.TensorFormat) (string, error) {
//	var flgs gocudnn.TensorFormat
//	switch frmt {
//	case flgs.NCHW():
//		return "NCHW", nil
//	case flgs.NHWC():
//		return "NHWC", nil
//	case flgs.NCHWvectC():
//		return "NCHWvectC", nil
//	}
//	return "Unsupported", errors.New("Unsupported Tensor Format")
//}
//
//func tofloat32(input interface{}) []float32 {
//	return utils.ToFloat32Slice(input)
//}
//func tofloat64(input interface{}) []float64 {
//	return utils.ToFLoat64Slice(input)
//}
//
