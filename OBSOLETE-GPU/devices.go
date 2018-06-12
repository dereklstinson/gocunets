package gpu

import (
	"fmt"

	"github.com/dereklstinson/GoCuNets/gpu/kerns"
	"github.com/dereklstinson/cuda"
)

//TODO: Eventually I would want to have a seperate "ptxinfo" string made for each layer

//Compute contains the stuff that is needed to use gpu compute
type Compute struct {
	Context *cuda.Context
	//Module  cuda.Module
	//Stream  cuda.Stream
	Device *cuda.Device
	Ptx    string
}

func FindAllDevices() ([]*cuda.Device, error) {
	return cuda.AllDevices()
}

func BuildCompute(device *cuda.Device, buffersize int) (Compute, error) {
	var newcompute Compute
	var err error
	newcompute.Device = device
	newcompute.Context, err = cuda.NewContext(device, buffersize)
	if err != nil {
		return newcompute, err
	}
	return newcompute, nil
}
func (compute *Compute) LoadPTXinfo(folderlocation, filename string) error {
	ptxname, err := kerns.MakeMakeFile(folderlocation, filename, *compute.Device)
	if err != nil {
		fmt.Println("error in making MakeFile")
		return err
	}
	compute.Ptx, err = kerns.ReadPTXFile(folderlocation, ptxname)
	if err != nil {
		fmt.Println("error in reading ptx file")
		return err
	}
	return nil
}
