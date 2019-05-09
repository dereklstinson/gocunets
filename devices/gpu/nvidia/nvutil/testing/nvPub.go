package main

import (
	"fmt"
	"os"
	"runtime"

	//"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/GoCudnn/gocu"
	"github.com/dereklstinson/GoCudnn/npp"

	"github.com/dereklstinson/GoCudnn/nvjpeg"

	"github.com/dereklstinson/GoCudnn/cudart"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/jpeg"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/nvutil"
)

func main() {
	runtime.LockOSThread()
	fmt.Println("Start")
	jpeghandle, err := jpeg.MakeHandle(0, 0)
	if err != nil {
		panic(err)
	}

	stream, err := cudart.CreateBlockingStream()
	if err != nil {
		panic(err)
	}
	var imode npp.InterpolationMode
	nvutilhandle := nvutil.CreateHandle(nil, imode.NN())
	decoder, err := jpeg.CreateDecoder(jpeghandle, stream)
	if err != nil {
		panic(err)
	}
	allocator, err := cudart.CreateMemManager(stream, 0)
	if err != nil {
		panic(err)
	}

	imgs := make([]*jpeg.Image, 3)
	for i := 0; i < 3; i++ {
		var imgreader *os.File
		if i == 0 {
			imgreader, err = os.Open("img1.JPG")
			if err != nil {
				panic(err)
			}
		}
		if i == 1 {
			imgreader, err = os.Open("img2.JPG")
			if err != nil {
				panic(err)
			}
		}
		if i == 2 {
			imgreader, err = os.Open("img3.JPG")
			if err != nil {
				panic(err)
			}
		}

		defer imgreader.Close()

		var frmt nvjpeg.OutputFormat
		imgs[i], err = decoder.DecodeAIO(imgreader, frmt.RGB(), allocator)
		if err != nil {
			panic(err)
		}
	}
	hlpers := nvutil.CreateTileHelpers(len(imgs))

	var window npp.Size
	window.Set(32, 32)

	err = nvutil.BatchTileSet(hlpers, imgs, window)
	if err != nil {
		panic(err)
	}
	totalelements := nvutil.BatchTileTotalElements(hlpers)
	fmt.Println("Total Amount of Elemenets,", totalelements)
	tiledspace := new(npp.Uint8)
	err = cudart.MallocManagedGlobal(tiledspace, (uint)(totalelements))
	if err != nil {
		panic(err)
	}

	err = nvutil.BatchTiles(nvutilhandle, hlpers, tiledspace, stream)
	if err != nil {
		panic(err)
	}
	databack := make([]byte, totalelements)
	databackptr, err := gocu.MakeGoMem(databack)
	if err != nil {
		panic(err)
	}
	err = nvidia.Memcpy(databackptr, tiledspace, (uint)(totalelements))
	fmt.Println(databack)
	if err != nil {
		panic(err)
	}
}
