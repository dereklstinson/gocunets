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
	nvutilhandle := nvutil.CreateHandle(nil, imode.LINEAR())
	decoder, err := jpeg.CreateDecoder(jpeghandle, stream)
	if err != nil {
		panic(err)
	}
	allocator, err := cudart.CreateMemManager(stream, 0)
	if err != nil {
		panic(err)
	}
	encoder, err := jpeg.CreateEncoder(jpeghandle, stream)
	if err != nil {
		panic(err)
	}
	encoder.SetQuality(100)

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

		/*
			data, err := ioutil.ReadAll(imgreader)
			if err != nil {
				panic(err)
			}
			chroma, w, h, err := nvjpeg.GetImageInfo(jpeghandle, data)
			fmt.Println(chroma.String(), w, h, err)
		*/
		var frmt nvjpeg.OutputFormat
		imgs[i], err = decoder.DecodeAIO(imgreader, frmt.RGB(), allocator)
		if err != nil {
			panic(err)
		}
	}
	hlpers := make([]nvutil.TileHelper, len(imgs))
	totalelements := int32(0)
	elementsperimage := make([]int32, 3)
	for i := range imgs {
		//	h, w := imgs[i].Size()
		//	fmt.Println("imgs[i].Size():", h, w)
		chans := imgs[i].GetChannels()
		for _, channel := range chans {
			fmt.Println("Chans for imgs[i]", channel)
		}
		var size npp.Size
		size.Set(32, 32)
		err = hlpers[i].Set(imgs[i], size, 16, 16)
		if err != nil {
			panic(err)
		}
		x := hlpers[i].GetDestNumOfElements()
		elementsperimage[i] = x
		totalelements += x
	}
	fmt.Println(totalelements)
	tiledspace := npp.Malloc8u(totalelements)
	offsets := make([]*npp.Uint8, 3)
	for i := range offsets {
		fmt.Println(i)

		offsets[i] = tiledspace.Offset(elementsperimage[i] * int32(i))
		err = hlpers[i].TiledCSHW(nvutilhandle, offsets[i], int(elementsperimage[i]))
		if err != nil {
			panic(err)
		}

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
