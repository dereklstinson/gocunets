package kerns

import "C"
import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/dereklstinson/cuda"
)

const nvccarg = "nvcc --gpu-architecture=compute_"
const nvccarg1 = " --gpu-code=compute_"
const nvccarg2 = " --ptx "

type makefile struct {
	lines []string
}

func MakeMakeFile(directory string, dotCUname string, device cuda.Device) string {

	attmajor, err := device.Attr(cuda.DevAttrComputeCapabilityMajor)
	if err != nil {
		fmt.Println(err)
	}
	majstr := strconv.Itoa(attmajor)
	attminor, err := device.Attr(cuda.DevAttrComputeCapabilityMinor)
	if err != nil {
		fmt.Println(err)
	}
	minstr := strconv.Itoa(attminor)
	computecapability := majstr + minstr

	newname := dotCUname
	if strings.Contains(dotCUname, ".cu") {
		newname = strings.TrimSuffix(dotCUname, ".cu")
	} else {
		dotCUname = dotCUname + ".cu"
	}
	var some makefile
	//some.lines=make([]string,13)
	some.lines = make([]string, 2)
	some.lines[0] = "run:\n"
	some.lines[1] = "\t" + nvccarg + computecapability + nvccarg1 + computecapability + nvccarg2 + dotCUname + "\n"

	data := []byte(some.lines[0] + some.lines[1])
	err = os.MkdirAll(directory, 0644)
	if err != nil {
		fmt.Println(err)
		panic(err)
	}
	err = ioutil.WriteFile(directory+"Makefile", data, 0644)
	if err != nil {
		fmt.Println(err)
		panic(err)
	}
	newcommand := exec.Command("make")
	newcommand.Dir = directory
	time.Sleep(time.Millisecond)
	err = newcommand.Run()
	if err != nil {
		fmt.Println("*****Something Is wrong with the" + dotCUname + "file*******")
		panic(err)
	}
	return newname
}

func LoadPTXFile(filelocation string) string {

	ptxdata, err := ioutil.ReadFile(filelocation)
	if err != nil {
		panic(err)
	}
	return string(ptxdata)
}
