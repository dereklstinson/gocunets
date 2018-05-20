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

//MakeMakeFile is a hack that I want to get rid of, but it is a way to prototype .cu files when making neural networks, using devices with different compute capapbilities
func MakeMakeFile(directory string, dotCUname string, device cuda.Device) (string, error) {

	attmajor, err := device.Attr(cuda.DevAttrComputeCapabilityMajor)
	if err != nil {
		fmt.Println(err)
		return "", err
	}
	majstr := strconv.Itoa(attmajor)
	attminor, err := device.Attr(cuda.DevAttrComputeCapabilityMinor)
	if err != nil {
		fmt.Println(err)
		return "", err
	}
	minstr := strconv.Itoa(attminor)
	computecapability := majstr + minstr

	newname := dotCUname
	if strings.Contains(dotCUname, ".cu") {
		newname = strings.TrimSuffix(dotCUname, ".cu")

	} else {
		dotCUname = dotCUname + ".cu"
	}
	newname = newname + ".ptx"
	var some makefile
	//some.lines=make([]string,13)
	some.lines = make([]string, 2)
	some.lines[0] = "run:\n"
	some.lines[1] = "\t" + nvccarg + computecapability + nvccarg1 + computecapability + nvccarg2 + dotCUname + "\n"

	data := []byte(some.lines[0] + some.lines[1])
	err = os.MkdirAll(directory, 0644)
	if err != nil {
		fmt.Println("can't make directory", err)
		return "", err
	}
	err = ioutil.WriteFile(directory+"Makefile", data, 0644)
	if err != nil {
		fmt.Println(err)
		return "", err
	}
	newcommand := exec.Command("make")
	newcommand.Dir = directory
	time.Sleep(time.Millisecond)
	err = newcommand.Run()
	if err != nil {
		fmt.Println("*****Something Is wrong with the" + dotCUname + "file*******")
		return "", err
	}
	return newname, nil
}

//ReadPTXFile will return the ptx data
func ReadPTXFile(folderlocation, filename string) (string, error) {

	ptxdata, err := ioutil.ReadFile(folderlocation + filename)
	if err != nil {
		return "", err
	}
	return string(ptxdata), nil
}
