package gocunets

import (
	"fmt"
	"strconv"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn/convolution"
	"github.com/dereklstinson/GoCuNets/layers"
)

const debugconvolutionperformance = false

//ConvolutionPerformance contains all the performance stats for different convolution algos
type ConvolutionPerformance struct {
	Layer *layer
	Fwd   []convolution.ForwardPerformance
	BwdD  []convolution.BackDataPerformance
	BwdF  []convolution.BackFilterPerformance
}

//GetFastestWSpaceSizes will return the largest workspace size of the whole network.  I am thinking that this could be used for the entire network.  I haven't used it yet though.
func GetFastestWSpaceSizes(perfs []ConvolutionPerformance) (fwd, bwdd, bwdf uint) {
	var (
		fsize  uint
		bdsize uint
		bfsize uint
	)
	for i := range perfs {
		fsize, bdsize, bfsize = perfs[i].FindFastestWspace()
		if fwd < fsize {
			fwd = fsize
		}
		if bwdd < bdsize {
			bwdd = bdsize
		}
		if bwdf < bfsize {
			bwdf = bfsize
		}
	}
	return fwd, bwdd, bwdf

}

//SetWSpaceSize sets the fastest algorithm for the convolution based on the limit of the workspace.
func (m *Network) SetWSpaceSize(handle *cudnn.Handler, wspacefwd, wspacebwdd, wspacebwdf uint, perfs []ConvolutionPerformance) {
	if wspacebwdd != 0 {
		m.usingwsbwdd = true
	}
	if wspacebwdf != 0 {
		m.usingwsbwdf = true
	}
	if wspacefwd != 0 {
		m.usingwsfwd = true
	}
	for i := range perfs {
		perfs[i].SetAlgoPerWspacesize(handle, wspacefwd, wspacebwdd, wspacebwdf)
	}
}

//SetAlgoPerWspacesize will set the fastest algorithm based on the size fo the workspaced sent
func (c *ConvolutionPerformance) SetAlgoPerWspacesize(handle *cudnn.Handler, wspacefwd, wspacebwdd, wspacebwdf uint) {
	if c.Fwd != nil {
		for i := range c.Fwd {
			if wspacefwd >= (c.Fwd[i].Memory) {
				c.Layer.setcudnnperformancefwd(handle, c.Fwd[i])
				break
			}
		}

	}
	if c.BwdD != nil {
		for i := range c.BwdD {
			if wspacebwdd >= uint(c.BwdD[i].Memory) {
				c.Layer.setcudnnperformancebwdd(handle, c.BwdD[i])
				break
			}
		}
	}
	if c.BwdF != nil {
		for i := range c.BwdD {
			if wspacebwdf >= uint(c.BwdF[i].Memory) {
				c.Layer.setcudnnperformancebwdf(handle, c.BwdF[i])
				break
			}
		}
	}
}

//FindFastestWspace retunrs the wspace of the fastest algorithm
func (c *ConvolutionPerformance) FindFastestWspace() (fwdwspace, bwddwspace, bwdfwspace uint) {

	if c.Fwd != nil {

		if fwdwspace < uint(c.Fwd[0].Memory) {
			fwdwspace = uint(c.Fwd[0].Memory)
		}
	}
	if c.BwdD != nil {
		if bwddwspace < uint(c.BwdD[0].Memory) {
			bwddwspace = uint(c.BwdD[0].Memory)
		}
	}
	if c.BwdF != nil {
		if bwdfwspace < uint(c.BwdF[0].Memory) {
			bwdfwspace = uint(c.BwdF[0].Memory)
		}
	}
	return fwdwspace, bwddwspace, bwdfwspace
}

//SetWorkSpaces will set the workspaces for the network.
func (m *Network) SetWorkSpaces(fwd, bwdd, bwdf *nvidia.Malloced) {
	m.wsfwd = fwd
	m.wsbwdd = bwdd
	m.wsbwdf = bwdf
}
func (m *Network) performance(handle *cudnn.Handler, x, y *layers.IO, workspace *nvidia.Malloced) ([]ConvolutionPerformance, error) {

	//	var err error
	performers := make([]ConvolutionPerformance, 0)
	fwd, bwdd, bwdf, err := m.layer[0].getcudnnperformance(handle, x, m.training.mem[0], workspace)
	if err != nil {
		if debugconvolutionperformance {
			dbprt("(m *Network) performance(handle *cudnn.Handler, x, y *layers.IO) ([]ConvolutionPerformance, error)")
		}
		return nil, wraperror("cudnn performance index:"+strconv.Itoa(0), err)
	}
	if bwdf != nil {
		performers = append(performers, ConvolutionPerformance{
			Layer: m.layer[0],
			Fwd:   fwd,
			BwdD:  bwdd,
			BwdF:  bwdf,
		})
	}
	lnum := len(m.layer)
	for i := 1; i < lnum-1; i++ {
		if debugconvolutionperformance {
			fmt.Println("index " + strconv.Itoa(i))
		}
		fwd1, bwdd1, bwdf1, err := m.layer[i].getcudnnperformance(handle, m.training.mem[i-1], m.training.mem[i], workspace)
		if err != nil {
			return nil, wraperror("cudnn performance index:"+strconv.Itoa(i), err)
		}
		if bwdf1 != nil {
			performers = append(performers, ConvolutionPerformance{
				Layer: m.layer[i],
				Fwd:   fwd1,
				BwdD:  bwdd1,
				BwdF:  bwdf1,
			})
		}
	}
	if debugconvolutionperformance {
		fmt.Println("index " + strconv.Itoa(lnum-1))
	}
	fwd1, bwdd1, bwdf1, err := m.layer[lnum-1].getcudnnperformance(handle, m.training.mem[lnum-2], y, workspace)
	if err != nil {
		return nil, wraperror("cudnn performance index:"+strconv.Itoa(lnum-1), err)
	}
	if bwdf1 != nil {
		performers = append(performers, ConvolutionPerformance{
			Layer: m.layer[lnum-1],
			Fwd:   fwd1,
			BwdD:  bwdd1,
			BwdF:  bwdf1,
		})
	}
	return performers, nil

}

func dbprt(comment string) {
	fmt.Sprint()
	fmt.Printf("Dubbing:{\n"+
		"File: gocunets_performance.go{\n"+
		"%s\n"+
		"}\n"+
		"})\n", comment)
}
