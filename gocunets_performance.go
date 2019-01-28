package gocunets

import (
	"strconv"

	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/convolution"
	"github.com/dereklstinson/GoCuNets/layers"
)

//ConvolutionPerformance contains all the performance stats for different convolution algos
type ConvolutionPerformance struct {
	Layer *layer
	Fwd   []convolution.ForwardPerformance
	BwdD  []convolution.BackDataPerformance
	BwdF  []convolution.BackFilterPerformance
}

//GetFastestWSpaceSize will return the largest workspace size of the whole network.  I am thinking that this could be used for the entire network.  I haven't used it yet though.
func GetFastestWSpaceSize(perfs []ConvolutionPerformance) (wspacesize cudnn.SizeT) {
	var size cudnn.SizeT
	for i := range perfs {
		size = perfs[i].FindFastestWspace()
		if wspacesize < size {
			wspacesize = size
		}
	}
	return wspacesize

}

//SetWSpaceSize sets the fastest algorithm for the convolution based on the limit of the workspace.
func SetWSpaceSize(handle *cudnn.Handler, wspacesize cudnn.SizeT, perfs []ConvolutionPerformance) {
	for i := range perfs {
		perfs[i].SetAlgoPerWspacesize(handle, wspacesize)
	}
}

//SetAlgoPerWspacesize will set the fastest algorithm based on the size fo the workspaced sent
func (c *ConvolutionPerformance) SetAlgoPerWspacesize(handle *cudnn.Handler, wspacesize cudnn.SizeT) {
	if c.Fwd != nil {
		for i := range c.Fwd {
			if wspacesize >= cudnn.SizeT(c.Fwd[i].Memory) {
				c.Layer.setcudnnperformancefwd(handle, c.Fwd[i])
				break
			}
		}

	}
	if c.BwdD != nil {
		for i := range c.BwdD {
			if wspacesize >= cudnn.SizeT(c.BwdD[i].Memory) {
				c.Layer.setcudnnperformancebwdd(handle, c.BwdD[i])
				break
			}
		}
	}
	if c.BwdF != nil {
		for i := range c.BwdD {
			if wspacesize >= cudnn.SizeT(c.BwdF[i].Memory) {
				c.Layer.setcudnnperformancebwdf(handle, c.BwdF[i])
				break
			}
		}
	}
}

//FindFastestWspace retunrs the wspace of the fastest algorithm
func (c *ConvolutionPerformance) FindFastestWspace() (wspacesize cudnn.SizeT) {

	if c.Fwd != nil {
		if wspacesize < cudnn.SizeT(c.Fwd[0].Memory) {
			wspacesize = cudnn.SizeT(c.Fwd[0].Memory)
		}
	}
	if c.BwdD != nil {
		if wspacesize < cudnn.SizeT(c.BwdD[0].Memory) {
			wspacesize = cudnn.SizeT(c.BwdD[0].Memory)
		}
	}
	if c.BwdF != nil {
		if wspacesize < cudnn.SizeT(c.BwdF[0].Memory) {
			wspacesize = cudnn.SizeT(c.BwdF[0].Memory)
		}
	}
	return wspacesize
}
func (m *Network) performance(handle *cudnn.Handler, x, y *layers.IO) ([]ConvolutionPerformance, error) {
	//	var err error
	performers := make([]ConvolutionPerformance, 0)
	fwd, bwdd, bwdf, err := m.layer[0].getcudnnperformance(handle, x, m.mem[0])
	if err != nil {
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

		fwd1, bwdd1, bwdf1, err := m.layer[i].getcudnnperformance(handle, m.mem[i-1], m.mem[i])
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

	fwd1, bwdd1, bwdf1, err := m.layer[lnum-1].getcudnnperformance(handle, m.mem[lnum-2], y)
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
