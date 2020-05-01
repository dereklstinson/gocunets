package gocunets

//"strconv"
//	"github.com/dereklstinson/gocunets/layers"

/*
const debugconvolutionperformance = false

//ConvolutionPerformance contains all the performance stats for different convolution algos
type ConvolutionPerformance struct {
	Layer *Layer
	Fwd   []convolution.ForwardPerformance
	BwdD  []convolution.BackDataPerformance
	BwdF  []convolution.BackFilterPerformance
	DFwd  []deconvolution.ForwardPerformance
	DBwdD []deconvolution.BackDataPerformance
	DBwdF []deconvolution.BackFilterPerformance
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
				c.Layer.setcudnnperformancefwd(c.Fwd[i])
				break
			}
		}

	}
	if c.BwdD != nil {
		for i := range c.BwdD {
			if wspacebwdd >= uint(c.BwdD[i].Memory) {
				c.Layer.setcudnnperformancebwdd(c.BwdD[i])
				break
			}
		}
	}
	if c.BwdF != nil {
		for i := range c.BwdD {
			if wspacebwdf >= uint(c.BwdF[i].Memory) {
				c.Layer.setcudnnperformancebwdf(c.BwdF[i])
				break
			}
		}
	}
}

func dbprt(comment string) {

	fmt.Printf("Dubbing:{\n"+
		"File: gocunets_performance.go{\n"+
		"%s\n"+
		"}\n"+
		"})\n", comment)
}
*/
