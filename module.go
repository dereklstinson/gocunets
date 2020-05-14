package gocunets

import (
	"errors"
	"fmt"
)

//Module is a wrapper around a neural network or set of operations
type Module interface {
	ID() int64
	Forward() error
	Backward() error
	OutputDims() ([]int32, error)
	Inference() error
	//GetWeights()  tensor order should be same as GetDeltaWeights().  Can return nil
	GetWeights() []*Tensor
	//GetDeltaWeights() tensor order should be same as GetWeights().  Can return nil
	//Some trainers don't require deltaweights.  So, it is not required.
	GetDeltaWeights() []*Tensor
	InitHiddenLayers() (err error)
	InitWorkspace() (err error)
	GetTensorX() (x *Tensor)
	GetTensorDX() (dx *Tensor)
	GetTensorY() (y *Tensor)
	GetTensorDY() (dy *Tensor)
	SetTensorX(x *Tensor)
	SetTensorDX(dx *Tensor)
	SetTensorY(y *Tensor)
	SetTensorDY(dy *Tensor)
}

var moduleforwarddebugging bool
var modulebackwarddatadebugging bool
var modulebackwardfilterdebugging bool
var moduleconcatdebugging bool
var moduleactivationdebugging bool

//ModuleActivationDebug is for debugging
func ModuleActivationDebug() {
	moduleactivationdebugging = true
}

//ModuleConcatDebug is for debugging
func ModuleConcatDebug() {
	moduleconcatdebugging = true
}

//ModuleForwardDebug is for debugging
func ModuleForwardDebug() {
	moduleforwarddebugging = true
}

//ModuleBackwardDataDebug is for debugging
func ModuleBackwardDataDebug() {
	modulebackwarddatadebugging = true
}

//ModuleBackwardFilterDebug is for debugging
func ModuleBackwardFilterDebug() {
	modulebackwardfilterdebugging = true
}

//ModuleDebugMode sets a flag that prints outputs of inner module outputs

//This func is way to complicated.
//But if the filterdim is odd.  It is suited for odd in odd out.
//With an odd input being 2^n + 1 output will be 2^(n-1) +1.
//If the filter dim is even then it is suited for even in even out. With an even input being 2^n. output will be 2^(n-1)
//it only does stride 2 for right now.  It will do a stride of 1 in a little while when I get my spreadsheet out.

func recommendedpaddilation(filterdim, index, stride, offset int32) (dilation, pad int32, err error) {
	if filterdim%2 == 0 {
		if stride == 1 {
			dilation = 2 * (index + 1)
			if (((filterdim-1)*dilation + 1 + offset) % 2) != 0 {
				return -1, -1, fmt.Errorf("(((filterdim-1)*dilation +1 + offset) module 2) != 0, (((%v-1)*%v)+1+%v)", filterdim, dilation, offset)
			}
			pad = ((filterdim-1)*dilation + 1 + offset) / 2

		} else {
			dilation = 2*index + 1
			pad = ((filterdim-1)*dilation + 1 + offset) / 2
		}

	} else {
		dilation = index + 1
		pad = ((filterdim-1)*dilation + 1 + offset) / 2
	}

	if pad < 0 {
		return -1, -1, errors.New("recommendedpaddilation params givin give pad< 0")
	}
	return dilation, pad, nil

}
func checkparamsconv(i, f, p, s, d int32) (isgood bool) {
	if (i+2*p-(((f-1)*d)+1))%s == 0 {
		isgood = true
	}
	return isgood
}

func dimoutput(i, f, p, s, d int32) (o int32) {
	////if p = (((f - 1) * d) + 1+offset)/2  &&  s=2 && f=2
	o = 1 + (i+2*p-(((f-1)*d)+1))/s
	return o
}
func dimoutputreverse(i, f, p, s, d int32) (o int32) {
	o = (i-1)*s - 2*p + ((f - 1) * d) + 1
	//if p = (((f - 1) * d) + 1+offset)/2  &&  s=2 && f=2
	//then p = (d+1 + offset)/2
	//then o = 2(i-1) - (d+1 + offset)/2 + d+1
	//
	return o
}

//SimpleModuleNetwork is a simple module network
type SimpleModuleNetwork struct {
	id         int64
	C          *Concat           `json:"c,omitempty"`
	Modules    []Module          `json:"modules,omitempty"`
	Output     *OutputModule     `json:"output,omitempty"`
	Classifier *ClassifierModule `json:"classifier,omitempty"`
	b          *Builder

	//	x, dx, y, dy        *Tensor
	//	firstinithiddenfirstinithidden    bool
	//	firstinitworkspace bool
	//	firstfindoutputdims bool
}

//CreateSimpleModuleNetwork a simple module network
func CreateSimpleModuleNetwork(id int64, b *Builder) (smn *SimpleModuleNetwork) {
	smn = new(SimpleModuleNetwork)
	smn.b = b
	smn.id = id

	return smn
}
func (m *SimpleModuleNetwork) GetWeights() []*Tensor {
	weights := make([]*Tensor, 0)

	for i := range m.Modules {
		miweights := m.Modules[i].GetWeights()
		if len(miweights) == 0 {
			panic(len(miweights))
		}

		weights = append(weights, miweights...)

	}
	weights = append(weights, m.Output.GetWeights()...)
	return weights

}
func (m *SimpleModuleNetwork) GetDeltaWeights() []*Tensor {
	dweights := make([]*Tensor, 0)

	for i := range m.Modules {
		dweights = append(dweights, m.Modules[i].GetDeltaWeights()...)

	}
	dweights = append(dweights, m.Output.GetDeltaWeights()...)
	return dweights
}

//SetMSEClassifier needs to be made
func (m *SimpleModuleNetwork) SetMSEClassifier() (err error) {
	return errors.New("(m *SimpleModuleNetwork) SetMSEClassifier() needs to be made")
}

//SetSoftMaxClassifier sets the classifier module it should be added last.
//Should be ran after OutputModule is set
func (m *SimpleModuleNetwork) SetSoftMaxClassifier() (err error) { //(y, dy *Tensor, err error) {

	lastmod := m.Output
	if lastmod.GetTensorDX() == nil {
		lastmod.SetTensorDX(m.Modules[len(m.Modules)-1].GetTensorDY())
	}
	if lastmod.GetTensorX() == nil {
		lastmod.SetTensorX(m.Modules[len(m.Modules)-1].GetTensorY())
	}
	if lastmod.GetTensorDY() == nil {
		lmoutputdims, err := lastmod.FindOutputDims()
		if err != nil {
			return err
		}
		lmdy, err := m.b.CreateTensor(lmoutputdims)
		if err != nil {
			return err
		}
		lastmod.SetTensorDY(lmdy)
	}
	if lastmod.GetTensorY() == nil {
		lmoutputdims, err := lastmod.FindOutputDims()
		if err != nil {
			return err
		}
		lmy, err := m.b.CreateTensor(lmoutputdims)
		if err != nil {
			return err
		}
		lastmod.SetTensorY(lmy)
	}
	lmoutputdims, err := lastmod.FindOutputDims()
	if err != nil {
		return err
	}
	y, err := m.b.CreateTensor(lmoutputdims)
	if err != nil {
		return err
	}
	dy, err := m.b.CreateTensor(lmoutputdims)
	if err != nil {
		return err
	}

	m.Classifier, err = CreateSoftMaxClassifier(lastmod.ID()+1, m.b, lastmod.GetTensorY(), lastmod.GetTensorDY(), y, dy)
	if err != nil {
		return err
	}
	return nil
}

//SetModules sets modules
func (m *SimpleModuleNetwork) SetModules(modules []Module) {
	m.Modules = modules
}

//ID satisfies Module interface
func (m *SimpleModuleNetwork) ID() int64 {
	return m.id
}

//GetTensorX Gets x tensor
func (m *SimpleModuleNetwork) GetTensorX() *Tensor {
	if m.Modules[0] != nil {
		return m.Modules[0].GetTensorX()
	}
	return nil
}

//GetTensorDX Gets dx tensor
func (m *SimpleModuleNetwork) GetTensorDX() *Tensor {
	if m.Modules[0] != nil {
		return m.Modules[0].GetTensorDX()

	}
	return nil
}

//GetTensorY Gets y tensor
func (m *SimpleModuleNetwork) GetTensorY() *Tensor {
	if m.Classifier != nil {
		return m.Classifier.GetTensorY()
		//	return m.y
	} else if m.Output != nil {
		return m.Output.GetTensorY()
	}
	if m.Modules != nil {
		return m.Modules[len(m.Modules)-1].GetTensorY()
	}

	return nil
}

//GetTensorDY Gets dy tensor
func (m *SimpleModuleNetwork) GetTensorDY() *Tensor {
	if m.Classifier != nil {
		return m.Classifier.GetTensorDY()
		//	return m.dy
	} else if m.Output != nil {
		return m.Output.GetTensorDY()
	} else if m.Modules != nil {
		return m.Modules[len(m.Modules)-1].GetTensorDY()
	}
	return nil
	//return m.dy
}

//SetTensorX sets x tensor
func (m *SimpleModuleNetwork) SetTensorX(x *Tensor) {
	//	m.x = x
	if m.Modules != nil {
		m.Modules[0].SetTensorX(x)
	}

}

//SetTensorDX sets dx tensor
func (m *SimpleModuleNetwork) SetTensorDX(dx *Tensor) {
	//	m.dx = dx
	if m.Modules != nil {
		m.Modules[0].SetTensorDX(dx)
	}
}

//SetTensorY sets y tensor
func (m *SimpleModuleNetwork) SetTensorY(y *Tensor) {
	//	m.y = y
	if m.Classifier != nil {
		m.Classifier.SetTensorY(y)
	} else if m.Output != nil {
		m.Output.SetTensorY(y)
	} else if len(m.Modules) > 0 {
		m.Modules[len(m.Modules)-1].SetTensorY(y)
	}

}

//SetTensorDY sets dy tensor
func (m *SimpleModuleNetwork) SetTensorDY(dy *Tensor) {
	//	m.dy = dy
	if m.Classifier != nil {
		m.Classifier.SetTensorDY(dy)
	} else if m.Output != nil {
		m.Output.SetTensorDY(dy)
	} else if len(m.Modules) > 0 {
		m.Modules[len(m.Modules)-1].SetTensorDY(dy)
	}

}

//InitHiddenLayers satisfies the Module interface
func (m *SimpleModuleNetwork) InitHiddenLayers() (err error) {

	if m.Modules == nil {
		return fmt.Errorf("(m *SimpleModuleNetwork) InitHiddenLayers: %s", "Modules are nil")
	}
	//if m.x == nil {
	//	return fmt.Errorf("(m *SimpleModuleNetwork) InitHiddenLayers: %s", "TensorX is nil")
	//}

	if m.Modules[0].GetTensorY() == nil {
		_, err = m.FindOutputDims() //m.FindOutputDims creates connections between Modules
		if err != nil {
			return fmt.Errorf("(m *SimpleModuleNetwork) InitHiddenLayers: %v", err)
		}

	}
	for i, mod := range m.Modules {
		err = mod.InitHiddenLayers()
		if err != nil {
			return fmt.Errorf("(m *SimpleModuleNetwork) InitHiddenLayers: index %v\n %v", i, err)
		}
	}

	err = m.Output.InitHiddenLayers()
	if err != nil {
		return fmt.Errorf("(m *SimpleModuleNetwork) InitHiddenLayers: m.Output: %v", err)
	}
	//m.firstinithidden = true
	return nil
}

//InitWorkspace inits workspace
func (m *SimpleModuleNetwork) InitWorkspace() (err error) {
	for i, mod := range m.Modules {
		err = mod.InitWorkspace()
		if err != nil {
			return fmt.Errorf("(m *SimpleModuleNetwork) InitWorkspace: index: %v\n err: %v", i, err)
		}
	}
	err = m.Output.InitWorkspace()
	if err != nil {
		return fmt.Errorf("(m *SimpleModuleNetwork) InitWorkspace: m.Output: %v", err)
	}
	//	m.firstinitworkspace = true
	return nil
}

//FindOutputDims satisfies the Module interface
//
//Have to run (m *SimpleModuleNetwork)SetTensorX().  If module network requires backpropdata to go to another module network.
//Then also run (m *SimpleModuleNetwork)SetTensorDX()
func (m *SimpleModuleNetwork) FindOutputDims() (dims []int32, err error) {
	//	if m.x == nil {
	//		return nil, errors.New("(m *SimpleModuleNetwork) FindOutputDims: TensorX hasn't been set")
	//	}
	if m.Modules == nil {
		return nil, errors.New("(m *SimpleModuleNetwork) FindOutputDims: No Modules have been set")
	}
	//if m.Output != nil {
	//	return m.Output.FindOutputDims()
	//}
	var px = m.GetTensorX()
	if px == nil {
		return nil, errors.New("(m *SimpleModuleNetwork) FindOutputDims: First Module's input not set")
	}
	var pdx = m.GetTensorDX()

	var outputdims []int32
	for i, mod := range m.Modules {
		if mod.GetTensorX() == nil {
			mod.SetTensorX(px)
		} else {
			if mod.GetTensorX() != px {
				panic("SHould be the same")
			}
		}
		if mod.GetTensorDX() == nil {
			mod.SetTensorDX(pdx)
		} else {
			if mod.GetTensorDX() != pdx {
				panic("SHould be the same")
			}
		}
		poutputdims := outputdims
		outputdims, err = mod.OutputDims()

		if err != nil {
			fmt.Println("previous outputdims", poutputdims)
			fmt.Println("Outputdim wrong at index:", i)
			return nil, err
		}
		if mod.GetTensorY() == nil {
			px, err = m.b.CreateTensor(outputdims)
			if err != nil {
				return nil, err
			}
			mod.SetTensorY(px)
		} else {
			px = mod.GetTensorY()
		}
		if mod.GetTensorDY() == nil {
			pdx, err = m.b.CreateTensor(outputdims)
			if err != nil {
				return nil, err
			}
			mod.SetTensorDY(pdx)
		} else {
			pdx = mod.GetTensorDY()
		}

	}

	outputdims, err = m.Modules[len(m.Modules)-1].OutputDims()
	if m.Output == nil {
		return outputdims, err
	}
	if m.Output.GetTensorX() == nil {
		m.Output.SetTensorX(px)
	} else {
		if m.Output.GetTensorX() != px {
			panic("SHould be the same")
		}
	}
	if m.Output.GetTensorDX() == nil {
		m.Output.SetTensorDX(pdx)
	} else {
		if m.Output.GetTensorDX() != pdx {
			panic("SHould be the same")
		}
	}
	if m.Output.GetTensorY() == nil {
		px, err = m.b.CreateTensor(outputdims)
		if err != nil {
			return nil, err
		}
		m.Output.SetTensorY(px)
	} else {
		px = m.Output.GetTensorY()
	}
	if m.Output.GetTensorDY() == nil {
		pdx, err = m.b.CreateTensor(outputdims)
		if err != nil {
			return nil, err
		}
		m.Output.SetTensorDY(pdx)
	} else {
		pdx = m.Output.GetTensorDY()
	}
	outputdims, err = m.Output.FindOutputDims()
	if m.Classifier == nil {
		return outputdims, nil
	}
	if m.Classifier.GetTensorX() == nil {
		m.Classifier.SetTensorX(px)
	} else {
		if px != m.Classifier.GetTensorX() {
			panic("Should be the same")
		}
	}
	if m.Classifier.GetTensorDX() == nil {
		m.Classifier.SetTensorDX(pdx)
	} else {
		if pdx != m.Classifier.GetTensorDX() {
			panic("Should be the same")
		}
	}

	return outputdims, nil
}

//Forward performs the forward operation
//of the simple module network.
//Forward satisfies the Module Interface.
func (m *SimpleModuleNetwork) Forward() (err error) {
	for i, mod := range m.Modules {

		err = mod.Forward()
		if err != nil {
			fmt.Println("Forward: Error in hidden mod,", i)
			return err
		}
	}
	if m.Output != nil {
		err = m.Output.Forward()
		if err != nil {
			fmt.Println("Forward: Error in Output mod")
			return err
		}
	}
	if m.Classifier != nil {
		err := m.Classifier.PerformError()
		if err != nil {
			fmt.Println("Forward: Error in Classifier mod")
			return err
		}

	}
	return nil
}

//BackPropForSharedInputForModuleNetworks is a hack to make up if two module networks share the same input.
//It will zero out the dx values for the module and then run back propagation
func BackPropForSharedInputForModuleNetworks(m []*SimpleModuleNetwork) (err error) {
	err = m[0].GetTensorDX().SetValues(m[0].b.h.Handler, 0)
	if err != nil {
		return err
	}
	for i := range m {
		err = m[i].Backward()
		if err != nil {
			return err
		}

	}
	return nil
}

//GetLoss returns the loss found.
func (m *SimpleModuleNetwork) GetLoss() float32 {
	return m.Classifier.GetAverageBatchLoss()
}

//Backward does a forward without a concat
func (m *SimpleModuleNetwork) Backward() (err error) {

	err = m.Output.Backward()
	if err != nil {
		return err
	}
	for i := len(m.Modules) - 1; i >= 0; i-- {

		err = m.Modules[i].Backward()
		if err != nil {
			return err
		}
	}

	return nil
}

//Inference does a forward without a concat
func (m *SimpleModuleNetwork) Inference() (err error) {
	for i := range m.Modules {
		err = m.Modules[i].Inference()
		if err != nil {
			return err
		}
	}
	if m.Output != nil {
		m.Output.Inference()
	}
	if m.Classifier == nil {
		return nil
	}
	return m.Classifier.Inference()
}

//TestForward does the forward prop but it still calculates loss for testing
func (m *SimpleModuleNetwork) TestForward() (err error) {
	for i := range m.Modules {
		err = m.Modules[i].Inference()
		if err != nil {
			return err
		}
	}
	if m.Output != nil {
		return m.Output.Inference()
	}
	if m.Classifier != nil {
		return m.Classifier.TestForward()
	}
	return nil
}

//ForwardCustom does a custom forward function
func (m *SimpleModuleNetwork) ForwardCustom(forward func() error) (err error) {
	return forward()
}

//BackwardCustom does a custom backward function
func (m *SimpleModuleNetwork) BackwardCustom(backward func() error) (err error) {
	return backward()
}

//Forwarder does the forward operation
type Forwarder interface {
	Forward() error
}

//Backwarder does the backward operation
type Backwarder interface {
	Backward() error
}

//Updater does the update interface
type Updater interface {
	Update() error
}
