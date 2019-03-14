package gocunets

import (
	"github.com/dereklstinson/GoCuNets/loss"
	"github.com/dereklstinson/GoCuNets/trainer"
	"github.com/dereklstinson/GoCuNets/trainer/pso"
)

func (m *Network) totalnumofscalarsalpha() int {
	adder := 0
	for i := range m.layer {
		adder += m.layer[i].scalarnumalpha
	}
	return adder
}

//GetTrainers returns the trainers for the network.  ...convienence function
func (m *Network) GetTrainers() (weights, bias []trainer.Trainer) {
	return m.wtrainers, m.btrainers
}

//ScalarOptimizer optimizes the scalars of the operators
type ScalarOptimizer struct {
	hasscalars     []*layer
	mse            *loss.MSE
	pso            pso.Swarm64
	index          int
	numofparticles int
	alpha          bool
}

func (m *Network) initializeslphascalarstuff() ([]*layer, int) {
	adder := 0
	layers := make([]*layer, 0)
	for i := range m.layer {
		x := m.layer[i].initalphascalarsamount()
		if x > 0 {
			layers = append(layers, m.layer[i])
		}
		adder += x
	}
	return layers, adder
}
func (m *Network) initializebetascalarstuff() ([]*layer, int) {
	adder := 0
	layers := make([]*layer, 0)
	for i := range m.layer {
		x := m.layer[i].initbetascalarsamount()
		if x > 0 {
			layers = append(layers, m.layer[i])
		}
		adder += x
	}
	return layers, adder
}

//SetupScalarAlphaPSO returns a pso to optimize the alpha scalars in the network
func SetupScalarAlphaPSO(mode pso.Mode, numofparticles, seed, kmax int, cognative, social, vmax, minstartposition, maxstartposition, alphamax, inertiamax float64, mse *loss.MSE, x ...*Network) ScalarOptimizer {
	hasscalars := make([]*layer, 0)
	totalscalars := 0
	for i := range x {
		for _, layer := range x[i].layer {
			amount := layer.initalphascalarsamount()

			if amount != 0 {
				hasscalars = append(hasscalars, layer)
				totalscalars += amount
			}

		}
	}
	totalscalars++ //mse has only one alpha
	swarm := pso.CreateSwarm64(mode, numofparticles, totalscalars, seed, kmax, cognative, social, vmax, minstartposition, maxstartposition, alphamax, inertiamax)
	position := swarm.GetParticlePosition(0)

	for i := range hasscalars {

		position = hasscalars[i].updatealphascalar(position)
	}
	return ScalarOptimizer{
		hasscalars: hasscalars,
		pso:        swarm,
		alpha:      true,
		mse:        mse,
	}
}

//SetupScalarBetaPSO returns a pso to optimize the beta scalars in the network
func SetupScalarBetaPSO(mode pso.Mode, numofparticles, seed, kmax int, cognative, social, vmax, minstartposition, maxstartposition, alphamax, inertiamax float64, mse *loss.MSE, x ...*Network) ScalarOptimizer {
	hasscalars := make([]*layer, 0)
	totalscalars := 0
	for i := range x {
		for _, layer := range x[i].layer {
			amount := layer.initbetascalarsamount()

			if amount != 0 {
				hasscalars = append(hasscalars, layer)
				totalscalars += amount
			}

		}
	}
	totalscalars++ //mse has only one beta
	swarm := pso.CreateSwarm64(mode, numofparticles, totalscalars, seed, kmax, cognative, social, vmax, minstartposition, maxstartposition, alphamax, inertiamax)
	position := swarm.GetParticlePosition(0)

	for i := range hasscalars {

		position = hasscalars[i].updateabetascalar(position)
	}
	return ScalarOptimizer{
		hasscalars: hasscalars,
		pso:        swarm,
		mse:        mse,
	}
}

//ReachedKmax will let the outside world know if max num of k was reached
func (m *ScalarOptimizer) ReachedKmax() bool {
	return m.pso.ReachedKmax()
}

//ResetInnerKCounter resets the inner k counter
func (m *ScalarOptimizer) ResetInnerKCounter() {
	m.pso.ResetInnerKCounter()
}

//Reset resets the Optimizer and resets a percent (between 0 and 1..1 being 100%) of the partilces
func (m *ScalarOptimizer) Reset(percent float32) error {
	err := m.pso.ResetSwarm(percent)
	if err != nil {
		return err
	}
	m.pso.ResetInnerKCounter()
	return nil
}

//AsyncUpdating does an asyncronus update of the scalar parameters
func (m *ScalarOptimizer) AsyncUpdating(fitness float32) error {
	if m.alpha {
		return m.asyncupdatealpha(fitness)
	}
	return m.asyncupdatebeta(fitness)
}

//AsyncUpdating updates the Swarm after each particle use
func (m *ScalarOptimizer) asyncupdatebeta(fitness float32) error {

	err := m.pso.AsyncUpdate(m.index, float64(fitness))
	if err != nil {
		return err
	}
	if m.index < m.numofparticles-1 {
		m.index++
	} else {
		m.index = 0
	}

	position := m.pso.GetParticlePosition(m.index)
	for i := range m.hasscalars {
		position = m.hasscalars[i].updateabetascalar(position)

	}
	if len(position) != 1 {
		panic("position should be one here")
	}
	m.mse.SetBetaScalars(position)
	return nil
}

//AsyncUpdating updates the Swarm after each particle use
func (m *ScalarOptimizer) asyncupdatealpha(fitness float32) error {

	err := m.pso.AsyncUpdate(m.index, float64(fitness))
	if err != nil {
		return err
	}
	if m.index < m.numofparticles-1 {
		m.index++
	} else {
		m.index = 0
	}

	position := m.pso.GetParticlePosition(m.index)
	for i := range m.hasscalars {
		position = m.hasscalars[i].updatealphascalar(position)

	}
	if len(position) != 1 {
		panic("position should be one here")
	}
	m.mse.SetAlphaScalars(position)
	return nil
}

//MetaOptimizer uses a PSO to optimize meta values
type MetaOptimizer struct {
	trainers       []trainer.Trainer
	pso            pso.Swarm
	index          int
	numofparticles int
}

//AsyncUpdating updates the Swarm after each particle use
func (m *MetaOptimizer) AsyncUpdating(fitness float32) error {
	err := m.pso.AsyncUpdate(m.index, fitness)
	if err != nil {
		return err
	}
	if m.index < m.numofparticles-1 {
		m.index++
	} else {
		m.index = 0
	}

	pctr := 0
	position := m.pso.GetParticlePosition(m.index)
	for i := range m.trainers {
		m.trainers[i].SetRates(position[pctr], position[pctr+1])
		m.trainers[i].SetDecays(position[pctr+2], position[pctr+3])
		pctr += 4
	}
	return nil
}

//ReachedKmax will let the outside world know if max num of k was reached
func (m *MetaOptimizer) ReachedKmax() bool {
	return m.pso.ReachedKmax()
}

//ResetInnerKCounter resets the inner k counter
func (m *MetaOptimizer) ResetInnerKCounter() {
	m.pso.ResetInnerKCounter()
}

//Reset resets the Optimizer and resets a percent (between 0 and 1..1 being 100%) of the partilces
func (m *MetaOptimizer) Reset(percent float32) error {
	err := m.pso.ResetSwarm(percent)
	if err != nil {
		return err
	}
	m.pso.ResetInnerKCounter()
	return nil
}

//SetUpPSO will set up the pso
func SetUpPSO(mode pso.Mode, numofparticles, seed, kmax int, cognative, social, vmax, minstartposition, maxstartposition, alphamax, inertiamax float32, x ...[]trainer.Trainer) MetaOptimizer {

	trainers := make([]trainer.Trainer, 0)
	for i := range x {
		trainers = append(trainers, x[i]...)
	}
	totaldims := len(trainers) * 4
	swarm := pso.CreateSwarm(mode, numofparticles, totaldims, seed, kmax, cognative, social, vmax, minstartposition, maxstartposition, alphamax, inertiamax)
	position := swarm.GetParticlePosition(0)
	pctr := 0
	for i := range trainers {
		trainers[i].SetRates(position[pctr], position[pctr+1])
		trainers[i].SetDecays(position[pctr+2], position[pctr+3])
		pctr += 4
	}
	return MetaOptimizer{
		trainers:       trainers,
		pso:            swarm,
		numofparticles: numofparticles,
	}
}
