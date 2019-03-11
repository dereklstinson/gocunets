package gocunets

import (
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
func SetupScalarAlphaPSO(mode pso.Mode, numofparticles, seed, kmax int, cognative, social, vmax, minstartposition, maxstartposition, alphamax, inertiamax float64, x ...*Network) ScalarOptimizer {
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
	swarm := pso.CreateSwarm64(mode, numofparticles, totalscalars, seed, kmax, cognative, social, vmax, minstartposition, maxstartposition, alphamax, inertiamax)
	position := swarm.GetParticlePosition(0)

	for i := range hasscalars {

		position = hasscalars[i].updatealphascalar(position)
	}
	return ScalarOptimizer{
		hasscalars: hasscalars,
		pso:        swarm,
		alpha:      true,
	}
}

//SetupScalarBetaPSO returns a pso to optimize the beta scalars in the network
func SetupScalarBetaPSO(mode pso.Mode, numofparticles, seed, kmax int, cognative, social, vmax, minstartposition, maxstartposition, alphamax, inertiamax float64, x ...*Network) ScalarOptimizer {
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
	swarm := pso.CreateSwarm64(mode, numofparticles, totalscalars, seed, kmax, cognative, social, vmax, minstartposition, maxstartposition, alphamax, inertiamax)
	position := swarm.GetParticlePosition(0)

	for i := range hasscalars {

		position = hasscalars[i].updateabetascalar(position)
	}
	return ScalarOptimizer{
		hasscalars: hasscalars,
		pso:        swarm,
	}
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
		m.trainers[i].SetRate(position[pctr])
		m.trainers[i].SetDecays(position[pctr+1], position[pctr+2])
		pctr = pctr + 3
	}
	return nil
}

//SetUpPSO will set up the pso
func SetUpPSO(mode pso.Mode, numofparticles, seed, kmax int, cognative, social, vmax, minstartposition, maxstartposition, alphamax, inertiamax float32, x ...[]trainer.Trainer) MetaOptimizer {

	trainers := make([]trainer.Trainer, 0)
	for i := range x {
		trainers = append(trainers, x[i]...)
	}
	totaldims := len(trainers) * 3
	swarm := pso.CreateSwarm(mode, numofparticles, totaldims, seed, kmax, cognative, social, vmax, minstartposition, maxstartposition, alphamax, inertiamax)
	position := swarm.GetParticlePosition(0)
	pctr := 0
	for i := range trainers {
		trainers[i].SetRate(position[pctr])
		trainers[i].SetDecays(position[pctr+1], position[pctr+2])
		pctr = pctr + 3
	}
	return MetaOptimizer{
		trainers:       trainers,
		pso:            swarm,
		numofparticles: numofparticles,
	}
}
