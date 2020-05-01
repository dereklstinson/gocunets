package gocunets

//import (
//	//"github.com/dereklstinson/gocunets/loss"
//	"github.com/dereklstinson/gocunets/trainer"
//	"github.com/dereklstinson/pso"
//)

/*
func (m *Network) totalnumofscalarsalpha() int {
	adder := 0
	for i := range m.layers {
		adder += m.layers[i].scalarnumalpha
	}
	return adder
}
*/

////GetTrainers returns the trainers for the network.  ...convienence function
//func (m *Network) GetTrainers() (trainers []trainer.Trainer) {
//	return m.trainers
//
//	//	return m.wtrainers, m.btrainers
//}

/*
//ScalarOptimizer optimizes the scalars of the operators
type ScalarOptimizer struct {
	hasscalars     []*Layer
	mse            *loss.MSE
	pso            *pso.Swarm64
	index          int
	numofparticles int
	alpha          bool
	alphaglobal    bool
	betaglobal     bool
}

//SetupScalarAlphaPSO returns a pso to optimize the alpha scalars in the network
func SetupScalarAlphaPSO(mode pso.Mode, numofparticles, seed int, cognative, social, vmax, minstartposition, maxstartposition, alphamax, inertiamax float64, mse *loss.MSE, x ...*Network) ScalarOptimizer {
	hasscalars := make([]*Layer, 0)
	totalscalars := 0
	for i := range x {
		for _, layer := range x[i].layers {
			amount := layer.initalphascalarsamount()

			if amount != 0 {
				hasscalars = append(hasscalars, layer)
				totalscalars += amount
			}

		}
	}
	totalscalars++ //mse has only one alpha
	swarm := pso.CreateSwarm64(seed)
	swarm.GenericSet(mode, numofparticles, totalscalars, cognative, social, vmax, minstartposition, maxstartposition, alphamax, inertiamax)

	position := swarm.ParticlePosition(0)

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
func SetupScalarBetaPSO(mode pso.Mode, numofparticles, seed int, cognative, social, vmax, minstartposition, maxstartposition, alphamax, inertiamax float64, mse *loss.MSE, x ...*Network) ScalarOptimizer {
	hasscalars := make([]*Layer, 0)
	totalscalars := 0
	for i := range x {
		for _, layer := range x[i].layers {
			amount := layer.initbetascalarsamount()

			if amount != 0 {
				hasscalars = append(hasscalars, layer)
				totalscalars += amount
			}

		}
	}
	totalscalars++ //mse has only one beta
	swarm := pso.CreateSwarm64(seed)
	swarm.GenericSet(mode, numofparticles, totalscalars, cognative, social, vmax, minstartposition, maxstartposition, alphamax, inertiamax)
	position := swarm.ParticlePosition(0)

	for i := range hasscalars {

		position = hasscalars[i].updateabetascalar(position)
	}
	return ScalarOptimizer{
		hasscalars: hasscalars,
		pso:        swarm,
		mse:        mse,
	}
}

//Reset resets the Optimizer and resets a percent (between 0 and 1..1 being 100%) of the partilces
func (m *ScalarOptimizer) Reset(indexes []pso.FitnessIndex64, resetglobalposition bool) error {
	err := m.pso.ResetParticles(indexes, resetglobalposition)
	if err != nil {
		return err
	}

	return nil
}

//AllFitnesses gets all the fitnesses
func (m *ScalarOptimizer) AllFitnesses(previousfitnesses []pso.FitnessIndex64) []pso.FitnessIndex64 {
	return m.pso.AllFitnesses(previousfitnesses)
}

//SetGlobal sets the global best.
func (m *ScalarOptimizer) SetGlobal() error {
	if m.alpha {

		return m.alphaglobalset()

	}
	return m.betaglobalset()
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

	position := m.pso.ParticlePosition(m.index)
	for i := range m.hasscalars {
		position = m.hasscalars[i].updateabetascalar(position)

	}
	if len(position) != 1 {
		panic("position should be one here")
	}
	m.mse.SetBetaScalars(position)
	m.betaglobal = false
	return nil
}

//AsyncUpdating updates the Swarm after each particle use
func (m *ScalarOptimizer) betaglobalset() error {
	if !m.betaglobal {
		position := m.pso.GlobalPosition()

		for i := range m.hasscalars {
			position = m.hasscalars[i].updateabetascalar(position)

		}
		if len(position) != 1 {
			panic("position should be one here")
		}
		m.mse.SetBetaScalars(position)
		m.betaglobal = true
	}
	return nil
}

//AsyncUpdating updates the Swarm after each particle use
func (m *ScalarOptimizer) alphaglobalset() error {
	if !m.alphaglobal {
		position := m.pso.GlobalPosition()
		for i := range m.hasscalars {
			position = m.hasscalars[i].updatealphascalar(position)

		}
		if len(position) != 1 {
			panic("position should be one here")
		}
		m.mse.SetAlphaScalars(position)

		m.alphaglobal = true
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

	position := m.pso.ParticlePosition(m.index)
	for i := range m.hasscalars {
		position = m.hasscalars[i].updatealphascalar(position)

	}
	if len(position) != 1 {
		panic("position should be one here")
	}
	m.mse.SetAlphaScalars(position)
	m.alphaglobal = false
	return nil
}
*/

////MetaOptimizer uses a PSO to optimize meta values
//type MetaOptimizer struct {
//	trainers       []trainer.Trainer
//	pso            *pso.Swarm32
//	index          int
//	numofparticles int
//	global         bool
//}
//
////AsyncUpdating updates the Swarm after each particle use
//func (m *MetaOptimizer) AsyncUpdating(fitness float32) error {
//	err := m.pso.AsyncUpdate(m.index, fitness)
//	if err != nil {
//		return err
//	}
//	if m.index < m.numofparticles-1 {
//		m.index++
//	} else {
//		m.index = 0
//	}
//
//	pctr := 0
//	position := m.pso.ParticlePosition(m.index)
//	for i := range m.trainers {
//		m.trainers[i].SetRates(position[pctr], position[pctr+1])
//		m.trainers[i].SetDecays(position[pctr+2], position[pctr+3])
//		pctr += 4
//	}
//	m.global = false
//	return nil
//}
//
////SetGlobal updates the Swarm after each particle use
//func (m *MetaOptimizer) SetGlobal() error {
//	if !m.global {
//		pctr := 0
//		position := m.pso.GlobalPosition()
//		for i := range m.trainers {
//			m.trainers[i].SetRates(position[pctr], position[pctr+1])
//			m.trainers[i].SetDecays(position[pctr+2], position[pctr+3])
//			pctr += 4
//		}
//		m.global = true
//	}
//
//	return nil
//}
//
////Reset resets the Optimizer and resets a percent (between 0 and 1..1 being 100%) of the partilces
//func (m *MetaOptimizer) Reset(indexes []pso.FitnessIndex32, resetglobalposition bool) error {
//	err := m.pso.ResetParticles(indexes, resetglobalposition)
//	if err != nil {
//		return err
//	}
//
//	return nil
//}
//
////AllFitnesses gets all the fitnesses
//func (m *MetaOptimizer) AllFitnesses(previousfitnesses []pso.FitnessIndex32) []pso.FitnessIndex32 {
//	return m.pso.AllFitnesses(previousfitnesses)
//}
//
////SetUpPSO will set up the pso
//func SetUpPSO(mode pso.Mode, numofparticles, seed int, cognative, social, vmax, minstartposition, maxstartposition, alphamax, inertiamax float32, x ...[]trainer.Trainer) MetaOptimizer {
//
//	trainers := make([]trainer.Trainer, 0)
//	for i := range x {
//		trainers = append(trainers, x[i]...)
//	}
//	totaldims := len(trainers) * 4
//	swarm := pso.CreateSwarm32(seed)
//	swarm.GenericSet(mode, numofparticles, totaldims, cognative, social, vmax, minstartposition, maxstartposition, alphamax, inertiamax)
//	position := swarm.ParticlePosition(0)
//	pctr := 0
//	for i := range trainers {
//		trainers[i].SetRates(position[pctr], position[pctr+1])
//		trainers[i].SetDecays(position[pctr+2], position[pctr+3])
//		pctr += 4
//	}
//	return MetaOptimizer{
//		trainers:       trainers,
//		pso:            swarm,
//		numofparticles: numofparticles,
//	}
//}
//
