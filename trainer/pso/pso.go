/*

pso is based on the slides from Jaco F. Schutte EGM 6365 - Structural Optimization Fall 2005
Link to slides https://www.mii.lt/zilinskas/uploads/Heuristic%20Algorithms/Lectures/Lect4/PSO2.pdf

TODO: ResetSwarm should check the opposite condition. So that I can minimize the number of calculations.
*/

package pso

import (
	ps "github.com/dereklstinson/pso"
)

//Mode is the mode flag for the swarm
type Mode int32

//Vanilla sets vanilla mode
func (m *Mode) Vanilla() Mode {
	*m = Mode(1)
	return *m
}

//ConstantInertia sets ConstantInertia mode
func (m *Mode) ConstantInertia() Mode {
	*m = Mode(2)
	return *m
}

//InertiaReduction sets InertiaReduction mode
func (m *Mode) InertiaReduction() Mode {
	*m = Mode(3)
	return *m
}

//Constriction sets Constriction mode
func (m *Mode) Constriction() Mode {
	*m = Mode(4)
	return *m
}

//DynamicInertiaMaxVelReduction sets DynamicInertiaMaxVelReduction mode
func (m *Mode) DynamicInertiaMaxVelReduction() Mode {
	*m = Mode(5)
	return *m
}

/*
func (m ModeFlag) SocialPressure() Mode {
	return Mode(6)
}
*/

//Swarm32 is a Swarm32
type Swarm32 struct {
	*ps.Swarm32
}

//Swarm32 is a Swarm32
type Swarm64 struct {
	*ps.Swarm64
}

//FitnessIndex32 is used when getting the fitnes of a particle
type FitnessIndex32 struct {
	ps.FitnessIndex32
}

//FitnessIndex32 is used when getting the fitnes of a particle
type FitnessIndex64 struct {
	ps.FitnessIndex64
}

//CreateSwarm32 creates a particle swarm with float32 precision
func CreateSwarm32(mode Mode, numofparticles, dims, seed, kmax int, cognative, social, vmax, xminstart, xmaxstart, alphamax, inertiamax float32) *Swarm32 {
	swarm := (ps.CreateSwarm32(seed))

	var m Mode
	switch mode {
	case m.Vanilla():
		swarm.SetVanilla(numofparticles, dims, cognative, social, vmax, xminstart, xmaxstart)

	case m.InertiaReduction():
		swarm.SetLinearInertiaReduce(numofparticles, dims, cognative, social, vmax, xmaxstart, xmaxstart, alphamax, inertiamax)
	case m.DynamicInertiaMaxVelReduction():
		swarm.SetDynamicInertiaMaxVelocityReduction(numofparticles, dims, cognative, social, vmax, xmaxstart, xmaxstart, inertiamax)
	case m.Constriction():
		swarm.SetConstriction(numofparticles, dims, cognative, social, vmax, xminstart, xmaxstart)
	case m.ConstantInertia():
		swarm.SetConstantInertia(numofparticles, dims, cognative, social, vmax, xminstart, xmaxstart, inertiamax)

	}
	return &Swarm32{(swarm)}
}

//CreateSwarm64 creates a swarm using float64 precision
func CreateSwarm64(mode Mode, numofparticles, dims, seed, kmax int, cognative, social, vmax, xminstart, xmaxstart, alphamax, inertiamax float64) *Swarm64 {
	swarm := ps.CreateSwarm64(seed)
	var m Mode
	switch mode {
	case m.Vanilla():
		swarm.SetVanilla(numofparticles, dims, cognative, social, vmax, xminstart, xmaxstart)

	case m.InertiaReduction():
		swarm.SetLinearInertiaReduce(numofparticles, dims, cognative, social, vmax, xmaxstart, xmaxstart, alphamax, inertiamax)
	case m.DynamicInertiaMaxVelReduction():
		swarm.SetDynamicInertiaMaxVelocityReduction(numofparticles, dims, cognative, social, vmax, xmaxstart, xmaxstart, inertiamax)
	case m.Constriction():
		swarm.SetConstriction(numofparticles, dims, cognative, social, vmax, xminstart, xmaxstart)
	case m.ConstantInertia():
		swarm.SetConstantInertia(numofparticles, dims, cognative, social, vmax, xminstart, xmaxstart, inertiamax)

	}
	return &Swarm64{swarm}
}
