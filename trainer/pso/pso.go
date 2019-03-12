/*

pso is based on the slides from Jaco F. Schutte EGM 6365 - Structural Optimization Fall 2005
Link to slides https://www.mii.lt/zilinskas/uploads/Heuristic%20Algorithms/Lectures/Lect4/PSO2.pdf

*/

package pso

import (
	"errors"
	"math"
	"math/rand"
)

type particle struct {
	rng      *rand.Rand
	mflg     ModeFlag
	source   rand.Source
	loss     float32
	position []float32
	indvbest []float32
	velocity []float32
	inertia  float32
	alpha    float32
	vmax     float32
}

//Swarm contains the particles and meta values
type Swarm struct {
	k, kmax                               int
	loss                                  float32
	cognative, social, vmax, constriction float32
	particles                             []particle
	globalposition                        []float32
	bestbool                              []bool
	mode                                  Mode
}

const testingvmax = true

//Mode is the mode flag for the swarm
type Mode int32

//ModeFlag returns modes
type ModeFlag struct {
}

//Vanilla sets vanilla mode
func (m ModeFlag) Vanilla() Mode {
	return Mode(1)
}

//ConstantInertia sets ConstantInertia mode
func (m ModeFlag) ConstantInertia() Mode {
	return Mode(2)
}

//InertiaReduction sets InertiaReduction mode
func (m ModeFlag) InertiaReduction() Mode {
	return Mode(3)
}

//Constriction sets Constriction mode
func (m ModeFlag) Constriction() Mode {
	return Mode(4)
}

//DynamicInertiaMaxVelReduction sets DynamicInertiaMaxVelReduction mode
func (m ModeFlag) DynamicInertiaMaxVelReduction() Mode {
	return Mode(5)
}

/*
func (m ModeFlag) SocialPressure() Mode {
	return Mode(6)
}
*/

//CreateSwarm creates a particle swarm
func CreateSwarm(mode Mode, numofparticles, dims, seed, kmax int, cognative, social, vmax, xminstart, xmaxstart, alphamax, inertiamax float32) Swarm {
	rand.Seed(int64(seed))

	particles := make([]particle, numofparticles)
	for i := range particles {
		particles[i] = createparticle(vmax, xminstart, xmaxstart, alphamax, inertiamax, dims, rand.Int63())

	}
	gamma := float64(social + cognative)
	constriction := 2 / (2 - gamma - math.Sqrt((gamma*gamma)-4*gamma))
	return Swarm{
		k:            1,
		cognative:    cognative,
		social:       social,
		kmax:         kmax,
		vmax:         vmax,
		loss:         9999999999,
		particles:    particles,
		constriction: float32(constriction),
	}
}

func createparticle(maxv, minxstart, maxxstart, maxalpha, maxinertia float32, dims int, seed int64) particle {
	source := rand.NewSource(seed)
	rng := rand.New(source)
	position := make([]float32, dims)
	indvbest := make([]float32, dims)
	velocity := make([]float32, dims)
	var val float32
	for i := range position {
		val = ((maxxstart - minxstart) * rng.Float32()) + minxstart
		position[i] = val
		indvbest[i] = val
		velocity[i] = rng.Float32() * maxv
	}
	return particle{
		rng:      rng,
		source:   source,
		loss:     999999999999,
		position: position,
		indvbest: indvbest,
		velocity: velocity,
		inertia:  rng.Float32() * maxinertia,
		alpha:    rng.Float32() * maxalpha,
	}
}

//AsyncUpdate does the update asyncrounusly
func (s *Swarm) AsyncUpdate(index int, loss float32) error {
	if index >= len(s.particles) {
		return errors.New("Index Out Of Bounds")
	}
	s.particles[index].isbest(loss)
	if loss < s.loss {
		s.loss = loss
		copy(s.globalposition, s.particles[index].position)
	}
	s.particles[index].update(s.mode, s.cognative, s.social, s.vmax, s.constriction, s.globalposition)
	if s.k < s.kmax {
		s.k++
	}
	return nil
}

//GetParticlePosition returns the particle position of the index passed
func (s *Swarm) GetParticlePosition(index int) []float32 {
	if s.k < s.kmax {
		return s.particles[index].position
	}
	return s.globalposition

}

//SyncUpdate updates the particle swarm after all particles tested
func (s *Swarm) SyncUpdate(losses []float32) error {
	if len(losses) != len(s.particles) {
		return errors.New("Sizes of losses and num of particles not the same")
	}

	for i := range losses {
		s.particles[i].isbest(losses[i])
		if losses[i] < s.loss {
			s.loss = losses[i]
			copy(s.globalposition, s.particles[i].position)

		}
	}
	for i := range s.particles {
		s.particles[i].update(s.mode, s.cognative, s.social, s.vmax, s.constriction, s.globalposition)
	}
	s.k++
	return nil
}
func (p *particle) isbest(loss float32) {
	if loss < p.loss {
		p.loss = loss
		copy(p.indvbest, p.position)
	}
}

//Update will update velocities,and position
func (p *particle) update(mode Mode, cognative, social, vmax, constriction float32, globalbest []float32) {
	switch mode {
	case p.mflg.Vanilla():
		p.vanilla(cognative, social, vmax, globalbest)
	case p.mflg.ConstantInertia():
		p.constant(cognative, social, vmax, globalbest)
	case p.mflg.InertiaReduction():
		p.linearinertiareduce(cognative, social, vmax, globalbest)
	case p.mflg.Constriction():
		p.constriction(cognative, social, vmax, constriction, globalbest)
	case p.mflg.DynamicInertiaMaxVelReduction():
		p.dimvr(cognative, social, vmax, globalbest)
		//case p.mflg.SocialPressure():
	}

}
func (p *particle) dimvr(cognative, social, vmaxgamma float32, globalbest []float32) {
	min := float32(999999999)
	max := float32(-99999999)
	for i := range p.velocity {
		p.velocity[i] = p.inertia*p.velocity[i] + cognative*p.rng.Float32()*(p.indvbest[i]-p.position[i]) + social*p.rng.Float32()*(globalbest[i]-p.position[i])
		if p.position[i] < min {
			min = p.position[i]
		}
		if p.position[i] > max {
			max = p.position[i]
		}

	}
	vmax := vmaxgamma * (max - min)
	for i := range p.velocity {

		if p.velocity[i] > vmax && testingvmax {
			p.velocity[i] = vmax
		}

		p.position[i] += p.velocity[i]
	}
}
func (p *particle) linearinertiareduce(cognative, social, vmax float32, globalbest []float32) {
	for i := range p.velocity {
		p.velocity[i] = (p.alpha * p.inertia * p.velocity[i]) + cognative*p.rng.Float32()*(p.indvbest[i]-p.position[i]) + social*p.rng.Float32()*(globalbest[i]-p.position[i])

		if p.velocity[i] > vmax && testingvmax {
			p.velocity[i] = vmax
		}
		p.inertia *= p.alpha
		p.position[i] += p.velocity[i]
	}
}
func (p *particle) vanilla(cognative, social, vmax float32, globalbest []float32) {
	for i := range p.velocity {
		p.velocity[i] += +cognative*p.rng.Float32()*(p.indvbest[i]-p.position[i]) + social*p.rng.Float32()*(globalbest[i]-p.position[i])

		if p.velocity[i] > vmax && testingvmax {
			p.velocity[i] = vmax
		}

		p.position[i] += p.velocity[i]
	}
}
func (p *particle) constant(cognative, social, vmax float32, globalbest []float32) {
	for i := range p.velocity {
		p.velocity[i] = (p.inertia * p.velocity[i]) + (cognative * p.rng.Float32() * (p.indvbest[i] - p.position[i])) + (social * p.rng.Float32() * (globalbest[i] - p.position[i]))

		if p.velocity[i] > vmax && testingvmax {
			p.velocity[i] = vmax
		}
		p.position[i] += p.velocity[i]
	}
}
func (p *particle) constriction(cognative, social, vmax, constriction float32, globalbest []float32) {
	for i := range p.velocity {
		p.velocity[i] = constriction * (p.velocity[i] + cognative*p.rng.Float32()*(p.indvbest[i]-p.position[i]) + social*p.rng.Float32()*(globalbest[i]-p.position[i]))

		if p.velocity[i] > vmax && testingvmax {
			p.velocity[i] = vmax
		}
		p.position[i] += p.velocity[i]
	}
}
