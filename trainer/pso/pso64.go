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

type particle64 struct {
	rng      *rand.Rand
	mflg     ModeFlag
	source   rand.Source
	loss     float64
	position []float64
	indvbest []float64
	velocity []float64
	inertia  float64
	alpha    float64
	vmax     float64
}

//Swarm64 contains the particles and meta values
type Swarm64 struct {
	k, kmax                               int
	loss                                  float64
	cognative, social, vmax, constriction float64
	particles                             []particle64
	globalposition                        []float64
	bestbool                              []bool
	mode                                  Mode
}

//CreateSwarm64 creates a particle swarm
func CreateSwarm64(mode Mode, numofparticles, dims, seed, kmax int, cognative, social, vmax, pminstart, pmaxstart, alphamax, inertiamax float64) Swarm64 {
	rand.Seed(int64(seed))

	particles64 := make([]particle64, numofparticles)
	for i := range particles64 {
		particles64[i] = createparticle64(vmax, pminstart, pmaxstart, alphamax, inertiamax, dims, rand.Int63())

	}
	gamma := float64(social + cognative)
	constriction := 2 / (2 - gamma - math.Sqrt((gamma*gamma)-4*gamma))
	return Swarm64{
		k:            1,
		cognative:    cognative,
		social:       social,
		kmax:         kmax,
		vmax:         vmax,
		loss:         9999999999,
		particles:    particles64,
		constriction: float64(constriction),
	}
}

func createparticle64(maxv, pminstart, pmaxstart, maxalpha, maxinertia float64, dims int, seed int64) particle64 {
	source := rand.NewSource(seed)
	rng := rand.New(source)
	position := make([]float64, dims)
	indvbest := make([]float64, dims)
	velocity := make([]float64, dims)
	var val float64
	for i := range position {
		val = (rng.Float64() * (pmaxstart - pminstart)) + pminstart
		position[i] = val
		indvbest[i] = val
		velocity[i] = rng.Float64() * maxv
	}
	return particle64{
		rng:      rng,
		source:   source,
		loss:     999999999999,
		position: position,
		indvbest: indvbest,
		velocity: velocity,
		inertia:  rng.Float64() * maxinertia,
		alpha:    rng.Float64() * maxalpha,
	}
}

//AsyncUpdate does the update asyncrounusly
func (s *Swarm64) AsyncUpdate(index int, loss float64) error {
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
func (s *Swarm64) GetParticlePosition(index int) []float64 {
	if s.k < s.kmax {
		return s.particles[index].position

	}
	return s.globalposition
}

//SyncUpdate updates the particle swarm after all particles tested
func (s *Swarm64) SyncUpdate(losses []float64) error {
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
func (p *particle64) isbest(loss float64) {
	if loss < p.loss {
		p.loss = loss
		copy(p.indvbest, p.position)
	}
}

//Update will update velocities,and position
func (p *particle64) update(mode Mode, cognative, social, vmax, constriction float64, globalbest []float64) {
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
func (p *particle64) dimvr(cognative, social, vmaxgamma float64, globalbest []float64) {
	min := float64(999999999)
	max := float64(-99999999)
	for i := range p.velocity {
		p.velocity[i] = p.inertia*p.velocity[i] + cognative*p.rng.Float64()*(p.indvbest[i]-p.position[i]) + social*p.rng.Float64()*(globalbest[i]-p.position[i])
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
func (p *particle64) linearinertiareduce(cognative, social, vmax float64, globalbest []float64) {
	for i := range p.velocity {
		p.velocity[i] = (p.alpha * p.inertia * p.velocity[i]) + cognative*p.rng.Float64()*(p.indvbest[i]-p.position[i]) + social*p.rng.Float64()*(globalbest[i]-p.position[i])

		if p.velocity[i] > vmax && testingvmax {
			p.velocity[i] = vmax
		}
		p.inertia *= p.alpha
		p.position[i] += p.velocity[i]
	}
}
func (p *particle64) vanilla(cognative, social, vmax float64, globalbest []float64) {
	for i := range p.velocity {
		p.velocity[i] += +cognative*p.rng.Float64()*(p.indvbest[i]-p.position[i]) + social*p.rng.Float64()*(globalbest[i]-p.position[i])

		if p.velocity[i] > vmax && testingvmax {
			p.velocity[i] = vmax
		}

		p.position[i] += p.velocity[i]
	}
}
func (p *particle64) constant(cognative, social, vmax float64, globalbest []float64) {
	for i := range p.velocity {
		p.velocity[i] = (p.inertia * p.velocity[i]) + (cognative * p.rng.Float64() * (p.indvbest[i] - p.position[i])) + (social * p.rng.Float64() * (globalbest[i] - p.position[i]))

		if p.velocity[i] > vmax && testingvmax {
			p.velocity[i] = vmax
		}
		p.position[i] += p.velocity[i]
	}
}
func (p *particle64) constriction(cognative, social, vmax, constriction float64, globalbest []float64) {
	for i := range p.velocity {
		p.velocity[i] = constriction * (p.velocity[i] + cognative*p.rng.Float64()*(p.indvbest[i]-p.position[i]) + social*p.rng.Float64()*(globalbest[i]-p.position[i]))

		if p.velocity[i] > vmax && testingvmax {
			p.velocity[i] = vmax
		}
		p.position[i] += p.velocity[i]
	}
}
