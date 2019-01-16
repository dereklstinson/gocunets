/*

pso is based on the slides from Jaco F. Schutte EGM 6365 - Structural Optimization Fall 2005
Link to slides https://www.mii.lt/zilinskas/uploads/Heuristic%20Algorithms/Lectures/Lect4/PSO2.pdf

*/

package pso

import (
	"errors"
	"math/rand"
)

type particle struct {
	rng      *rand.Rand
	source   rand.Source
	loss     float32
	position []float32
	indvbest []float32
	velocity []float32
}

//Swarm contains the particles and meta values
type Swarm struct {
	k, kmax                                 int
	loss                                    float32
	alpha, inertia, cognative, social, vmax float32
	particles                               []particle
	globalposition                          []float32
	bestbool                                []bool
	mode                                    Mode
}

//Mode is the mode flag for the swarm
type Mode int32

type ModeFlag struct {
}

func (m ModeFlag) Vanilla() Mode {
	return Mode(1)
}
func (m ModeFlag) ConstantInertia() Mode {
	return Mode(2)
}
func (m ModeFlag) InertiaReduction() Mode {
	return Mode(3)
}
func (m ModeFlag) Constriction() Mode {
	return Mode(4)
}
func (m ModeFlag) DynamicInertiaMaxVelReduction() Mode {
	return Mode(5)
}
func (m ModeFlag) SocialPressure() Mode {
	return Mode(6)
}

//CreateSwarm creates a particle swarm
func CreateSwarm(mode Mode, numofparticles, dims, seed, kmax int, cognative, social, vmax, alpha, inertia float32) Swarm {
	rand.Seed(int64(seed))

	particles := make([]particle, numofparticles)
	for i := range particles {
		particles[i] = createparticle(vmax, dims, rand.Int63())

	}
	return Swarm{
		k:         1,
		cognative: cognative,
		social:    social,
		kmax:      kmax,
		vmax:      vmax,
		loss:      9999999999,
		particles: particles,
		alpha:     alpha,
		inertia:   inertia,
	}
}

func createparticle(maxv float32, dims int, seed int64) particle {
	source := rand.NewSource(seed)
	rng := rand.New(source)
	position := make([]float32, dims)
	indvbest := make([]float32, dims)
	velocity := make([]float32, dims)
	for i := range position {
		val := rng.Float32()
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
		s.globalposition = s.particles[index].position
	}
	s.particles[index].update(s.mode, s.cognative, s.social, s.vmax, s.alpha, s.inertia, s.globalposition)
	return nil
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
			s.globalposition = s.particles[i].position
		}
	}
	for i := range s.particles {
		s.particles[i].update(s.mode, s.cognative, s.social, s.vmax, s.alpha, s.inertia, s.globalposition)
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
func (p *particle) update(mode Mode, cognative, social, vmax, alpha, inertia float32, globalbest []float32) {
	if alpha == inertia && alpha == 1.0 {
		for i := range p.velocity {
			p.velocity[i] = p.velocity[i] + cognative*p.rng.Float32()*(p.indvbest[i]-p.position[i]) + social*p.rng.Float32()*(globalbest[i]-p.position[i])

			if p.velocity[i] > vmax {
				p.velocity[i] = vmax
			}

			p.position[i] += p.velocity[i]
		}
	} else {
		for i := range p.velocity {
			p.velocity[i] = alpha*inertia*p.velocity[i] + cognative*p.rng.Float32()*(p.indvbest[i]-p.position[i]) + social*p.rng.Float32()*(globalbest[i]-p.position[i])

			if p.velocity[i] > vmax {
				p.velocity[i] = vmax
			}

			p.position[i] += p.velocity[i]
		}
	}

}
