// Copyright 2021 The Hebbian Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
	"math/cmplx"

	"github.com/pointlander/datum/iris"
)

const (
	// LFSRMask is a LFSR mask with a maximum period
	LFSRMask = 0x80000057
	// LFSRInit is an initial LFSR state
	LFSRInit = 0x55555555
)

var (
	// FlagXOR xor flag
	FlagXOR = flag.Bool("xor", false, "xor mode")
	// FlagQXOR xor flag
	FlagQXOR = flag.Bool("qxor", false, "quantum xor mode")
	// FlagIRIS iris flag
	FlagIRIS = flag.Bool("iris", false, "iris mode")
)

// Rand is a random number generator
type Rand uint32

// Float32 returns a random float32 between 0 and 1
func (r *Rand) Float32() float32 {
	lfsr := *r
	if lfsr&1 == 1 {
		lfsr = (lfsr >> 1) ^ LFSRMask
	} else {
		lfsr = lfsr >> 1
	}
	*r = lfsr
	return float32(lfsr) / ((1 << 32) - 1)
}

// Uint32 returns a random uint32
func (r *Rand) Uint32() uint32 {
	lfsr := *r
	if lfsr&1 == 1 {
		lfsr = (lfsr >> 1) ^ LFSRMask
	} else {
		lfsr = lfsr >> 1
	}
	*r = lfsr
	return uint32(lfsr)
}

// Neuron is a neuron
type Neuron struct {
	Inputs   []float32
	Output   float32
	Learn    float32
	Weights  []float32
	DWeights []float32
}

// Layer is a neural network layer
type Layer []Neuron

// Layers is layers of networks
type Layers []Layer

// Inference computes the neuron
func (n *Neuron) Inference() {
	var sum, sumWeight, sumInput float32
	for i := range n.Weights {
		sum += n.Weights[i] * n.Inputs[i]
		sumWeight += n.Weights[i]
		sumInput += n.Inputs[i]
	}
	e := float32(math.Exp(float64(sum)))
	i := float32(math.Exp(float64(-sum)))
	n.Output = (e - i) / (e + i)

	sum -= sumWeight * sumInput
	e = float32(math.Exp(float64(sum)))
	i = float32(math.Exp(float64(-sum)))
	n.Learn = (e - i) / (e + i)
}

// QuantumNeuron is a quantum neuron
type QuantumNeuron struct {
	Inputs   []complex64
	Output   complex64
	Learn    complex64
	Weights  []complex64
	DWeights []complex64
}

// QuantumLayer is a quantum neural network layer
type QuantumLayer []QuantumNeuron

// QuantumLayers is layers of quantum networks
type QuantumLayers []QuantumLayer

// Inference computes the neuron
func (n *QuantumNeuron) Inference() {
	var sum, sumWeight, sumInput complex64
	for i := range n.Weights {
		sum += n.Weights[i] * n.Inputs[i]
		sumWeight += n.Weights[i]
		sumInput += n.Inputs[i]
	}
	e := complex64(cmplx.Exp(complex128(sum)))
	i := complex64(cmplx.Exp(complex128(-sum)))
	n.Output = (e - i) / (e + i)

	sum -= sumWeight * sumInput
	e = complex64(cmplx.Exp(complex128(sum)))
	i = complex64(cmplx.Exp(complex128(-sum)))
	n.Learn = (e - i) / (e + i)
}

func main() {
	flag.Parse()

	if *FlagXOR {
		XOR()
		return
	} else if *FlagQXOR {
		QXOR()
		return
	} else if *FlagIRIS {
		IRIS()
		return
	}
}

// XOR mode
func XOR() {
	network := make([]Layer, 2)
	network[0] = make([]Neuron, 2)
	network[1] = make([]Neuron, 1)
	g, factor := Rand(LFSRInit), float32(math.Sqrt(2/float64(2)))
	for i := range network {
		for j := range network[i] {
			network[i][j].Inputs = make([]float32, 2)
			network[i][j].Weights = make([]float32, 2)
			network[i][j].DWeights = make([]float32, 2)
			for k := range network[i][j].Weights {
				network[i][j].Weights[k] = (2*g.Float32() - 1) * factor
			}
		}
	}

	xor := [][]float32{
		[]float32{-1, -1, -1},
		[]float32{1, -1, 1},
		[]float32{-1, 1, 1},
		[]float32{1, 1, -1},
	}
	n := float32(.01)
	for i := 0; i < 16; i++ {
		cost := float32(0)
		for j := range xor {
			network[0][0].Inputs[0] = xor[j][0]
			network[0][0].Inputs[1] = xor[j][1]
			network[0][0].Inference()
			network[0][1].Inputs[0] = xor[j][0]
			network[0][1].Inputs[1] = xor[j][1]
			network[0][1].Inference()
			network[1][0].Inputs[0] = network[0][0].Output
			network[1][0].Inputs[1] = network[0][1].Output
			network[1][0].Inference()
			for k := range network {
				for l := range network[k] {
					for m := range network[k][l].Inputs {
						network[k][l].DWeights[m] -= n * network[k][l].Inputs[m] * network[k][l].Learn
					}
				}
			}
			diff := network[1][0].Output - xor[j][2]
			fmt.Println(network[1][0].Output, xor[j][2])
			cost += diff * diff
		}
		for k := range network {
			for l := range network[k] {
				for m := range network[k][l].Inputs {
					network[k][l].Weights[m] += network[k][l].DWeights[m]
					network[k][l].DWeights[m] = 0
				}
			}
		}
		fmt.Println(cost)
	}
}

// QXOR mode
func QXOR() {
	network := make([]QuantumLayer, 2)
	network[0] = make([]QuantumNeuron, 2)
	network[1] = make([]QuantumNeuron, 1)
	g, factor := Rand(LFSRInit), float32(math.Sqrt(2/float64(2)))
	for i := range network {
		for j := range network[i] {
			network[i][j].Inputs = make([]complex64, 2)
			network[i][j].Weights = make([]complex64, 2)
			network[i][j].DWeights = make([]complex64, 2)
			for k := range network[i][j].Weights {
				network[i][j].Weights[k] = complex((2*g.Float32()-1)*factor, (2*g.Float32()-1)*factor)
			}
		}
	}

	xor := [][]complex64{
		[]complex64{-1, -1, -1},
		[]complex64{1, -1, 1},
		[]complex64{-1, 1, 1},
		[]complex64{1, 1, -1},
	}
	n := complex64(.01)
	for i := 0; i < 16; i++ {
		cost := float32(0)
		for j := range xor {
			network[0][0].Inputs[0] = xor[j][0]
			network[0][0].Inputs[1] = xor[j][1]
			network[0][0].Inference()
			network[0][1].Inputs[0] = xor[j][0]
			network[0][1].Inputs[1] = xor[j][1]
			network[0][1].Inference()
			network[1][0].Inputs[0] = network[0][0].Output
			network[1][0].Inputs[1] = network[0][1].Output
			network[1][0].Inference()
			for k := range network {
				for l := range network[k] {
					for m := range network[k][l].Inputs {
						network[k][l].DWeights[m] -= n * network[k][l].Inputs[m] * network[k][l].Learn
					}
				}
			}
			diff := float32(cmplx.Abs(complex128(network[1][0].Output - xor[j][2])))
			fmt.Println(network[1][0].Output, xor[j][2])
			cost += diff * diff
		}
		for k := range network {
			for l := range network[k] {
				for m := range network[k][l].Inputs {
					network[k][l].Weights[m] += network[k][l].DWeights[m]
					network[k][l].DWeights[m] = 0
				}
			}
		}
		fmt.Println(cost)
	}
}

// IRIS mode
func IRIS() {
	network := make([]Layer, 2)
	network[0] = make([]Neuron, 4)
	network[1] = make([]Neuron, 4)
	g, factor := Rand(LFSRInit), float32(math.Sqrt(2/float64(2)))
	for i := range network {
		for j := range network[i] {
			network[i][j].Inputs = make([]float32, 4)
			network[i][j].Weights = make([]float32, 4)
			network[i][j].DWeights = make([]float32, 4)
			for k := range network[i][j].Weights {
				network[i][j].Weights[k] = (2*g.Float32() - 1) * factor
			}
		}
	}

	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}

	n := float32(.5)
	for i := 0; i < 1; i++ {
		for _, flower := range datum.Fisher {
			for neuron := range network[0] {
				for i, value := range flower.Measures {
					network[0][neuron].Inputs[i] = float32(value)
				}
				network[0][neuron].Inference()
			}
			for neuron := range network[1] {
				network[1][neuron].Inputs[0] = network[0][0].Output
				network[1][neuron].Inputs[1] = network[0][1].Output
				network[1][neuron].Inputs[2] = network[0][2].Output
				network[1][neuron].Inputs[3] = network[0][3].Output
				network[1][neuron].Inference()
			}
			for k := range network {
				for l := range network[k] {
					for m := range network[k][l].Inputs {
						network[k][l].DWeights[m] -= n * network[k][l].Inputs[m] * network[k][l].Learn
					}
				}
			}
			fmt.Println(iris.Labels[flower.Label],
				network[1][0].Output, network[1][1].Output,
				network[1][2].Output, network[1][3].Output)
		}
		for k := range network {
			for l := range network[k] {
				for m := range network[k][l].Inputs {
					network[k][l].Weights[m] += network[k][l].DWeights[m]
					network[k][l].DWeights[m] = 0
				}
			}
		}
	}
}
