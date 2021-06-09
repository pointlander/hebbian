// Copyright 2021 The Hebbian Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
)

const (
	// LFSRMask is a LFSR mask with a maximum period
	LFSRMask = 0x80000057
	// LFSRInit is an initial LFSR state
	LFSRInit = 0x55555555
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

func main() {
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
