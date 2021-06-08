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
	Inputs   [2]float32
	Output   float32
	Learn    float32
	Weights  [2]float32
	DWeights [2]float32
}

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
	g, factor, neurons := Rand(LFSRInit), float32(math.Sqrt(2/float64(2))), make([]Neuron, 3)
	for i := range neurons {
		for j := range neurons[i].Weights {
			neurons[i].Weights[j] = (2*g.Float32() - 1) * factor
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
			neurons[0].Inputs[0] = xor[j][0]
			neurons[0].Inputs[1] = xor[j][1]
			neurons[0].Inference()
			neurons[1].Inputs[0] = xor[j][0]
			neurons[1].Inputs[1] = xor[j][1]
			neurons[1].Inference()
			neurons[2].Inputs[0] = neurons[0].Output
			neurons[2].Inputs[1] = neurons[1].Output
			neurons[2].Inference()
			for k := range neurons {
				for l := range neurons[k].Inputs {
					neurons[k].DWeights[l] -= n * neurons[k].Inputs[l] * neurons[k].Learn
				}
			}
			diff := neurons[2].Output - xor[j][2]
			fmt.Println(neurons[2].Output, xor[j][2])
			cost += diff * diff
		}
		for k := range neurons {
			for l := range neurons[k].DWeights {
				neurons[k].Weights[l] += neurons[k].DWeights[l]
				neurons[k].DWeights[l] = 0
			}
		}
		fmt.Println(cost)
	}
}
