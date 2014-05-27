package mlalg

import (
	"math/rand"
	"testing"

	"github.com/reggo/reggo/common/regtest"
	"github.com/reggo/reggo/supervised/nnet"
)

var nSampleSlice = []int{1, 2, 3, 4, 5, 8, 16, 100, 102}

func TestDeriv(t *testing.T) {
	//for i, test := range netIniters {
	inputDim := 5
	outputDim := 3
	nHiddenLayers := 2
	nNeuronsPerLayer := 10
	finalLayerActivator := nnet.Linear{}

	//nSamples := 100

	net, err := nnet.NewSimpleTrainer(inputDim-1, outputDim, nHiddenLayers, nNeuronsPerLayer, finalLayerActivator)
	if err != nil {
		panic(err)
	}

	mul := MulTrainer{net}

	for _, nSamples := range nSampleSlice {
		inputs := regtest.RandomMat(nSamples, inputDim, rand.NormFloat64)
		trueOutputs := regtest.RandomMat(nSamples, outputDim, rand.NormFloat64)
		regtest.TestDeriv(t, mul, inputs, trueOutputs, "mul_trainer")
	}
}
