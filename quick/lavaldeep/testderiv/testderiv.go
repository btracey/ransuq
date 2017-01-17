package main

import (
	"fmt"
	"log"
	"math/rand"
	"runtime"

	"github.com/gonum/diff/fd"
	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/optimize"
	"github.com/gonum/optimize/functions"
	"github.com/reggo/reggo/loss"
	"github.com/reggo/reggo/regularize"
	"github.com/reggo/reggo/supervised/nnet"
	"github.com/reggo/reggo/train"
)

func init() {
	runtime.GOMAXPROCS(runtime.NumCPU())
}

func main() {
	nSamples := 30
	inputDim := 3
	outputDim := 1
	inputs := mat64.NewDense(nSamples, inputDim, nil)
	outputs := mat64.NewDense(nSamples, outputDim, nil)

	x := make([]float64, inputDim)
	for i := 0; i < nSamples; i++ {
		for j := range x {
			x[j] = rand.Float64() * 10
		}
		f := functions.ExtendedRosenbrock{}.Func(x)
		inputs.SetRow(i, x)
		outputs.Set(i, 0, f)
	}

	nHiddenLayers := 1
	nNeuronsPerLayer := 4           // I usually use more, but let's keep this example cheap
	finalActivator := nnet.Linear{} // doing regression, so use a linear activator in the last output
	algorithm, err := nnet.NewSimpleTrainer(inputDim, outputDim, nHiddenLayers, nNeuronsPerLayer, nnet.Tanh{}, finalActivator)
	if err != nil {
		log.Fatal(err)
	}

	algorithm.RandomizeParameters()
	param := algorithm.Parameters(nil)
	fmt.Println("num param = ", len(param))

	// Now let's define other things
	var weights []float64 = nil                  // Don't weight our data
	losser := loss.SquaredDistance{}             // SquaredDistance loss function
	var regularizer regularize.Regularizer = nil // Let's not place any penalty on large nnet parameter values

	problem := &train.BatchGradient{
		Trainable: algorithm,
		Inputs:    inputs,
		Outputs:   outputs,
		Weights:   weights,

		Workers:     runtime.GOMAXPROCS(0),
		Losser:      losser,
		Regularizer: regularizer,
	}

	settings := optimize.DefaultSettings()
	settings.FuncEvaluations = 1000
	settings.GradientThreshold = 1e-4
	settings.FunctionThreshold = 1e-4

	problem.Init()

	//fdSettings := fd.DefaultSettings()
	//fdSettings.Concurrent = false
	fdgrad := fd.Gradient(nil, problem.Func, param, nil)
	angrad := make([]float64, len(param))
	problem.Grad(param, angrad)
	if !floats.EqualApprox(fdgrad, angrad, 1e-5) {
		fmt.Println(fdgrad)
		fmt.Println(angrad)
		log.Fatal("gradient mismatch")
	}
}
