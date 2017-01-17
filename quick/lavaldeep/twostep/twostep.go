package main

import (
	"davebench/nnet"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"

	"github.com/btracey/numcsv"
	"github.com/btracey/ransuq/internal/util"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/optimize"
	"github.com/reggo/reggo/loss"
	"github.com/reggo/reggo/regularize"
	"github.com/reggo/reggo/scale"
	"github.com/reggo/reggo/train"
)

func main() {
	wd, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	filename := filepath.Join(wd, "lavalsmall", "laval_csv_100000.dat")
	outputFeatures := []string{"this_Source"}
	ignoreFeatures := []string{}

	inputs, outputs := loadData(filename, outputFeatures)

	inputScaler := &scale.Normal{}
	inputScaler.SetScale(inputData)
	scale.ScaleData(inputScaler, inputData)

	outputScaler := &scale.Normal{}
	outputScaler.SetScale(outputData)
	scale.ScaleData(outputScaler, outputData)

	// Train first algorithm
	nHiddenLayers := 4
	nNeuronsPerLayer := 6
	finalActivator := nnet.Linear{}
	algorithm, err := nnet.NewSimpleTrainer(inputDim, outputDim, nHiddenLayers, nNeuronsPerLayer, nnet.Tanh{}, finalActivator)
	if err != nil {
		log.Fatal(err)
	}

	trainNet(inputs, outputs, algorithm)

	// predictions := mat64.NewDense(, c, mat)

	nSamples, outputDim := outputs.Dims()
	incorrect := mat64.NewDense(nSamples, outputDim, nil)

}

func loadData(filename string, outputFeatures []string) (inputs, outputs *mat64.Dense) {
	f, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	r := numcsv.NewReader(f)
	headings, err := r.ReadHeading()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(headings)
	fmt.Println("num headings = ", len(headings))
	allData, err := r.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	rows, cols := allData.Dims()
	fmt.Println("num samples = ", rows, cols)

	inputIdxs := make([]int, 0)
	outputIdxs := make([]int, 0)
	for i, str := range headings {
		idx := util.FindStringLocation(outputFeatures, str)
		if idx == -1 {
			inputIdxs = append(inputIdxs, i)
		} else {
			outputIdxs = append(outputIdxs, i)
		}
	}
	inputData := mat64.NewDense(rows, len(inputIdxs), nil)
	outputData := mat64.NewDense(rows, len(outputIdxs), nil)
	for i := 0; i < rows; i++ {
		for j, idx := range inputIdxs {
			inputData.Set(i, j, allData.At(i, idx))
		}
		for j, idx := range outputIdxs {
			outputData.Set(i, j, allData.At(i, idx))
		}
	}
	_, _ = inputData.Dims()
	_, outputDim := outputData.Dims()
	if outputDim != 1 {
		log.Fatal("must have exactly one output")
	}
	return inputData, outputData
}

func trainNet(inputs, outputs *mat64.Dense, algorithm *nnet.Trainer) {
	// Now let's define other things
	var weights []float64 = nil                  // Don't weight our data
	losser := loss.SquaredDistance{}             // SquaredDistance loss function
	var regularizer regularize.Regularizer = nil // Let's not place any penalty on large nnet parameter values

	problem := &train.BatchGradient{
		Trainable: algorithm,
		Inputs:    inputData,
		Outputs:   outputData,
		Weights:   weights,

		Workers:     runtime.GOMAXPROCS(0),
		Losser:      losser,
		Regularizer: regularizer,
	}

	problem.Init()

	settings := optimize.DefaultSettings()
	settings.FuncEvaluations = 10000
	settings.GradientThreshold = 1e-4
	settings.FunctionThreshold = 1e-4

	result, err := optimize.Local(problem, param, settings, &optimize.LBFGS{})
	if err != nil {
		log.Fatal(err)
	}
	//problem.Close()

	log.Println("Finished optimization")

	algorithm.SetParameters(result.X)
}
