package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/btracey/myplot"
	"github.com/btracey/numcsv"
	"github.com/btracey/ransuq/internal/util"
	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/optimize"
	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/vg"
	"github.com/reggo/reggo/common"
	"github.com/reggo/reggo/loss"
	"github.com/reggo/reggo/regularize"
	"github.com/reggo/reggo/scale"
	"github.com/reggo/reggo/supervised/nnet"
	"github.com/reggo/reggo/train"
)

//var _ optimize.Gradient = &train.GradOptimizable{}

var outputFeatures = []string{"this_Source"}

func init() {
	runtime.GOMAXPROCS(runtime.NumCPU())
}

func main() {
	rn := time.Now().UnixNano()
	fmt.Println("rn = ", rn)
	rand.Seed(rn)

	//filename := "/Users/brendan/Documents/mygo/data/ransuq/laval/laval_csv_mldata.dat"

	wd, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	filename := filepath.Join(wd, "lavalsmall", "laval_csv_100000.dat")

	inputData, outputData := getInputOutputData(filename)
	rows, inputDim := inputData.Dims()
	_, outputDim := outputData.Dims()

	// Let's scale the data to have mean zero and variance 1
	inputScaler := &scale.Normal{}
	inputScaler.SetScale(inputData)
	scale.ScaleData(inputScaler, inputData)

	outputScaler := &scale.Normal{}
	outputScaler.SetScale(outputData)
	scale.ScaleData(outputScaler, outputData)

	// Great! Data is ready. Now let's set up a problem. First, let's define
	// our algoritm
	// 7 and 100 for true deep net
	nHiddenLayers := 4
	nNeuronsPerLayer := 6           // I usually use more, but let's keep this example cheap
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

	prediction := make([]float64, rows)
	truth := make([]float64, rows)
	pred := make([]float64, 1)
	for i := range prediction {
		truth[i] = outputData.At(i, 0)
		algorithm.Predict(inputData.Row(nil, i), pred)
		prediction[i] = pred[0]
	}

	plotPredVsTruth(prediction, truth, "pred_vs_true_training.jpg")

	f3, err := os.Create("algorithm.json")
	if err != nil {
		log.Fatal(err)
	}
	sp := ScalePredictor{algorithm, inputScaler, outputScaler}
	b, err := json.MarshalIndent(sp, "", "\t")
	f3.Write(b)
	f3.Close()

	// Read in the full data
	fullFilename := "/Users/brendan/Documents/mygo/data/ransuq/laval/laval_csv_mldata.dat"
	fullInputs, fullOutputs := getInputOutputData(fullFilename)
	fullRows, _ := fullInputs.Dims()
	fullPred := make([]float64, fullRows)
	fullTruth := make([]float64, fullRows)
	for i := 0; i < fullRows; i++ {
		fullTruth[i] = fullOutputs.At(i, 0)
		v, err := sp.Predict(fullInputs.Row(nil, i), nil)
		if err != nil {
			log.Fatal(err)
		}
		fullPred[i] = v[0]
	}
	plotPredVsTruth(fullPred, fullTruth, "pred_vs_true_full.jpg")
	dist := floats.Distance(fullPred, fullTruth, 2)
	fmt.Println("dist is ", dist)
	fmt.Println("per is ", dist/float64(fullRows))
}

func plotPredVsTruth(prediction, truth []float64, filename string) {
	pts := myplot.VecXY{X: prediction, Y: truth}
	p, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}
	scatter, err := plotter.NewScatter(pts)
	if err != nil {
		log.Fatal(err)
	}
	p.X.Label.Text = "prediction"
	p.Y.Label.Text = "truth"
	p.Add(scatter)
	err = p.Save(4*vg.Inch, 4*vg.Inch, filename)
	if err != nil {
		fmt.Println(err)
	}
}

func getInputOutputData(filename string) (inputs, outputs *mat64.Dense) {
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

// Fit is the data that has been trained (and can be saved etc.)
type ScalePredictor struct {
	common.Predictor
	InputScaler  scale.Scaler
	OutputScaler scale.Scaler
}

func (s *ScalePredictor) Predict(input, output []float64) ([]float64, error) {
	s.InputScaler.Scale(input)
	var err error
	output, err = s.Predictor.Predict(input, output)
	s.InputScaler.Unscale(input)
	s.OutputScaler.Unscale(output)
	return output, err
}

func (s *ScalePredictor) MarshalJSON() ([]byte, error) {
	// Make a couple of random inputs
	nRand := 5
	inputs := make([][]float64, nRand)
	outputs := make([][]float64, nRand)
	for i := range inputs {
		inputs[i] = make([]float64, s.Predictor.InputDim())
		for j := range inputs[i] {
			inputs[i][j] = rand.NormFloat64()
		}
		outputs[i] = make([]float64, s.Predictor.OutputDim())
		// Inputs are already scaled. May need to change this for different scalers
		s.Predictor.Predict(inputs[i], outputs[i])
		s.InputScaler.Unscale(inputs[i])
		s.OutputScaler.Unscale(outputs[i])
	}

	// Save the algorithm as well as the scalares
	saveStruct := struct {
		Predictor    common.InterfaceMarshaler
		InputScaler  common.InterfaceMarshaler
		OutputScaler common.InterfaceMarshaler
		TestInputs   [][]float64
		TestOutputs  [][]float64
	}{
		Predictor:    common.InterfaceMarshaler{I: s.Predictor},
		InputScaler:  common.InterfaceMarshaler{I: s.InputScaler},
		OutputScaler: common.InterfaceMarshaler{I: s.OutputScaler},
		TestInputs:   inputs,
		TestOutputs:  outputs,
	}

	return json.Marshal(saveStruct)
}
