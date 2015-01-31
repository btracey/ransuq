package main

import (
	"fmt"
	"log"
	"math"
	"os"
	"runtime"

	"code.google.com/p/plotinum/plot"
	"code.google.com/p/plotinum/plotter"

	"github.com/btracey/numcsv"
	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/optimize"
	"github.com/gonum/stat"
	"github.com/reggo/reggo/loss"
	"github.com/reggo/reggo/regularize"
	"github.com/reggo/reggo/scale"
	"github.com/reggo/reggo/supervised/nnet"
	regtrain "github.com/reggo/reggo/train"
)

/*
func init() {
	//mat64.Register(goblas.Blas{}) // use a go-based blas library
	//dbw.Register(goblas.Blas{})
}
*/

var gopath string

func init() {
	gopath = os.Getenv("GOPATH")
	if gopath == "" {
		panic("gopath not set")
	}
}

func main() {
	//rand.Seed(time.Now().UnixNano())     // Set the random number seed
	runtime.GOMAXPROCS(runtime.NumCPU()) // Set the number of processors to use

	trainfile := "exp4_training_800k.txt"
	testfile := "exp4_testing_200k.txt"

	// Open the data file
	f, err := os.Open(trainfile)
	if err != nil {
		log.Fatal(err)
	}

	// Read in the data
	// numcsv is a wrapper I wrote over the normal go csv parser. The Go csv
	// parser returns strings. This assumes that the data is numeric with possibly
	// some column headings at the top, so it returns a matrix of data instead
	// of strings.
	r := numcsv.NewReader(f)
	r.Comma = " " // the file is space dilimeted (ish)
	headings, err := r.ReadHeading()
	if err != nil {
		log.Fatal(err)
	}
	log.Println("The headings are: ", headings)

	allData, err := r.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	log.Print("data loaded")

	f.Close()

	// We know from the file that the output is the last quantity and the inputs
	// are on the first three. We of course could do something more clever if we
	// wanted to, but to keep this code simple, we'll just assume that
	nSamples, nDim := allData.Dims()
	if nDim != 4 {
		log.Fatal("Code assumes there are 4 columns")
	}
	fmt.Println("training data has", nSamples, "rows")

	inputData, outputData := splitData(allData)

	// Let's scale the data to have mean zero and variance 1
	inputScaler := &scale.Normal{}
	inputScaler.SetScale(inputData)
	scale.ScaleData(inputScaler, inputData)

	outputScaler := &scale.Normal{}
	outputScaler.SetScale(outputData)
	scale.ScaleData(outputScaler, outputData)

	// Great! Data is ready. Now let's set up a problem. First, let's define
	// our algoritm
	inputDim := nDim - 1
	outputDim := 1
	nHiddenLayers := 2
	nNeuronsPerLayer := 50          // I usually use more, but let's keep this example cheap
	finalActivator := nnet.Linear{} // doing regression, so use a linear activator in the last output
	algorithm, err := nnet.NewSimpleTrainer(inputDim, outputDim, nHiddenLayers, nNeuronsPerLayer, nnet.Tanh{}, finalActivator)
	if err != nil {
		log.Fatal(err)
	}

	algorithm.RandomizeParameters()
	param := algorithm.Parameters(nil)

	// Now let's define other things
	var weights []float64 = nil                  // Don't weight our data
	losser := loss.SquaredDistance{}             // SquaredDistance loss function
	var regularizer regularize.Regularizer = nil // Let's not place any penalty on large nnet parameter values

	// Create the trainer
	problem := &regtrain.GradOptimizable{
		Trainable: algorithm,
		Inputs:    inputData,
		Outputs:   outputData,
		Weights:   weights,

		NumWorkers:  runtime.GOMAXPROCS(0),
		Losser:      losser,
		Regularizer: regularizer,
	}

	settings := optimize.DefaultSettings()
	settings.FunctionEvals = 100
	settings.GradientAbsTol = 1e-4
	settings.FunctionAbsTol = 1e-4

	problem.Init()
	result, err := optimize.Local(problem, param, settings, &optimize.BFGS{})
	if err != nil {
		log.Fatal(err)
	}
	problem.Close()

	log.Println("Finished optimization")

	algorithm.SetParameters(result.X)

	// Save algorithm

	// Grace added to do more predictions //////////////////
	// create test input matrix
	// Open the data file
	ftest, errtest := os.Open(testfile)
	if errtest != nil {
		log.Fatal(errtest)
	}
	defer ftest.Close()

	rtest := numcsv.NewReader(ftest)
	rtest.Comma = " " // the file is space dilimeted (ish)
	headingstest, errtest := rtest.ReadHeading()
	if errtest != nil {
		log.Fatal(errtest)
	}
	log.Println("The test data headings are: ", headingstest)

	allDatatest, errtest := rtest.ReadAll()
	if errtest != nil {
		log.Fatal(errtest)
	}
	log.Print("test data loaded")

	// We know from the file that the output is the last quantity and the inputs
	// are on the first three. We of course could do something more clever if we
	// wanted to, but to keep this code simple, we'll just assume that
	nSamplestest, nDimtest := allDatatest.Dims()
	if nDimtest != 4 {
		log.Fatal("Code assumes there are 4 columns")
	}

	fmt.Println("nSamplestest =", nSamplestest)

	// Make the input and output data, copied from submatrices of all data
	// Uses the gonum matrix package: https://godoc.org/github.com/gonum/matrix/mat64

	inputDatatest, outputDatatest := splitData(allDatatest)

	// Let's scale the data to have mean zero and variance 1
	scale.ScaleData(inputScaler, inputDatatest)
	//scale.ScaleData(outputScaler, outputDatatest)

	testr, testc := inputDatatest.Dims()
	fmt.Println(testr, testc)

	predOutputRM, err := algorithm.PredictBatch(inputDatatest, nil)
	if err != nil {
		log.Fatal(err)
	}
	predOutput := predOutputRM.(*mat64.Dense)

	scale.UnscaleData(inputScaler, inputDatatest)
	scale.UnscaleData(outputScaler, predOutput)

	truth := outputDatatest.Col(nil, 0)
	pred := predOutput.Col(nil, 0)

	diff := make([]float64, len(truth))
	floats.SubTo(diff, truth, pred)

	rms := floats.Norm(diff, 2)
	rms /= math.Sqrt(float64(len(diff)))

	fmt.Println("root mean squared error = ", rms)
	fmt.Println("MSE", rms*rms)

	sresid := stat.Variance(diff, nil)
	stot := stat.Variance(truth, nil)
	rSquared := 1 - sresid/stot
	fmt.Println("Rsquared =", rSquared)

	scatter, err := plotter.NewScatter(VecXY{truth, pred})
	if err != nil {
		log.Fatal(err)
	}
	scatter.GlyphStyle.Radius = 1

	p, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}

	p.Add(scatter)
	perfect := plotter.NewFunction(func(x float64) float64 { return x })
	// add a straight line through a perfect prediction
	p.Add(perfect)
	p.X.Label.Text = "Truth"
	p.Y.Label.Text = "Predicted"
	err = p.Save(8, 8, "predvstruth.jpg")
	if err != nil {
		log.Fatal(err)
	}
}

type VecXY struct {
	X []float64
	Y []float64
}

func (v VecXY) Len() int {
	if len(v.X) != len(v.Y) {
		panic("length mismatch")
	}
	return len(v.X)
}

func (v VecXY) XY(i int) (x, y float64) {
	return v.X[i], v.Y[i]
}

// splitData splits the data matrix into input and output data. Assumes that the
// there is one output in the last row of the data. A copy is made so modifications
// to inputData and outputData do not affect the original data.
func splitData(allData *mat64.Dense) (inputData, outputData *mat64.Dense) {
	// Make the input and output data, copied from submatrices of all data
	// Uses the gonum matrix package: https://godoc.org/github.com/gonum/matrix/mat64

	nSamples, nDim := allData.Dims()
	inputData = &mat64.Dense{} // allocate a new matrix that the data can be copied into
	outputData = &mat64.Dense{}
	inputData.Clone(allData.View(0, 0, nSamples, nDim-1))
	outputData.Clone(allData.View(0, nDim-1, nSamples, 1))
	return inputData, outputData
}
