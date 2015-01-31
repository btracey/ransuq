package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/btracey/numcsv"
	"github.com/btracey/opt/multivariate"
	"github.com/gonum/blas/dbw"
	"github.com/gonum/blas/goblas"
	"github.com/gonum/matrix/mat64"
	"github.com/reggo/reggo/loss"
	"github.com/reggo/reggo/regularize"
	"github.com/reggo/reggo/scale"
	"github.com/reggo/reggo/supervised/nnet"
	"github.com/reggo/reggo/train"
)

func init() {
	mat64.Register(goblas.Blas{}) // use a go-based blas library
	dbw.Register(goblas.Blas{})
}

var gopath string

func init() {
	gopath = os.Getenv("GOPATH")
	if gopath == "" {
		panic("gopath not set")
	}
}

func main() {
	rand.Seed(time.Now().UnixNano())     // Set the random number seed
	runtime.GOMAXPROCS(runtime.NumCPU()) // Set the number of processors to use

	//filename := "exp4_training_800k.txt"
	filename := filepath.Join(gopath, "data", "ransuq", "HiFi", "exp4.txt")

	// Open the data file
	f, err := os.Open(filename)
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

	// We know from the file that the output is the last quantity and the inputs
	// are on the first three. We of course could do something more clever if we
	// wanted to, but to keep this code simple, we'll just assume that
	nSamples, nDim := allData.Dims()
	if nDim != 4 {
		log.Fatal("Code assumes there are 4 columns")
	}

	// Make the input and output data, copied from submatrices of all data
	// Uses the gonum matrix package: https://godoc.org/github.com/gonum/matrix/mat64
	inputData := &mat64.Dense{} // allocate a new matrix that the data can be copied into
	outputData := &mat64.Dense{}
	//inputData.Submatrix(allData, 0, 0, nSamples, nDim-1)  // copy the first nDim - 1 columns to inputs
	//outputData.Submatrix(allData, 0, nDim-1, nSamples, 1) // copy the last column

	tmp := &mat64.Dense{}
	tmp.View(allData, 0, 0, nSamples, nDim-1)
	inputData.Copy(tmp)

	tmp = &mat64.Dense{}
	tmp.View(allData, 0, nDim-1, nSamples, 1)
	inputData.Copy(tmp)

	// Let's scale the data to have mean zero and variance 1
	inputScaler := &scale.Normal{}
	scale.ScaleData(inputScaler, inputData)

	outputScaler := &scale.Normal{}
	scale.ScaleData(outputScaler, outputData)

	// Great! Data is ready. Now let's set up a problem. First, let's define
	// our algoritm
	inputDim := nDim - 1
	outputDim := 1
	nHiddenLayers := 2
	nNeuronsPerLayer := 30          // I usually use more, but let's keep this example cheap
	finalActivator := nnet.Linear{} // doing regression, so use a linear activator in the last output
	algorithm, err := nnet.NewSimpleTrainer(inputDim, outputDim, nHiddenLayers, nNeuronsPerLayer, nnet.Tanh{}, finalActivator)
	if err != nil {
		log.Fatal(err)
	}

	// Now let's define other things
	var weights []float64 = nil                  // Don't weight our data
	cacheFeatures := true                        // Have the algorithm use more memory to cache temporary results
	losser := loss.SquaredDistance{}             // SquaredDistance loss function
	var regularizer regularize.Regularizer = nil // Let's not place any penalty on large nnet parameter values

	// Now, let's define how we want to compute the gradient. I only have Batch
	// coded up. Batch will sum up the losses and derivatives from all of the data
	// at each step.
	batch := train.NewBatchGradBased(algorithm, cacheFeatures, inputData, outputData, weights, losser, regularizer)

	// Calling batch.ObjGrad(parameters) computes the loss function, and the derivative
	// of the loss function with respect to the parameters. Now, we need to choose
	// how we want to define that loss function
	optSettings := multivariate.DefaultSettings()
	optSettings.GradAbsTol = 1e-4               // Stop if the gradient gets below 1e-4
	optSettings.ObjAbsTol = 1e-4                // Stop if the average prediction is within 1% accurate
	optSettings.MaximumFunctionEvaluations = 50 // Otherwise, stop after 50 cycles, I usually use more

	// Set a random initial starting condition
	algorithm.RandomizeParameters()
	initLoc := algorithm.Parameters(nil)

	// Run the optimization using whatever the optimization package deems appropriate
	// (Because I also wrote the optimization package, this is BFGS)
	//
	// Optimizer will say that the objective is infinity at the first iteration.
	// Don't worry, it's a printing bug in the optimizer I haven't had time to fix.
	result, err := multivariate.OptimizeGrad(batch, initLoc, optSettings, nil)
	if err != nil {
		log.Fatal(err)
	}

	log.Println("Finished running optimization. Average squared loss is ", result.Obj)

	// Set the parameters of the net
	algorithm.SetParameters(result.Loc)

	//	// Generate a random point to predict
	//	newloc := make([]float64, inputDim)
	//	for i := range newloc {
	//		newloc[i] = rand.NormFloat64()
	//	}
	//
	//	// This newloc is already scaled, which is not how data normally comes. Let's unscale it first
	//	// so we can pretend it's a real new datapoint.
	//	inputScaler.Unscale(newloc)
	//
	//	// Now newloc looks like a normal point. Let's scale it because the nnet data
	//	// is scaled
	//	inputScaler.Scale(newloc)
	//
	//	prediction := make([]float64, outputDim)
	//	algorithm.Predict(newloc, prediction)

	// Grace added to do more predictions //////////////////
	// create test input matrix
	testfile := "exp4_testing_200k.txt"
	// Open the data file
	ftest, errtest := os.Open(testfile)
	if errtest != nil {
		log.Fatal(errtest)
	}
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

	// Make the input and output data, copied from submatrices of all data
	// Uses the gonum matrix package: https://godoc.org/github.com/gonum/matrix/mat64
	inputDatatest := &mat64.Dense{} // allocate a new matrix that the data can be copied into
	outputDatatest := &mat64.Dense{}
	inputDatatest.Submatrix(allDatatest, 0, 0, nSamplestest, nDimtest-1)  // copy the first nDim - 1 columns to inputs
	outputDatatest.Submatrix(allDatatest, 0, nDimtest-1, nSamplestest, 1) // copy the last column

	// Let's scale the data to have mean zero and variance 1
	scale.ScaleData(inputScaler, inputDatatest)
	scale.ScaleData(outputScaler, outputDatatest)

	// Now, test batch
	// With non-nil
	outputs := mat64.NewDense(nSamplestest, outputDim, nil)
	predOutput, errtest := algorithm.PredictBatch(inputDatatest, outputs)

	scale.UnscaleData(inputScaler, inputDatatest)
	scale.UnscaleData(outputScaler, mat64.DenseCopyOf(predOutput))

	fmt.Println(fmt.Sprintf("%v", predOutput))

}
