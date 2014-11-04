// linnet takes out the linear term from the data and then trains a neural net
// on the answer

package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"partial"
	"runtime"
	"sort"
	"time"

	"code.google.com/p/plotinum/plot"
	"code.google.com/p/plotinum/plotter"

	"github.com/btracey/ransuq"
	"github.com/gonum/blas/cblas"
	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/optimize"
	"github.com/gonum/stat"
	"github.com/reggo/reggo/loss"
	"github.com/reggo/reggo/regularize"
	"github.com/reggo/reggo/scale"
	"github.com/reggo/reggo/supervised/nnet"
	"github.com/reggo/reggo/train"

	"github.com/btracey/ransuq/grid"
	"github.com/btracey/ransuq/mlalg"
	"github.com/btracey/ransuq/settings"
	"github.com/btracey/su2tools/driver"
)

func init() {
	mat64.Register(cblas.Blas{})
	//mat64.Register(goblas.Blas{})
}

type BoundKind int

const (
	NormalBounds BoundKind = iota
	QuantileBounds
)

// TODO: Stochastic gradient descent

type Case struct {
	Name          string
	Dataset       string
	Features      string
	MulNet        bool
	BoundKind     BoundKind
	BoundPoints   int
	OptIter       int
	Neurons       int
	Regularizer   regularize.Regularizer
	TwoStage      bool
	MakeLinearFit bool
}

var SALavalSmall = Case{
	Name:      "SALavalSmall",
	Dataset:   "laval_dns_sa",
	Features:  "nondim_source_irrotational",
	MulNet:    false,
	BoundKind: QuantileBounds,
	//BoundPoints:   1000,
	BoundPoints:   20,
	OptIter:       10000,
	Neurons:       25,
	Regularizer:   nil,
	TwoStage:      false,
	MakeLinearFit: true,
}

var SALavalMedium10k = Case{
	Name:          "SALavalSmall",
	Dataset:       "laval_dns_sa",
	Features:      "nondim_source_irrotational",
	MulNet:        false,
	BoundKind:     QuantileBounds,
	BoundPoints:   40,
	OptIter:       10000,
	Neurons:       50,
	Regularizer:   nil,
	TwoStage:      false,
	MakeLinearFit: true,
}

var TurbKinSmall0k = Case{
	Name:          "TurbKinSmall10K",
	Dataset:       "laval_dns_crop",
	Features:      "nondim_turb_kin_source",
	MulNet:        false,
	BoundKind:     QuantileBounds,
	BoundPoints:   40,
	OptIter:       10000,
	Neurons:       25,
	Regularizer:   nil,
	TwoStage:      false,
	MakeLinearFit: true,
}

var TurbKinMedium10k = Case{
	Name:          "TurbKinSmall10K",
	Dataset:       "laval_dns_crop",
	Features:      "nondim_turb_kin_source",
	MulNet:        false,
	BoundKind:     QuantileBounds,
	BoundPoints:   60,
	OptIter:       10000,
	Neurons:       25,
	Regularizer:   nil,
	TwoStage:      false,
	MakeLinearFit: true,
}

var TurbSpecDissMedium10k = Case{
	Name:          "TurbSpecDiss10K",
	Dataset:       "laval_dns_crop",
	Features:      "nondim_turb_spec_diss_source",
	MulNet:        false,
	BoundKind:     QuantileBounds,
	BoundPoints:   60,
	OptIter:       10000,
	Neurons:       25,
	Regularizer:   nil,
	TwoStage:      false,
	MakeLinearFit: true,
}

var TurbSpecDissMedium10kNoLinear = Case{
	Name:          "TurbSpecDiss10K",
	Dataset:       "laval_dns_crop",
	Features:      "nondim_turb_spec_diss_source",
	MulNet:        false,
	BoundKind:     QuantileBounds,
	BoundPoints:   60,
	OptIter:       10000,
	Neurons:       25,
	Regularizer:   nil,
	TwoStage:      false,
	MakeLinearFit: false,
}

func main() {
	nCPU := runtime.NumCPU() - 2
	runtime.GOMAXPROCS(nCPU)
	rand.Seed(time.Now().UnixNano())

	run := TurbSpecDissMedium10kNoLinear

	boundKind := run.BoundKind
	nPointsPerBound := run.BoundPoints
	nOptIter := run.OptIter
	datasetStr := run.Dataset
	featureStr := run.Features
	nNeurons := run.Neurons
	optMethod := &optimize.BFGS{}
	regularizer := run.Regularizer
	twoStage := run.TwoStage
	useMulNet := run.MulNet

	hiddenLayerActivator := nnet.Tanh{}

	/********** Load the data ************/
	datasets, err := settings.GetDatasets(datasetStr, driver.Serial{})
	if err != nil {
		log.Fatal(err)
	}

	inputFeatures, outputFeatures, err := settings.GetFeatures(featureStr)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("inputFeatures = ", inputFeatures)
	fmt.Println("outputFeatures = ", outputFeatures)

	inputDataMat, outputDataMat, weights, err := ransuq.DenseLoadAll(datasets, inputFeatures, outputFeatures, nil, nil)

	if err != nil {
		log.Fatal(err)
	}

	if weights != nil {
		log.Fatal("not coded for weighted data")
	}

	inputData := inputDataMat.(*mat64.Dense)
	outputData := outputDataMat.(*mat64.Dense)

	nSamples, inputDim := inputData.Dims()
	_, outputDim := outputData.Dims()

	fmt.Println("Total nSamples =", nSamples)

	/********** Scale the data ************/
	var inputScaler scale.Scaler
	var outputScaler scale.Scaler

	if useMulNet {
		out := &mlalg.MulOutputScaler{}
		outputScaler = out

		inputScaler = &mlalg.MulInputScaler{
			Scaler:          &scale.Normal{},
			MulOutputScaler: out,
		}
	} else {
		inputScaler = &scale.Normal{}
		outputScaler = &scale.Normal{}
	}

	inputScaler.SetScale(inputData)
	outputScaler.SetScale(outputData)

	scale.ScaleData(inputScaler, inputData)
	scale.ScaleData(outputScaler, outputData)

	/********* Linear fit to the data ********/

	fmt.Println("Making linear fit")
	var linPred *mat64.Dense
	var coeffs *mat64.Dense
	if useMulNet {
		linPred, coeffs = mulFitPlane(inputData, outputData)
	} else {
		linPred, coeffs = fitPlane(inputData, outputData)
	}

	if !run.MakeLinearFit {
		r, c := linPred.Dims()
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				linPred.Set(i, j, 0)
			}
		}
		r, c = coeffs.Dims()
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				coeffs.Set(i, j, 0)
			}
		}
	}

	fmt.Println("plane coeffs = ", coeffs)

	plotPrediction(linPred.RawMatrix().Data, outputData.RawMatrix().Data, "linearpred.jpg")

	fmt.Println("linear prediction plotted")

	fmt.Println("norm pre", floats.Norm(outputData.RawMatrix().Data, 2))
	nonlinPred := &mat64.Dense{}

	nonlinPred.Sub(outputData, linPred)

	normPost := floats.Norm(nonlinPred.RawMatrix().Data, 2)
	fmt.Println("norm post is", normPost)

	_, nonLinOut := nonlinPred.Dims()
	if nonLinOut != outputDim {
		panic("wrong size")
	}

	bounds := findBounds(boundKind, inputData, nPointsPerBound)

	netIdxs, err := grid.Grid(inputData, bounds)
	if err != nil {
		log.Fatal(err)
	}

	nNetSamples := len(netIdxs)
	fmt.Println("nNetSamples = ", nNetSamples)

	netInputData := mat64.NewDense(nNetSamples, inputDim, nil)
	netOutputData := mat64.NewDense(nNetSamples, outputDim, nil)

	for i := 0; i < nNetSamples; i++ {
		row := netIdxs[i]
		for j := 0; j < inputDim; j++ {
			v := inputData.At(row, j)
			netInputData.Set(i, j, v)
		}
		for j := 0; j < outputDim; j++ {
			v := outputData.At(row, j)
			netOutputData.Set(i, j, v)
		}
	}

	nHiddenLayers := 2
	nNeuronsPerLayer := nNeurons    // I usually use more, but let's keep this example cheap
	finalActivator := nnet.Linear{} // doing regression, so use a linear activator in the last output
	algorithm, err := nnet.NewSimpleTrainer(inputDim, outputDim, nHiddenLayers, nNeuronsPerLayer, hiddenLayerActivator, finalActivator)
	if err != nil {
		log.Fatal(err)
	}

	//cacheFeatures := true                        // Have the algorithm use more memory to cache temporary results
	losser := loss.SquaredDistance{} // SquaredDistance loss function

	//batch := train.NewBatchGradBased(algorithm, cacheFeatures, netInputData, netOutputData, weights, losser, regularizer)
	//problem := OptValToFDf{batch}

	//	r, c := netInputData.Dims()

	problem := &train.GradOptimizable{
		Trainable: algorithm,
		Inputs:    netInputData,
		Outputs:   netOutputData,
		Weights:   nil,

		NumWorkers:  nCPU,
		Losser:      losser,
		Regularizer: regularizer,
	}

	settings := optimize.DefaultSettings()
	settings.FunctionEvals = nOptIter
	settings.GradientAbsTol = 1e-6

	// Set a random initial starting condition
	algorithm.RandomizeParameters()
	initLoc := algorithm.Parameters(nil)

	fmt.Println("starting optimize")
	fmt.Println("number of parameters to optimize is ", len(initLoc))

	problem.Init()

	if twoStage {
		// Get all of the indices of the final index
		// There are outputDim neurons in the last node
		var endIndices []int

		for i := 0; i < outputDim; i++ {
			// layers are zero indexed, so it's nHiddenLayers + 1 - 1
			newIdxs := algorithm.ParameterIdx(nHiddenLayers, i)
			endIndices = append(endIndices, newIdxs...)
		}

		/*
			for i := 0; i < nNeuronsPerLayer; i++ {
				newIdxs := algorithm.ParameterIdx(0, i)
				endIndices = append(endIndices, newIdxs...)
			}
		*/
		fmt.Println(endIndices)
		fulldim := algorithm.NumParameters()
		dim := algorithm.NumParameters() - len(endIndices)

		innerSettings := optimize.DefaultSettings()
		//innerSettings.Recorder = nil
		innerSettings.GradientAbsTol = 1e-10
		ts := partial.NewTwoStage(fulldim, endIndices, problem, innerSettings, nil)

		initTS := make([]float64, 0, dim)
		for i := 0; i < len(initLoc); i++ {
			var easy bool
			for j := 0; j < len(endIndices); j++ {
				if i == endIndices[j] {
					easy = true
					break
				}
			}
			if !easy {
				initTS = append(initTS, initLoc[i])
			}
		}
		result, err := optimize.Local(ts, initTS, settings, optMethod)
		if err != nil {
			log.Print(err)
		}
		fmt.Println(result.X)
	} else {
		result, err := optimize.Local(problem, initLoc, settings, optMethod)
		if err != nil {
			log.Print(err)
		}
		algorithm.SetParameters(result.X)
	}
	problem.Close()

	fmt.Println("done optimize")

	// Set the parameters of the net

	netPred := mat64.NewDense(nSamples, outputDim, nil)

	algorithm.PredictBatch(inputData, netPred)

	plotPrediction(netPred.RawMatrix().Data, outputData.RawMatrix().Data, "netpred.jpg")

	netTrainPred := mat64.NewDense(nNetSamples, outputDim, nil)
	netTrainReal := mat64.NewDense(nNetSamples, outputDim, nil)

	for i := 0; i < nNetSamples; i++ {
		row := netIdxs[i]
		for j := 0; j < outputDim; j++ {
			v := netPred.At(row, j)
			netTrainPred.Set(i, j, v)

			v = outputData.At(row, j)
			netTrainReal.Set(i, j, v)
		}
	}

	plotPrediction(netTrainPred.RawMatrix().Data, netTrainReal.RawMatrix().Data, "netpred_train.jpg")

	avgSqLossTrain := floats.Distance(netTrainPred.RawMatrix().Data, netTrainReal.RawMatrix().Data, 2) / math.Sqrt(float64(len(netTrainPred.RawMatrix().Data)))
	avgSqLossAll := floats.Distance(netPred.RawMatrix().Data, outputData.RawMatrix().Data, 2) / math.Sqrt(float64(len(outputData.RawMatrix().Data)))
	fmt.Println("Squared loss train points", avgSqLossTrain)
	fmt.Println("Squared loss all points", avgSqLossAll)

	// IF want to get back original, need to add back in linear prediction (to both) and unscale
}

/*
type OptValToFDf struct {
	multivariate.ObjGrader
}

func (o OptValToFDf) F(x []float64) float64 {
	g := make([]float64, len(x))
	return o.ObjGrad(x, g)
}

func (o OptValToFDf) FDf(x, g []float64) float64 {
	f := o.ObjGrad(x, g)
	return f
}
*/

func plotPrediction(predictedValues, trueValues []float64, plotName string) error {
	p, err := plot.New()
	if err != nil {
		return err
	}

	fmt.Println("len pred ", len(predictedValues))
	fmt.Println("len true ", len(trueValues))

	pts := make(plotter.XYs, len(predictedValues))
	for i := range pts {
		pts[i].X = trueValues[i]
		pts[i].Y = predictedValues[i]
	}
	p.X.Label.Text = "True"
	p.Y.Label.Text = "Pred"

	s, err := plotter.NewScatter(pts)
	if err != nil {
		return err
	}
	p.Add(s)

	f := plotter.NewFunction(func(x float64) float64 { return x })
	p.Add(f)
	err = p.Save(4, 4, plotName)
	if err != nil {
		return err
	}
	return nil
}

func fitPlane(inputData, outputData *mat64.Dense) (linPred, coeffs *mat64.Dense) {
	nSamples, inputDim := inputData.Dims()

	planedata := mat64.NewDense(nSamples, inputDim+1, nil)
	// Set the intercept term
	for i := 0; i < nSamples; i++ {
		planedata.Set(i, 0, 1)
	}
	dataView := &mat64.Dense{}
	dataView.View(planedata, 0, 1, nSamples, inputDim)
	dataView.Copy(inputData)

	coeffs = mat64.Solve(planedata, outputData)

	linPred = &mat64.Dense{}

	linPred.Mul(planedata, coeffs)
	return linPred, coeffs
}

func mulFitPlane(inputData, outputData *mat64.Dense) (linPred, coeffs *mat64.Dense) {

	nSamples, inputDim := inputData.Dims()
	// Create the function
	f := squaredLoss{
		inputData:  inputData,
		outputData: outputData,
	}

	initialLocation := make([]float64, inputDim)
	/*
		// Check the derivatives of f
		trueDeriv := make([]float64, inputDim)
		estDeriv := make([]float64, inputDim)
		//for i := range initialLocation {
		//	initialLocation[i] = rand.NormFloat64()
		//}
		fmt.Println("Finding true derivative")
		f.FDf(initialLocation, trueDeriv)
		fmt.Println("Done true derivative")

		for i := range initialLocation {
			if i != 0 {
				log.Fatal("Done first")
			}
			fofi := func(x float64) float64 {
				initialLocation[i] += x
				v := f.FDf(initialLocation, trueDeriv)
				initialLocation[i] -= x
				return v
			}
			estDeriv[i] = fd.Derivative(fofi, 0, fd.DefaultSettings())

			fmt.Println("estI  =", estDeriv[i])
			fmt.Println("true i ", trueDeriv[i])
		}
		if !floats.EqualApprox(estDeriv, trueDeriv, 1e-6) {
			fmt.Println("initial Location = ", initialLocation)
			fmt.Println("fd deriv = ", estDeriv)
			fmt.Println("true derin = ", trueDeriv)
			log.Fatal("deriv mismatch")
		}
	*/

	//initialLocation := make([]float64, inputDim+1)
	settings := optimize.DefaultSettings()
	settings.GradientAbsTol = 1e-14
	result, err := optimize.Local(f, initialLocation, settings, nil)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Done optimizing")

	coeffs = mat64.NewDense(inputDim, 1, result.X)

	// Now, do the linear prediction
	lincoeffs := mat64.NewDense(inputDim-1, 1, result.X[1:])

	linPred = &mat64.Dense{}
	data := &mat64.Dense{}
	data.View(inputData, 0, 1, nSamples, inputDim-1)
	linPred.Mul(data, lincoeffs)

	// add the linear term
	floats.AddConst(result.X[0], linPred.RawMatrix().Data)

	first := &mat64.Dense{}
	first.View(inputData, 0, 0, nSamples, 1)

	linPred.MulElem(linPred, first)

	return linPred, coeffs
}

type squaredLoss struct {
	inputData  *mat64.Dense
	outputData *mat64.Dense
}

func (s squaredLoss) F(params []float64) float64 {
	deriv := make([]float64, len(params))
	return s.FDf(params, deriv)
}

func (s squaredLoss) FDf(params []float64, deriv []float64) (loss float64) {
	// First, do the planar multiply
	nSamples, inputDim := s.inputData.Dims()
	if len(params) != inputDim {
		panic("param size mismatch")
	}

	planeCoeffs := mat64.NewDense(inputDim-1, 1, params[1:])

	preds := mat64.NewDense(nSamples, 1, nil)

	// Do the linear part
	nonMul := &mat64.Dense{}
	nonMul.View(s.inputData, 0, 1, nSamples, inputDim-1)
	preds.Mul(nonMul, planeCoeffs)

	// add the offset
	for i := 0; i < nSamples; i++ {
		v := preds.At(i, 0)
		preds.Set(i, 0, v+params[0])
	}

	for i := range deriv {
		deriv[i] = 0
	}

	var predZero float64

	predZero += params[0]

	for i := 1; i < inputDim; i++ {
		predZero += params[i] * s.inputData.At(0, i)
	}
	predZero *= s.inputData.At(0, 0)

	// Now, preds contains the true non-multiplied prediction
	for i := 0; i < nSamples; i++ {
		mul := s.inputData.At(i, 0)
		pred := preds.At(i, 0) * mul

		truth := s.outputData.At(i, 0)
		diff := pred - truth
		loss += diff * diff

		deriv[0] += 2 * diff * mul // for linear term
		for j := 1; j < inputDim; j++ {
			deriv[j] += 2 * diff * mul * s.inputData.At(i, j)
		}
	}

	nSamplesFloat := float64(nSamples)
	loss /= nSamplesFloat
	for i := range deriv {
		deriv[i] /= nSamplesFloat
	}
	//	fmt.Println(loss, floats.Norm(deriv, 2)/math.Sqrt(float64(len(deriv))))
	return loss
}

func findBounds(kind BoundKind, data *mat64.Dense, nPoints int) [][]float64 {
	rows, cols := data.Dims()
	bounds := make([][]float64, cols)
	for i := range bounds {
		bounds[i] = make([]float64, nPoints+1)
	}
	switch kind {
	case QuantileBounds:
		col := make([]float64, rows)
		quantiles := make([]float64, nPoints+1)
		floats.Span(quantiles, 0, 1)
		quantiles = quantiles[1 : len(quantiles)-1]
		for i := 0; i < cols; i++ {
			data.Col(col, i)
			sort.Float64s(col)
			//fmt.Println("i = ", i)
			//fmt.Println("first 10", col[:10])
			//fmt.Println("last 10", col[len(col)-10:])
			bounds[i][0] = math.Inf(-1)
			for j, q := range quantiles {
				bounds[i][j+1] = stat.Quantile(q, stat.Empirical, col, nil)
			}
			bounds[i][len(bounds[i])-1] = math.Inf(1)
		}
	}
	return bounds
}
