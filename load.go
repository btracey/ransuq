package ransuq

import (
	"errors"
	"fmt"
	"sync"

	"github.com/gonum/matrix/mat64"

	"github.com/reggo/reggo/common"
)

// UniqueFeatures combines the needed features fon inputs, outputs, and weights
// to get a single list of features where no feature is there twice. Unique features
// also returns slices of integers to say where in the feature list
func uniqueFeatures(input, output, weight []string) (features []string, inputInds, outputInds, weightInds []int) {
	m := make(map[string]int)
	features = make([]string, 0, len(input)+len(output)+len(weight))
	inputInds, features = addStrings(input, m, features)
	outputInds, features = addStrings(output, m, features)
	weightInds, features = addStrings(weight, m, features)
	return
}

// addStrings adds strings to the map and features when necessary and returns
// a list mapping string to index in features
func addStrings(strs []string, m map[string]int, features []string) ([]int, []string) {
	inds := make([]int, len(strs))
	for i := range inds {
		inds[i] = -1 // Just to make sure it works
	}
	for i, str := range strs {
		ind, exist := m[str]
		if exist {
			inds[i] = ind
		} else {
			l := len(m)
			m[str] = l
			inds[i] = l
			features = append(features, str)
		}
	}
	return inds, features
}

type LoadStyle int

const (
	// Loads the data into dense matrices. May be memory intensive, but should be
	// faster on analysis methods
	DenseLoad LoadStyle = iota
)

var UnknownLoadStyle = errors.New("unknown load style")

type LoadError []error

func (l LoadError) Error() string {
	var str string
	for i := range l {
		if l[i] != nil {
			str += fmt.Sprintf("error loading ", i, ": ", l[i].Error())
		}
	}
	return str
}

// LoadData returns the data in the dataset.
func LoadData(dataset Dataset, loadStyle LoadStyle, inputFeatures, outputFeatures, weightFeatures []string) (
	inputs, outputs, weights common.RowMatrix, err error) {
	// This currently uses Dense matrices, but in
	// the future this could have options added to take better advantage of the Matrix
	// interface, especially for lange data.

	switch loadStyle {
	default:
		err = UnknownLoadStyle
		return
	case DenseLoad:
		features, inputInds, outputInds, weightInds := uniqueFeatures(inputFeatures, outputFeatures, weightFeatures)
		return loadDenseData(dataset, features, inputInds, outputInds, weightInds)

	}
}

func loadDenseData(dataset Dataset, features []string, inputInds, outputInds, weightInds []int) (
	inputData, outputData, weightData common.RowMatrix, err error) {

	data, err := dataset.Load(features)
	if err != nil {
		return
	}

	_, nDim := data.Dims()
	if nDim != len(features) {
		err = errors.New("unexpected number of columns")
	}

	if nDim != len(features) {
		return
	}

	// Break out the data into dense forms
	inputData = denseUnpack(data, inputInds)
	outputData = denseUnpack(data, outputInds)
	weightData = denseUnpack(data, weightInds)

	return inputData, outputData, weightData, nil
}

func denseUnpack(data common.RowMatrix, inds []int) common.RowMatrix {

	if len(inds) == 0 {
		return nil
	}

	nSamples, _ := data.Dims()

	newdata := mat64.NewDense(nSamples, len(inds), nil)

	for i := 0; i < nSamples; i++ {
		for j := range inds {
			newdata.Set(i, j, data.At(i, inds[j]))
		}
	}

	return newdata
}

func denseLoadAll(datasets []Dataset, inputFeatures, outputFeatures, weightFeatures []string, weightFunc func([]float64) float64) (
	inputData, outputData common.RowMatrix, weights []float64, err error) {
	// TODO: This is really memory intensive at the moment. Need to make this better
	// (too much copying between dense matrices)
	// Probably easiest done by not calling denseUnpack and instead coming from the
	// feature matrix

	inputMats := make([]common.RowMatrix, len(datasets))
	outputMats := make([]common.RowMatrix, len(datasets))
	weightMats := make([]common.RowMatrix, len(datasets))

	features, inputInds, outputInds, weightInds := uniqueFeatures(inputFeatures, outputFeatures, weightFeatures)

	errs := make([]error, len(datasets))
	wg := &sync.WaitGroup{}
	for i := range datasets {
		wg.Add(1)
		go func(i int) {
			inputMats[i], outputMats[i], weightMats[i], errs[i] = loadDenseData(datasets[i], features, inputInds, outputInds, weightInds)
			wg.Done()
		}(i)
	}
	wg.Wait()

	errs = reduceError(errs)
	if errs != nil {
		return nil, nil, nil, LoadError(errs)
	}

	// start ind
	var totalNSamples int
	startInds := make([]int, len(datasets))
	for i := range datasets {
		startInds[i] = totalNSamples
		nSamples, _ := inputMats[i].Dims()
		totalNSamples += nSamples
	}
	inputs := mat64.NewDense(totalNSamples, len(inputFeatures), nil)
	outputs := mat64.NewDense(totalNSamples, len(outputFeatures), nil)
	if weightFunc != nil {
		weights = make([]float64, totalNSamples)
	}

	// For each dataset, copy the data into inputs and outputs, and evaluate the
	// weight function

	for i := range datasets {
		wg.Add(1)
		go func(i int) {
			// Copy the data to the right place
			mat := inputMats[i]
			nSamples, nCols := mat.Dims()
			mat3 := outputMats[i]
			_, nOutCols := mat3.Dims()
			startInd := startInds[i]
			for i := 0; i < nSamples; i++ {
				for j := 0; j < nCols; j++ {
					inputs.Set(i+startInd, j, mat.At(i, j))
				}
				for j := 0; j < nOutCols; j++ {
					outputs.Set(i+startInd, j, mat3.At(i, j))
				}
			}

			// Evaluate the weight func on all the rows
			mat2 := weightMats[i]
			weightData := make([]float64, len(weightFeatures))
			for i := 0; i < nSamples; i++ {
				for j := range weightData {
					weightData[i] = mat2.At(i, j)
				}
				if weightFunc != nil {
					weights[i+startInd] = weightFunc(weightData)
				}
			}
			wg.Done()
		}(i)
	}
	wg.Wait()

	return inputs, outputs, weights, nil

}

// reduceError returns the slice of errors if any of the errors are non-nil
// and nil if all of the errors are nil
func reduceError(errs []error) []error {
	for i := range errs {
		if errs[i] != nil {
			return errs
		}
	}
	return nil
}

// LoadTrainingData returns
func LoadTrainingData(datasets []Dataset, loadStyle LoadStyle, inputFeatures, outputFeatures, weightFeatures []string, weightFunc func([]float64) float64) (
	inputs, outputs common.RowMatrix, weights []float64, err error) {

	if len(weightFeatures) != 0 && weightFunc == nil {
		err = errors.New("non-zero weights but nil weightFunc")
		return
	}

	if weightFunc != nil && len(weightFeatures) == 0 {
		err = errors.New("non-nil weightfunc but zero weight features")
		return
	}

	switch loadStyle {
	default:
		err = UnknownLoadStyle
		return
	case DenseLoad:
		return denseLoadAll(datasets, inputFeatures, outputFeatures, weightFeatures, weightFunc)
	}
}
