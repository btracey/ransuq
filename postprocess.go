package ransuq

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"code.google.com/p/plotinum/plot"
	"code.google.com/p/plotinum/plotter"
	"code.google.com/p/plotinum/vg"

	"github.com/gonum/matrix/mat64"
	"github.com/reggo/reggo/common"

	myplot "github.com/btracey/plot"
)

func pltName(path, subpath, name string) string {
	return filepath.Join(path, subpath, name)
}

// path is the path to where the files should be stored
func makeComparisons(inputData, outputData common.RowMatrix, sp ScalePredictor, inputNames []string, outputNames []string, path string) error {

	nSamples, inputDim := inputData.Dims()
	nOutputs := len(outputNames)

	directExists := make([]bool, nOutputs)
	indirectExists := make([]bool, nOutputs)

	directEnd := "pred_vs_truth.pdf"
	indirectEnd := "err_vs_truth.pdf"
	conPltEnd := "err_scat.pdf"

	//TODO: Add contour bit up here

	for i := 0; i < nOutputs; i++ {

		direct := pltName(path, outputNames[i], directEnd)
		indirect := pltName(path, outputNames[i], indirectEnd)

		// See if the direct comparison is already there
		_, err := os.Open(direct)
		if err == nil {
			directExists[i] = true
		}
		// See if the indirect comparison plot is already there
		_, err = os.Open(indirect)
		if err == nil {
			indirectExists[i] = true
		}
	}

	allMade := true
	// See if all of the plots have already been made
	for _, b := range directExists {
		if !b {
			allMade = false
		}
	}
	for _, b := range indirectExists {
		if !b {
			allMade = false
		}
	}
	if allMade {
		fmt.Println("Postprocessing plots generated")
		//return nil
	}

	// Find the predictions at the data
	pred := mat64.NewDense(nSamples, nOutputs, nil)
	input := make([]float64, inputDim)
	for i := 0; i < nSamples; i++ {
		output := pred.RowView(i)
		inputData.Row(input, i)

		_, err := sp.Predict(input, output)
		if err != nil {
			return err
		}
	}

	// Now, make the plots comparing the predictions
	pts := make(plotter.XYs, nSamples)
	errPts := make(plotter.XYs, nSamples)
	for j := 0; j < nOutputs; j++ {
		name := outputNames[j]
		direct := pltName(path, outputNames[j], directEnd)
		indirect := pltName(path, outputNames[j], indirectEnd)
		contour := pltName(path, outputNames[j], conPltEnd)

		err := os.MkdirAll(filepath.Join(path, outputNames[j]), 0700)
		if err != nil {
			return err
		}
		for i := 0; i < nSamples; i++ {
			pts[i].X = outputData.At(i, j)
			pts[i].Y = pred.At(i, j)

			errPts[i].X = outputData.At(i, j)
			errPts[i].Y = pred.At(i, j) - outputData.At(i, j)
		}

		plt, err := plot.New()
		if err != nil {
			return err
		}

		scatter, err := plotter.NewScatter(pts)
		if err != nil {
			return err
		}

		equalLine := plotter.NewFunction(func(x float64) float64 { return x })
		plt.Add(equalLine, scatter)
		//plt.Title.Text = title
		plt.X.Label.Text = "True value of " + name
		plt.Y.Label.Text = "Predicted value of " + name
		plt.Title.Text = "Prediction vs. Truth for " + name

		err = plt.Save(4, 4, direct)
		fmt.Println("filename", direct)
		fmt.Println(err)
		if err != nil {
			fmt.Println("returning")
			return err
		}
		fmt.Println("saved 2 plot")

		errPlt, err := plot.New()
		if err != nil {
			fmt.Println("er plt nil: ")
			fmt.Println("err plt: ", err)
			return err
		}
		fmt.Println("gen err plt")

		errScatter, err := plotter.NewScatter(errPts)
		if err != nil {
			return err
		}

		fmt.Println("created err scatt")

		errPlt.Add(errScatter)

		fmt.Println("added err scatt")
		//plt.Title.Text = title
		plt.X.Label.Text = "True value of " + name
		plt.Y.Label.Text = "Difference in predicted value of " + name
		plt.Title.Text = "Prediction Error for " + name

		// TODO: Save the data used to make the plot

		err = errPlt.Save(4, 4, indirect)
		if err != nil {
			fmt.Println("Error saving errPLts")
			return err
		}
		fmt.Println("saved error plot")

		if inputDim == 2 {
			// Make a contour plot if the data is 2-D
			conPts := make(plotter.XYZs, nSamples)
			for i := 0; i < nSamples; i++ {
				conPts[i].X = inputData.At(i, 0)
				conPts[i].Y = inputData.At(i, 1)
				conPts[i].Z = errPts[i].Y
			}

			conPlt, err := plot.New()
			if err != nil {
				return err
			}

			conPlt.X.Label.Text = inputNames[0]
			conPlt.Y.Label.Text = inputNames[1]

			scat, err := myplot.NewColoredScatter(conPts)
			if err != nil {
				return err
			}

			scat.GlyphStyle.Radius = vg.Centimeters(0.01)
			scat.GlyphStyle.Shape = plot.CircleGlyph{}
			conPlt.Add(scat)
			err = conPlt.Save(4, 4, contour)
			if err != nil {
				return err
			}

		}
	}
	return nil
}

func postprocess(sp ScalePredictor, settings *Settings) error {

	wg := &sync.WaitGroup{}

	trainingErr := make(ErrorList, len(settings.TrainingData))

	basepath := filepath.Join(settings.Savepath, "postprocess")

	// Plot the training comparisons
	for i := 0; i < len(settings.TrainingData); i++ {
		wg.Add(1)
		go func(i int) {
			fmt.Println("Starting training postprocess ", i)
			defer wg.Done()
			inputs, outputs, _, err := LoadData(settings.TrainingData[i], DenseLoad, settings.InputFeatures, settings.OutputFeatures, nil)
			if err != nil {
				trainingErr[i] = err
				return
			}
			savepath := filepath.Join(basepath, settings.TrainingData[i].ID())
			trainingErr[i] = makeComparisons(inputs, outputs, sp, settings.InputFeatures, settings.OutputFeatures, savepath)
			fmt.Println("training err ", trainingErr[i])
		}(i)
	}

	// This is here so there is no race condition if the same dataset is in training
	// and testing
	wg.Wait()

	testingErr := make(ErrorList, len(settings.TestingData))

	// Plot the testing comparisons
	for i := 0; i < len(settings.TestingData); i++ {
		wg.Add(1)
		go func(i int) {
			fmt.Println("Starting testing postprocess ", i)
			defer wg.Done()
			inputs, outputs, _, err := LoadData(settings.TestingData[i], DenseLoad, settings.InputFeatures, settings.OutputFeatures, nil)
			if err != nil {
				testingErr[i] = err
				return
			}
			savepath := filepath.Join(basepath, settings.TestingData[i].ID())
			testingErr[i] = makeComparisons(inputs, outputs, sp, settings.InputFeatures, settings.OutputFeatures, savepath)
		}(i)
	}
	wg.Wait()

	noTestErr := trainingErr.AllNil()
	noTrainErr := testingErr.AllNil()

	if noTestErr && noTrainErr {
		return nil
	}
	return PostprocessError{
		Training: trainingErr,
		Testing:  testingErr,
	}
}
