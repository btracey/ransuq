package main

import (
	"fmt"
	"log"
	"math"
	"myplot"

	"code.google.com/p/plotinum/plot"
	"code.google.com/p/plotinum/plotter"

	"github.com/btracey/ransuq"
	"github.com/btracey/ransuq/settings"
	"github.com/btracey/su2tools/driver"
	"github.com/gonum/matrix/mat64"
)

func main() {
	datasetStr := "laval_dns"

	/*
		nX := 2304
		nY := 385
	*/

	/********** Load the data ************/
	datasets, err := settings.GetDatasets(datasetStr, driver.Serial{})
	if err != nil {
		log.Fatal(err)
	}

	inputFeatures := []string{"Chi", "NuHatGradMag", "SourceNondimerUNorm",
		"LogSourceNondimerUNorm", "VelVortOverNorm", "VelDetOverNorm", "NuHatGradMagUNorm",
		"VelNormOverNorm", "idx_x", "idx_y", "XLoc", "YLoc",
	}
	outputFeatures := []string{"NondimSourceUNorm", "Source"}

	IdxX := findStringLocation(inputFeatures, "idx_x")
	IdxY := findStringLocation(inputFeatures, "idx_y")
	XLoc := findStringLocation(inputFeatures, "XLoc")
	YLoc := findStringLocation(inputFeatures, "YLoc")

	inputDataMat, outputDataMat, weights, err := ransuq.DenseLoadAll(datasets, inputFeatures, outputFeatures, nil, nil)
	if err != nil {
		log.Fatal("error in dense load all" + err.Error())
	}

	if weights != nil {
		log.Fatal("not coded for weighted data")
	}

	inputData := inputDataMat.(*mat64.Dense)
	outputData := outputDataMat.(*mat64.Dense)

	nSamples, inputDim := inputData.Dims()
	_, outputDim := outputData.Dims()

	fmt.Println("nSamples = ", nSamples)

	// Get the locations
	xIdxs := make([]float64, nSamples)
	yIdxs := make([]float64, nSamples)
	xLocs := make([]float64, nSamples)
	yLocs := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		xIdxs[i] = inputData.At(i, IdxX)
		yIdxs[i] = inputData.At(i, IdxY)
		xLocs[i] = inputData.At(i, XLoc)
		yLocs[i] = inputData.At(i, YLoc)
	}

	// First, make histograms of all the data
	for j := 0; j < inputDim; j++ {
		data := make([]float64, nSamples)
		for i := 0; i < nSamples; i++ {
			data[i] = inputData.At(i, j)
			if j == 0 && math.Abs(data[i]) > 250 {
				fmt.Println(data[i])
			}
		}
		makeHistogram(data, inputFeatures[j]+"_hist.jpg")
		makeContour(xIdxs, yIdxs, data, inputFeatures[j]+"_contour.jpg")
		makeContour(xLocs, yLocs, data, inputFeatures[j]+"_contour2.jpg")
	}
	for j := 0; j < outputDim; j++ {
		data := make([]float64, nSamples)
		for i := 0; i < nSamples; i++ {
			data[i] = outputData.At(i, j)
		}
		makeHistogram(data, outputFeatures[j]+"_hist.jpg")
		makeContour(xIdxs, yIdxs, data, outputFeatures[j]+"_contour.jpg")
		makeContour(xLocs, yLocs, data, outputFeatures[j]+"_contour2.jpg")
	}

	// plot remaining grid locations
	pts := make(plotter.XYs, nSamples)
	for i := range pts {
		pts[i].X = xIdxs[i]
		pts[i].Y = yIdxs[i]
	}
	scatter, err := plotter.NewScatter(pts)
	if err != nil {
		log.Fatal(err)
	}
	p, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}
	p.Add(scatter)
	err = p.Save(8, 8, "keptidx.jpg")
	if err != nil {
		log.Fatal(err)
	}

}

func makeHistogram(data []float64, name string) {
	p, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}
	h, err := plotter.NewHist(plotter.Values(data), 1000)
	if err != nil {
		log.Fatal(err)
	}
	p.Add(h)
	err = p.Save(4, 4, name)
	if err != nil {
		log.Fatal(err)
	}
}

func makeContour(x, y, z []float64, name string) {
	pts := make(plotter.XYZs, len(x))
	for i := range pts {
		pts[i].X = x[i]
		pts[i].Y = y[i]
		pts[i].Z = z[i]
	}
	contour, err := myplot.NewColoredScatter(pts)
	if err != nil {
		log.Fatal(err)
	}
	contour.GlyphStyle.Radius = 1
	contour.GlyphStyle.Shape = plot.CircleGlyph{}
	p, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}
	p.Add(contour)
	err = p.Save(8, 8, name)
	if err != nil {
		log.Fatal(err)
	}
}

func findStringLocation(s []string, str string) int {
	for i, v := range s {
		if v == str {
			return i
		}
	}
	return -1
}
