// Compute a piecewise linear model of the data

package main

import (
	"fmt"
	"log"
	"myplot"
	"runtime"
	"strconv"
	"sync"

	"code.google.com/p/plotinum/plot"
	"code.google.com/p/plotinum/plotter"

	"github.com/btracey/ransuq/grid"
	"github.com/btracey/ransuq/quickload"
	"github.com/gonum/blas/goblas"
	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
)

type Case struct {
	Name        string
	Dataset     string
	Features    string
	BoundPoints []int
}

var SALavalMedium = Case{
	Name:        "SALavalMedium",
	Dataset:     "laval_dns_crop",
	Features:    "nondim_source_irrotational",
	BoundPoints: []int{2, 5, 7, 10, 20, 50, 70, 100, 200, 500, 700, 1000},
}

var SATurbKinNondimSource = Case{
	Name:        "SATurbKinMedium",
	Dataset:     "laval_dns_crop",
	Features:    "nondim_turb_kin_source",
	BoundPoints: []int{2, 5, 7, 10, 20, 50, 70, 100, 200, 500, 700, 1000},
}

var SATurbSpecDissNondimSource = Case{
	Name:        "SATurbSpecDissMedium",
	Dataset:     "laval_dns_crop",
	Features:    "nondim_turb_spec_diss_source",
	BoundPoints: []int{2, 5, 7, 10, 20, 50, 70, 100, 200, 500, 700, 1000},
}

func init() {
	mat64.Register(goblas.Blas{})
	runtime.GOMAXPROCS(runtime.NumCPU() - 2)
}

func main() {
	run := SATurbSpecDissNondimSource

	inputData, outputData, weights, err := quickload.Load(run.Dataset, run.Features)
	if err != nil {
		log.Fatal(err)
	}
	if weights != nil {
		log.Fatal("not coded for weighted data")
	}
	/*
		extraFeatures := []string{"XLoc", "YLoc"}
		extraData, _, err := quickload.LoadExtra(run.Dataset, extraFeatures)
		if err != nil {
			log.Fatal(err)
		}
		XLoc := findStringLocation(extraFeatures, "XLoc")
		YLoc := findStringLocation(extraFeatures, "YLoc")
	*/
	nSamples, inputDim := inputData.Dims()
	outSamples, outputDim := outputData.Dims()
	if nSamples != outSamples {
		panic("nSample mismatch")
	}
	fmt.Println("nSamples =", nSamples)
	preds := make([]*mat64.Dense, len(run.BoundPoints))
	// Make a local linear fit
	predsLinear := make([]*mat64.Dense, len(run.BoundPoints))

	//mux := &sync.Mutex{}
	var wg sync.WaitGroup
	for boundIdx, nBounds := range run.BoundPoints {
		wg.Add(1)
		go func(boundIdx, nBounds int) {
			defer wg.Done()
			// map of grid cells. String is converted box value. []int are the list
			// of indices that are in that box.
			m := make(map[string][]int)
			bounds := grid.FindBounds(grid.QuantileBounds, inputData, nBounds)
			box := make([]int, inputDim)
			for i := 0; i < nSamples; i++ {
				for j := 0; j < inputDim; j++ {
					box[j] = floats.Within(bounds[j], inputData.At(i, j))
				}
				str := grid.BoxToString(box)
				m[str] = append(m[str], i)
			}
			// Check that all the bounds worked
			var check int
			for _, val := range m {
				check += len(val)
			}
			if check != nSamples {
				fmt.Println(check, nSamples)
				panic("mismatch")
			}
			fmt.Println("nBounds =", nBounds, "nUnique =", len(m))
			pred := mat64.NewDense(nSamples, outputDim, nil)
			preds[boundIdx] = pred
			predLinear := mat64.NewDense(nSamples, outputDim, nil)
			predsLinear[boundIdx] = predLinear
			for _, val := range m {
				locs := mat64.NewDense(len(val), inputDim+1, nil)
				outputs := mat64.NewDense(len(val), outputDim, nil)
				for i, idx := range val {
					for j := 0; j < inputDim; j++ {
						locs.Set(i, j, inputData.At(idx, j))
					}
					locs.Set(i, inputDim, 1) // offset term
					for j := 0; j < outputDim; j++ {
						out := outputData.At(idx, j)
						outputs.Set(i, j, out)
					}
				}
				// Make the prediction the mean of the block
				for j := 0; j < outputDim; j++ {
					col := outputs.Col(nil, j)
					mean := stat.Mean(col, nil)
					for _, idx := range val {
						pred.Set(idx, j, mean)
					}
				}
				// Need a check about the number of samples == 1
				// Make the prediction the value at a plane in thet block
				coeffs := mat64.Solve(locs, outputs)
				linout := &mat64.Dense{}
				linout.Mul(locs, coeffs)
				for i, idx := range val {
					for j := 0; j < outputDim; j++ {
						predsLinear[boundIdx].Set(idx, j, linout.At(i, j))
					}
				}
			}
		}(boundIdx, nBounds)
	}
	wg.Wait()

	// should add a mean all and std all and then multiply by the std so avg point is
	// O(1)
	meanAll := make([]float64, outputDim)
	stdAll := make([]float64, outputDim)
	for j := 0; j < outputDim; j++ {
		c := outputData.Col(nil, j)
		meanAll[j] = stat.Mean(c, nil)
		stdAll[j] = stat.StdDev(c, meanAll[j], nil)
		fmt.Println("mean =", meanAll[j], "std =", stdAll[j])
	}

	for i, pred := range preds {
		for j := 0; j < outputDim; j++ {
			truth := outputData.Col(nil, j)
			pred := pred.Col(nil, j)
			sqerr := floats.Distance(truth, pred, 2) * stdAll[j] * stdAll[j]
			fmt.Println("nBounds =", run.BoundPoints[i], "output =", j, "avg dist =", sqerr)
			boundStr := strconv.Itoa(run.BoundPoints[i])
			outputStr := strconv.Itoa(j)
			err := makePlot(truth, pred, "pred_"+boundStr+"_"+outputStr+".jpg")
			if err != nil {
				log.Fatal(err)
			}
			predLinear := predsLinear[i].Col(nil, j)
			err = makePlot(truth, predLinear, "predlinear_"+boundStr+"_"+outputStr+".jpg")
			if err != nil {
				log.Fatal(err)
			}
		}
	}

	// Need to plot predictions, plot the clusters,do some sort of historgram over
	// number of points per cell
	// Plot nUnique vs. Squared error
}

func makePlot(truth, pred []float64, name string) error {
	xy := myplot.NewXY(truth, pred)
	scatter, err := plotter.NewScatter(xy)
	if err != nil {
		return err
	}
	p, err := plot.New()
	if err != nil {
		return err
	}
	f := plotter.NewFunction(func(x float64) float64 { return x })
	p.Add(scatter)
	p.Add(f)
	p.Save(8, 8, name)
	return nil
}

func findStringLocation(s []string, str string) int {
	for i, v := range s {
		if v == str {
			return i
		}
	}
	return -1
}
