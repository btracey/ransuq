package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/btracey/myplot"
	"github.com/btracey/numcsv"
	"github.com/btracey/ransuq"
	"github.com/btracey/ransuq/mlalg"
	"github.com/btracey/turbulence/sa"
	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/plotutil"
	"github.com/gonum/plot/vg"
	"github.com/gonum/plot/vg/draw"

	"github.com/reggo/reggo/scale"
	"github.com/reggo/reggo/supervised/nnet"
)

var (
	// Here so that types get registered for unmarshaling
	_ = mlalg.MulInputScaler{}
	_ = nnet.NewSimpleTrainer
)

func findIdx(strs []string, str string) int {
	for i, v := range strs {
		if str == v {
			return i
		}
	}
	return -1
}

func main() {
	// Analyze nnet to see if it's reproducing the actual equation or if it's just
	// learning an alternate solution

	// Open neural network file
	//seed := time.Now().UnixNano()
	//var seed int64 = 1432235956179865706
	//var seed int64 = 1432237699134146365

	// For OmegaBar and Fw, 2 and 4 (no adaptive)
	// For Chi and Fw, 1 and 2 (with adaptive)
	// For OmegaBar and Source, 0 and 3 are interesting (no adaptive)
	// For Chi and Source, 0 and 1 (adaptive)
	// NuHatBar 0, 2 (adaptive)

	var seed int64 = 0
	rand.Seed(seed)
	fmt.Println(seed)

	/*
		mul := false
		blCase := true
		fileInputNames := []string{"Chi", "OmegaBar"}
		fileOutputNames := []string{"Fw", "IsInBL"}
		outputname := "fw"
		outputNameAxis := "Fw"
		sweepname := "OmegaBar"
		adaptiveBounds := false
	*/

	mul := true
	blCase := true
	fileInputNames := []string{"SourceNondimer", "Chi", "OmegaBar", "NuHatGradNormBar"}
	fileOutputNames := []string{"Residual_3", "IsInBL"}
	outputname := "source"
	outputNameAxis := "Source"
	sweepname := "NuHatGradNormBar"
	adaptiveBounds := true

	var algname string
	if mul {
		algname = "mul_net_2_50"
	} else {
		algname = "net_2_50"
	}
	savepath := filepath.Join("/", "Users", "brendan", "Documents", "mygo", "results",
		"ransuq", "multi_flatplate_bl", outputname, "none", algname, "10kiter")

	// Load it into a scale predictor
	sp, err := ransuq.LoadScalePredictor(savepath)
	if err != nil {
		log.Fatal(err)
	}

	// Load the data for that case

	_, ismulnet := sp.Predictor.(mlalg.MulPredictor)
	fmt.Println("Is MulNet? ", ismulnet)

	inputDim := sp.InputDim()
	outputDim := sp.OutputDim()
	if outputDim != 1 {
		panic("only coded for one output")
	}
	// Since it's a mul net, the first dimension is actually irrelevant.
	//baseInput := make([]float64, inputDim)
	//baseInput[0] = 1
	/*
		chi := 8.993142585003163e-05
		omegaBar := 9.330000000000001e-01
		nuhat := 9.341626279711292e-02
		mul :=
	*/

	// Load in data to check the truth
	/*
		csv := filepath.Join("/", "Users", "brendan", "Documents", "mygo", "results",
			"ransuq", "multi_flatplate_bl", outputname, "none", algname, "10kiter", "comparison",
			"Flatplate_Re_4e_06_mlBlOnly", "su2run", "turb_flatplate_sol.dat")
	*/
	/*
		csv := filepath.Join("/", "Users", "brendan", "Documents", "mygo", "data",
			"ransuq", "flatplate", "med_scitech", "Flatplate_Re_5e_06", "turb_flatplate_sol.dat")
	*/

	// List of training dataset CSVs
	csvs := []string{
		filepath.Join("/", "Users", "brendan", "Documents", "mygo", "data", "ransuq", "flatplate", "med_scitech", "Flatplate_Re_3e_06", "turb_flatplate_sol.dat"),
		filepath.Join("/", "Users", "brendan", "Documents", "mygo", "data", "ransuq", "flatplate", "med_scitech", "Flatplate_Re_5e_06", "turb_flatplate_sol.dat"),
		filepath.Join("/", "Users", "brendan", "Documents", "mygo", "data", "ransuq", "flatplate", "med_scitech", "Flatplate_Re_7e_06", "turb_flatplate_sol.dat"),
	}

	// Load in the data.
	inputs, outputs := loadCsvs(csvs, fileInputNames, fileOutputNames, blCase)

	// Check that the true output matches
	for i, test := range inputs {
		trueValue := getTrueMulNet(test, outputname, ismulnet)
		compValue := outputs[i]
		//predValue, _ := sp.Predict(test, nil)
		if math.Abs(trueValue-compValue) > 1e-14 {
			log.Fatal("true value mismatch")
		}
	}

	scaledData := mat64.NewDense(len(inputs), len(inputs[0]), nil)
	for i := range inputs {
		for j := range inputs[i] {
			scaledData.Set(i, j, inputs[i][j])
		}
	}

	scale.ScaleData(sp.InputScaler, scaledData)
	nScaledData, _ := scaledData.Dims()

	// Now, hack it to remove the scaled constant
	if ismulnet {
		for i := 0; i < nScaledData; i++ {
			scaledData.Set(i, 0, 1)
		}
	}

	dim := findIdx(fileInputNames, sweepname)

	// Make the baseInput a random data slice
	randInt := rand.Intn(nScaledData)
	fmt.Println("randInt = ", randInt)
	baseInput := scaledData.Row(nil, randInt)
	fmt.Println("baseInput = ", baseInput)
	//baseInput := make([]float64, inputDim)
	if ismulnet {
		baseInput[0] = 1
	}

	eps := 0.1
	closeData := findCloseData(scaledData, baseInput, eps, dim, ismulnet)
	nClose, _ := closeData.Dims()
	fmt.Println("nclose = ", nClose)

	// Base input is in scaled terms
	var min, max float64
	if adaptiveBounds {
		// Find the min and max from the data points
		row := closeData.Col(nil, dim)
		min = floats.Min(row)
		max = floats.Max(row)
		fmt.Println("min, max", min, max)
	} else {
		min = -1.0
		max = 4.0
	}

	nPredict := 10000
	locs := make([]float64, nPredict)
	floats.Span(locs, min, max)
	input := make([]float64, inputDim)
	unscaleLoc := make([]float64, nPredict)
	predict := make([]float64, nPredict)
	truth := make([]float64, nPredict)

	// Do it in transformed space?
	// How close does the point need to be

	closeDataUnscaled := mat64.NewDense(0, 0, nil)
	closeDataUnscaled.Clone(closeData)
	scale.UnscaleData(sp.InputScaler, closeDataUnscaled)

	for i := range locs {
		copy(input, baseInput)
		input[dim] = locs[i]
		sp.InputScaler.Unscale(input)
		output, err := sp.Predict(input, nil)
		if err != nil {
			log.Fatal(err)
		}
		unscaleLoc[i] = input[dim]
		predict[i] = output[0]
		truth[i] = getTrueMulNet(input, outputname, ismulnet)
	}

	//fmt.Println("predict =", predict)
	//fmt.Println("true =", truth)

	maxIdx := floats.MaxIdx(truth)
	minIdx := floats.MinIdx(truth)
	fmt.Println("Max truth = ", floats.Max(truth), "X =", unscaleLoc[maxIdx])
	fmt.Println("Min truth = ", floats.Min(truth), "X = ", unscaleLoc[minIdx])

	fmt.Println("Max pred = ", floats.Max(predict))
	fmt.Println("Min pred = ", floats.Min(predict))

	p, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}
	line, err := plotter.NewLine(myplot.VecXY{X: unscaleLoc, Y: truth})
	if err != nil {
		log.Fatal(err)
	}
	line.Color = plotutil.SoftColors[0]
	line.Dashes = plotutil.Dashes(0)
	line2, err := plotter.NewLine(myplot.VecXY{X: unscaleLoc, Y: predict})
	if err != nil {
		log.Fatal(err)
	}
	line2.Color = plotutil.SoftColors[1]
	line2.Dashes = plotutil.Dashes(1)
	// Add the close data
	closeX := make([]float64, 0)
	closeY := make([]float64, 0)
	for i := 0; i < nClose; i++ {
		x := closeDataUnscaled.At(i, dim)
		y := getTrueMulNet(closeDataUnscaled.Row(nil, i), outputname, ismulnet)
		closeX = append(closeX, x)
		closeY = append(closeY, y)
	}

	fmt.Println("Max data = ", floats.Max(closeY))
	fmt.Println("Min data = ", floats.Min(closeY))

	scatter, err := plotter.NewScatter(myplot.VecXY{X: closeX, Y: closeY})
	if err != nil {
		log.Fatal(err)
	}
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	scatter.GlyphStyle.Radius = 1
	p.X.Label.Text = fileInputNames[dim]
	p.Y.Label.Text = outputNameAxis
	p.Add(scatter)
	p.Add(line)
	p.Add(line2)
	p.Legend.Add("True", line)
	p.Legend.Add("Predicted", line2)
	p.Legend.Add("Data", scatter)
	//plotutil.AddLines(p, "True", line, "Pred", line2)
	//plotutil.AddScatters(p, "Data", scatter)
	p.Legend.Top = false
	p.Legend.Left = false
	p.X.Min = unscaleLoc[0]
	p.X.Max = unscaleLoc[len(unscaleLoc)-1]
	//p.Add(scatter)
	p.Save(4*vg.Inch, 3*vg.Inch, "func_sweep.pdf")

	//fmt.Println("Line is")
	//fmt.Println(line)

}

// getTrueMulNet gets the true value taking into account the fact that it's a mul net
func getTrueMulNet(loc []float64, outputname string, ismulnet bool) float64 {
	var truth float64
	if ismulnet {
		truth = loc[0] * getTrue(loc[1:], outputname)
	} else {
		truth = getTrue(loc, outputname)
	}
	return truth
}

func getTrue(loc []float64, outputname string) float64 {
	switch outputname {
	case "source":
		chi := loc[0]
		omegaBar := loc[1]
		nBar := loc[2]

		fv1 := sa.Fv1(chi)
		fv2 := sa.Fv2(chi, fv1)

		// Need r
		shatbar := ShatBar(omegaBar, chi, fv2)
		r := R(shatbar, chi)
		g := sa.G(r)
		fw := sa.Fw(g)
		sp := sa.CB1 * (chi / (chi + 1)) * shatbar
		sd := (chi / (chi + 1)) * (chi / (chi + 1)) * sa.CW1 * fw
		scp := sa.CB2 / sa.Sigma * nBar
		sbar := sp - sd + scp
		return sbar
	case "fw":
		chi := loc[0]
		omegaBar := loc[1]

		fv1 := sa.Fv1(chi)
		fv2 := sa.Fv2(chi, fv1)
		shatbar := ShatBar(omegaBar, chi, fv2)
		r := R(shatbar, chi)
		g := sa.G(r)
		fw := sa.Fw(g)
		return fw
	}
	panic("unknown output name")
}

func ShatBar(omegaBar, chi, fv2 float64) float64 {
	return omegaBar + chi/(sa.K2*(chi+1))*fv2
}

func R(shatbar, chi float64) float64 {
	if shatbar < 10e-10 {
		shatbar = 10e-10
	}

	r := chi / (shatbar * sa.K2 * (1 + chi))
	if r > 10 {
		r = 10
	}
	return r
}

func findCloseData(data *mat64.Dense, baseLoc []float64, eps float64, dim int, ismulnet bool) (closeData *mat64.Dense) {
	r, c := data.Dims()
	close := make([][]float64, 0)
	for i := 0; i < r; i++ {
		row := data.Row(nil, i)
		isClose := true
		for j := range row {
			if ismulnet && j == 0 {
				continue
			}
			if j != dim {
				if math.Abs(data.At(i, j)-baseLoc[j]) > eps {
					isClose = false
				}
			}
		}
		if isClose {
			close = append(close, row)
		}
	}
	closeData = mat64.NewDense(len(close), c, nil)
	for i := 0; i < len(close); i++ {
		closeData.SetRow(i, close[i])
	}
	return closeData
}

func loadCsvs(csvs []string, inputNames []string, outputNames []string, blCase bool) (inputs [][]float64, outputs []float64) {
	inputs = make([][]float64, 0)
	outputs = make([]float64, 0)
	for _, csv := range csvs {
		f2, err := os.Open(csv)
		if err != nil {
			log.Fatal(err)
		}
		defer f2.Close()

		r := numcsv.NewReader(f2)
		r.Comma = "\t"
		headings, err := r.ReadHeading()
		if err != nil {
			log.Fatal(err)
		}
		data, err := r.ReadAll()
		if err != nil {
			log.Fatal(err)
		}

		inputDim := len(inputNames)
		outputDim := len(outputNames)
		// Select the correct indices of the data slice. Return the input ones
		idxs := make([]int, inputDim)
		for i, s := range inputNames {
			idxs[i] = findIdx(headings, s)
		}

		outputIdxs := make([]int, outputDim)
		for i, s := range outputNames {
			outputIdxs[i] = findIdx(headings, s)
		}
		nData, _ := data.Dims()
		blIdx := findIdx(headings, outputNames[1])
		for i := 0; i < nData; i++ {
			// Remove non-BL data if it's a BL case
			if blCase && data.At(i, blIdx) != 1 {
				continue
			}
			// Remove data at the wall
			wallDist := data.At(i, findIdx(headings, "WallDist"))
			if wallDist < 1e-10 {
				continue
			}
			row := make([]float64, inputDim)
			for j, v := range idxs {
				row[j] = data.At(i, v)
			}
			inputs = append(inputs, row)
			outputs = append(outputs, data.At(i, outputIdxs[0]))
		}
	}
	return inputs, outputs
}
