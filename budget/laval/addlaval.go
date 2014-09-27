package main

import (
	"errors"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"

	"github.com/btracey/diff/scattered"
	"github.com/btracey/fluid/fluid2d"
	"github.com/btracey/matrix/twod"
	"github.com/btracey/numcsv"
	"github.com/btracey/ransuq/dataloader"
	"github.com/btracey/ransuq/datawrapper"
	"github.com/btracey/turbulence/sa"
	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
)

var features = []string{"idx_x", "idx_y", "grid_x", "grid_yx", "DUDX", "DUDY", "DVDX", "DVDY", "TauUU", "TauUV", "TauVV", "Nu", "WallDistance", "UVel", "VVel"}

var (
	IdxX         = findStringLocation(features, "idx_x")
	IdxY         = findStringLocation(features, "idx_y")
	XLoc         = findStringLocation(features, "grid_x")
	YLoc         = findStringLocation(features, "grid_yx")
	DUDX         = findStringLocation(features, "DUDX")
	DUDY         = findStringLocation(features, "DUDY")
	DVDX         = findStringLocation(features, "DVDX")
	DVDY         = findStringLocation(features, "DVDY")
	Nu           = findStringLocation(features, "Nu")
	UU           = findStringLocation(features, "TauUU")
	UV           = findStringLocation(features, "TauUV")
	VV           = findStringLocation(features, "TauVV")
	WallDistance = findStringLocation(features, "WallDistance")
	UVel         = findStringLocation(features, "UVel")
	VVel         = findStringLocation(features, "VVel")
)

var newFeatures = []string{"StrainRateMag", "VorticityMag", "VorticityMagNondim", "TotalVelGradNorm",
	"VelGradDet", "VelVortOverNorm", "NuTilde", "Chi", "SourceNondimer", "NuGradAngle",
	"DNuHatDX", "DNuHatDY", "NuHatGradMag", "NuHatGradMagBar", "Source", "SourceNondimerUNorm",
	"NondimSourceUNorm", "VelDetOverNorm", "NuVelGradNormRatio",
}

var (
	StrainRateMag       = findStringLocation(newFeatures, "StrainRateMag")
	VorticityMag        = findStringLocation(newFeatures, "VorticityMag")
	VorticityMagNondim  = findStringLocation(newFeatures, "VorticityMagNondim")
	NuTilde             = findStringLocation(newFeatures, "NuTilde")
	Chi                 = findStringLocation(newFeatures, "Chi")
	SourceNondimer      = findStringLocation(newFeatures, "SourceNondimer")
	DNuHatDX            = findStringLocation(newFeatures, "DNuHatDX")
	DNuHatDY            = findStringLocation(newFeatures, "DNuHatDY")
	Source              = findStringLocation(newFeatures, "Source")
	NuHatGradMag        = findStringLocation(newFeatures, "NuHatGradMag")
	NuHatGradMagBar     = findStringLocation(newFeatures, "NuHatGradMagBar")
	TotalVelGradMag     = findStringLocation(newFeatures, "TotalVelGradNorm")
	VelGradDet          = findStringLocation(newFeatures, "VelGradDet")
	VelVortOverNorm     = findStringLocation(newFeatures, "VelVortOverNorm")
	VelDetOverNorm      = findStringLocation(newFeatures, "VelDetOverNorm")
	NuGradAngle         = findStringLocation(newFeatures, "NuGradAngle")
	SourceNondimerUNorm = findStringLocation(newFeatures, "SourceNondimerUNorm")
	NondimSourceUNorm   = findStringLocation(newFeatures, "NondimSourceUNorm")
	NuVelGradNormRatio  = findStringLocation(newFeatures, "NuVelGradNormRatio")
)

// features that shouldn't be appended but are convenient to store
var extraFeatures = []string{"NuSum", "NuSumTimesDX", "NuSumTimesDY", "LHS"}

var (
	NuSum        = findStringLocation(extraFeatures, "NuSum")
	NuSumTimesDX = findStringLocation(extraFeatures, "NuSumTimesDX")
	NuSumTimesDY = findStringLocation(extraFeatures, "NuSumTimesDY")
	LHS          = findStringLocation(extraFeatures, "LHS")
)

const NuTildeEpsilon = 1e-10

type gridSorter [][]float64

func (g gridSorter) Len() int {
	return len(g)
}

func (g gridSorter) Less(i, j int) bool {
	// Sort first by y
	if g[i][IdxY] < g[j][IdxY] {
		return true
	}
	if g[i][IdxY] > g[j][IdxY] {
		return false
	}
	// Sort by x next
	return g[i][IdxX] < g[j][IdxX]
}

func (g gridSorter) Swap(i, j int) {
	g[i], g[j] = g[j], g[i]
}

func main() {
	nDim := 2
	nX := 2304
	//nY := 385

	nProcs := runtime.NumCPU() - 1
	runtime.GOMAXPROCS(nProcs)
	dataset := "/Users/brendan/Documents/mygo/data/ransuq/laval/laval_csv.dat"

	// Need to load the data
	data := loadData(dataset, features)

	g := gridSorter(data)
	sort.Sort(g)
	data = g

	// Find all the neighbors of each point. Assumes structured grid.

	neighbors := make([][]int, len(data)) // Slice of neighbor coordinates
	xStencil := 1
	yStencil := 1

	// Data is sorted, first by x then y.
	// Ones next to equal x values are neighbors
	// We know it's on a grid, so adding the number of x points should give the same y index
	for i, pt := range data {
		//fmt.Println(i)
		for j := -xStencil; j <= xStencil; j++ {
			if j == 0 {
				continue
			}
			newIdx := i + j
			if newIdx < 0 || newIdx >= len(data) {
				continue
			}
			if pt[IdxY] == data[newIdx][IdxY] {
				//fmt.Println("Point ", pt[IdxX], pt[IdxY], " neighbor ", data[newIdx][IdxX], data[newIdx][IdxY])
				neighbors[i] = append(neighbors[i], newIdx)
			}
		}
		for j := -yStencil; j <= yStencil; j++ {
			if j == 0 {
				continue
			}
			newIdx := i + j*nX
			if newIdx < 0 || newIdx >= len(data) {
				continue
			}
			/*
				fmt.Println(" new IDx = ", newIdx)
				fmt.Println("pt idxx", pt[IdxX], " new idxx", data[newIdx][IdxX])
				fmt.Println("pt idxy", pt[IdxY], " new idxy", data[newIdx][IdxY])
			*/
			if pt[IdxX] == data[newIdx][IdxX] {
				//fmt.Println("Point ", pt[IdxX], pt[IdxY], " neighbor ", data[newIdx][IdxX], data[newIdx][IdxY])
				neighbors[i] = append(neighbors[i], newIdx)
			}
		}
	}

	// make sure there are no obvious bugs with the neighbors
	for i := range neighbors {
		if len(neighbors[i]) == 0 {
			log.Fatal("point has no neighbors", len(neighbors))
		}
		// quick check, not proof of correctness
		if len(neighbors[i]) > (xStencil+1)*(yStencil+1) {
			log.Fatal("point has too many neighbors", len(neighbors))
		}
	}

	// Add weights for all the neighbors
	planePoints := make([][]*scattered.PointMV, len(neighbors))
	for i := range planePoints {
		planePoints[i] = make([]*scattered.PointMV, len(neighbors[i]))

		thisX := data[i][XLoc]
		thisY := data[i][YLoc]
		thisLoc := []float64{thisX, thisY}
		for j := range planePoints[i] {
			neighborID := neighbors[i][j]
			xLoc := data[neighborID][XLoc]
			yLoc := data[neighborID][YLoc]
			loc := []float64{xLoc, yLoc}

			planePoints[i][j] = &scattered.PointMV{
				Location: loc,
				Weight:   invDist(thisLoc, loc),
			}
		}
	}

	newData := make([][]float64, len(data))
	extraData := make([][]float64, len(data))
	for i, pt := range data {
		extraData[i] = make([]float64, len(extraFeatures))
		newpt := make([]float64, len(newFeatures))
		newData[i] = newpt

		velGrad := fluid2d.VelGrad{}
		(&velGrad).Set(pt[DUDX], pt[DUDY], pt[DVDX], pt[DVDY])

		sym, skewsym := velGrad.Split()
		strainRate := fluid2d.StrainRate{sym}
		vorticity := fluid2d.Vorticity{skewsym}
		detVel := velGrad.Det()
		normVel := velGrad.Norm(twod.Frobenius2)

		strainRateMag := strainRate.Norm(twod.Frobenius2)
		vorticityMag := vorticity.Norm(twod.Frobenius2)
		newpt[StrainRateMag] = strainRateMag
		newpt[VorticityMag] = vorticityMag
		newpt[VelGradDet] = detVel
		newpt[TotalVelGradMag] = normVel
		newpt[VelVortOverNorm] = vorticityMag / normVel
		newpt[VelDetOverNorm] = detVel / normVel

		//if math.Abs(normVel-vorticityMag-strainRateMag) > 1e-6 {
		if math.Abs(normVel*normVel-vorticityMag*vorticityMag-strainRateMag*strainRateMag) > 1e-8 {
			fmt.Println("dudx", velGrad.DUDX(), " dudy ", velGrad.DVDY(), " sum = ", velGrad.DUDX()+velGrad.DVDY())
			//fmt.Println("dudy", velGrad.DUDY(), pt[DUDY])
			//fmt.Println("dvdx", velGrad.DVDX(), pt[DVDX])
			fmt.Println("dvdy", velGrad.DVDY(), pt[DVDY])
			fmt.Println("norm vel = ", normVel)
			fmt.Println("vorticityMag = ", vorticityMag)
			fmt.Println("strainRateMag = ", strainRateMag)
			fmt.Println("diff = ", normVel*normVel-vorticityMag*vorticityMag-strainRateMag*strainRateMag)
			log.Fatal("Norms don't agree")
		}

		tau := fluid2d.ReynoldsStress{}
		(&tau).Set(pt[UU], pt[UV], pt[VV])

		nuTilde := fluid2d.TurbKinVisc(tau, strainRate, NuTildeEpsilon)

		newpt[NuTilde] = float64(nuTilde)
		newpt[Chi] = newpt[NuTilde] / pt[Nu]

		walldist := pt[WallDistance]
		nusum := pt[Nu] + newpt[NuTilde]

		extraData[i][NuSum] = nusum

		sourceNondimer := (nusum * nusum) / (walldist * walldist)
		omegaNondimer := nusum / (walldist * walldist)
		newpt[VorticityMagNondim] = float64(strainRateMag) / (omegaNondimer)
		newpt[SourceNondimer] = sourceNondimer

	}

	// Need to compute the source term. This is
	// u * d Nuhat/dx - d/dx (nu + nuhat )(dNuhat/dx)

	nancount := 0

	// First, compute the derivatives of nutilde
	deriv := make([]float64, nDim)
	loc := make([]float64, nDim)
	for i, pt := range data {
		newpt := newData[i]
		setPointValues(data, NuTilde, planePoints[i], neighbors[i])
		//for i := range data {
		loc[0] = data[i][XLoc]
		loc[1] = data[i][YLoc]
		intercept := scattered.Intercept{
			Force: true,
			Value: data[i][NuTilde],
		}
		_ = intercept
		scattered.Plane(loc, planePoints[i], intercept, deriv)

		dNuHatDX := deriv[0]
		dNuHatDY := deriv[1]
		newpt[DNuHatDX] = dNuHatDX
		newpt[DNuHatDY] = dNuHatDY

		vel := []float64{pt[UVel], pt[VVel]}
		velDotDNuHat := floats.Dot(deriv, vel)

		nuHatGradMag := dNuHatDX*dNuHatDX + dNuHatDY*dNuHatDY
		velMag := vel[0]*vel[0] + vel[1]*vel[1]
		angle := velDotDNuHat / (velMag * nuHatGradMag)
		if velMag == 0 {
			angle = 0
		}

		if math.IsNaN(angle) {
			nancount++
			fmt.Println("wall dist ", pt[WallDistance])
			fmt.Println("velMag = ", velMag)
			fmt.Println("nuHatGradMag", nuHatGradMag)

		}

		extraData[i][LHS] = velDotDNuHat
		extraData[i][NuSumTimesDX] = extraData[i][NuSum] * dNuHatDX
		extraData[i][NuSumTimesDY] = extraData[i][NuSum] * dNuHatDY
		newData[i][NuHatGradMag] = nuHatGradMag
		newData[i][NuHatGradMagBar] = newData[i][NuHatGradMag] / newData[i][SourceNondimer]

		newData[i][NuGradAngle] = angle
		sourceNondimerUNorm := extraData[i][NuSum] * newData[i][TotalVelGradMag]
		newData[i][SourceNondimerUNorm] = sourceNondimerUNorm
		newData[i][NuVelGradNormRatio] = (nuHatGradMag * nuHatGradMag) / ((extraData[i][NuSum]) * newData[i][TotalVelGradMag])
	}
	// Now, compute the source term

	fmt.Println("Total nan is ", nancount)

	for i, pt := range data {
		newpt := newData[i]
		_ = pt
		loc[0] = data[i][XLoc]
		loc[1] = data[i][YLoc]

		// for the source term, need dnuhat dx

		setPointValues(extraData, NuSumTimesDX, planePoints[i], neighbors[i])
		intercept := scattered.Intercept{
			Force: true,
			Value: extraData[i][NuSumTimesDX],
		}
		scattered.Plane(loc, planePoints[i], intercept, deriv)

		nuSumDXX := deriv[0]
		nuSumDXY := deriv[1]

		setPointValues(extraData, NuSumTimesDY, planePoints[i], neighbors[i])
		intercept = scattered.Intercept{
			Force: true,
			Value: extraData[i][NuSumTimesDY],
		}
		scattered.Plane(loc, planePoints[i], intercept, deriv)

		nuSumDYX := deriv[0]
		nuSumDYY := deriv[1]

		// rhs is the trace of the derivative divided by sigma
		rhs := (nuSumDXX + nuSumDYY) / sa.Sigma

		source := extraData[i][LHS] - rhs

		if math.Abs(source) > 10000 {
			fmt.Println("crazy source")
			fmt.Println("Nu grad ", nuSumDXX, nuSumDXY, nuSumDYX, nuSumDYY)
			fmt.Println("Vel grad = ", pt[DUDX], pt[DUDY], pt[DVDX], pt[DVDY])
			fmt.Println("Wall dist = ", pt[WallDistance])
			os.Exit(1)
		}
		newpt[Source] = source

		newData[i][NondimSourceUNorm] = source / newData[i][SourceNondimerUNorm]
	}

	// do a quick check to make sure all the fields of newData got set
	for j := 0; j < len(newData[0]); j++ {
		fmt.Println(j)
		var hasNonzero bool
		for i, pt := range newData {
			if pt[j] != 0 {
				hasNonzero = true
			}
			if math.IsInf(pt[j], 0) {
				fmt.Println("i = ", i, "j = ", j, " is inf, name is ", newFeatures[j])
				fmt.Println(data[i][WallDistance])
			}

			if math.IsNaN(pt[j]) {
				fmt.Println("i = ", i, "j = ", j, " is nan, name is ", newFeatures[j])
			}
		}
		if !hasNonzero {
			fmt.Println("j is ", j, " name is ", newFeatures[j])
			log.Fatal("Idx ", j, " not set in newData")
		}
	}

	// Append the data

	fmt.Println("Saving file")
	f, err := os.Open(dataset)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	r := numcsv.NewReader(f)

	ext := filepath.Ext(dataset)
	pre := dataset[:len(dataset)-len(ext)]
	newFilename := pre + "_computed" + ext

	f2, err := os.Create(newFilename)
	if err != nil {
		log.Fatal(err)
	}
	defer f2.Close()

	w := numcsv.NewWriter(f2)
	w.FloatFmt = 'g'

	err = appendCSV(r, newFeatures, newData, w)
	if err != nil {
		log.Fatal(err)
	}
}

func appendCSV(r *numcsv.Reader, newHeadings []string, newData [][]float64, w *numcsv.Writer) error {
	// Read all of the headings
	headings, err := r.ReadHeading()
	if err != nil {
		return err
	}
	data, err := r.ReadAll()
	if err != nil {
		return err
	}

	rows, cols := data.Dims()

	if len(newData) != rows {
		return errors.New("nData mismatch")
	}
	dim := len(newData[0])
	for _, pt := range newData {
		if len(pt) != dim {
			return errors.New("dim mismatch")
		}
	}
	if len(newHeadings) != dim {
		fmt.Println(newHeadings)
		fmt.Println(dim)
		return errors.New("Heading length mismatch")
	}

	// Append the extra data
	m := mat64.NewDense(rows, cols+dim, nil)
	m.Copy(data)
	for i := 0; i < rows; i++ {
		for j := cols; j < cols+dim; j++ {
			m.Set(i, j, newData[i][j-cols])
		}
	}
	headings = append(headings, newHeadings...)

	return w.WriteAll(headings, m)

}

func loadData(dataset string, features []string) [][]float64 {

	set := &dataloader.Dataset{
		Filename: dataset,
		Format:   datawrapper.Laval,
	}

	allData, err := dataloader.Load(features, []*dataloader.Dataset{set})
	if err != nil {
		log.Fatal("error loading laval data: " + err.Error())
	}
	data := allData[0] // just limit it to this dataset
	return data
}

func findStringLocation(s []string, str string) int {
	for i, v := range s {
		if v == str {
			return i
		}
	}
	return -1
}

func invDist(x, y []float64) float64 {
	dist := floats.Distance(x, y, 2)
	return math.Sqrt(1.0 / (dist * dist))
	//return 1.0 / (dist * dist)
}

func setPointValues(data [][]float64, idx int, points []*scattered.PointMV, neighbors []int) {
	for i := range points {
		neighborIdx := neighbors[i]
		points[i].Value = data[neighborIdx][idx]
	}
}
