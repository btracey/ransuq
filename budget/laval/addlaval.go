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
	"github.com/btracey/fluid"
	"github.com/btracey/fluid/fluid2d"
	"github.com/btracey/matrix/twod"
	"github.com/btracey/numcsv"
	"github.com/btracey/ransuq/dataloader"
	"github.com/btracey/ransuq/datawrapper"
	"github.com/btracey/turbulence/sa"
	"github.com/btracey/turbulence/sst"
	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/unit"
)

const (
	Rho     = 1
	Dx      = 3.346608355059288e-05
	NormEps = 1e-5
)

var features = []string{"idx_x", "idx_y", "grid_x", "grid_yx", "DUDX", "DUDY",
	"DVDX", "DVDY", "TauUU", "TauUV", "TauVV", "Nu", "WallDistance", "UVel", "VVel",
	"DissUU", "DissUV", "DissVV", "Pressure", "DPDX", "DPDY"}

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
	DissUU       = findStringLocation(features, "DissUU")
	DissUV       = findStringLocation(features, "DissUV")
	DissVV       = findStringLocation(features, "DissVV")
	Pressure     = findStringLocation(features, "Pressure")
	DPDX         = findStringLocation(features, "DPDX")
	DPDY         = findStringLocation(features, "DPDY")
)

var newFeatures = []string{
	"Rho", "StrainRateMag", "VorticityMag", "VorticityMagNondim", "NuTilde", "Chi",
	"DNuHatDX", "DNuHatDY", "NuHatGradMag", "NuHatGradMagBar", "NuHatGradMagUNorm", "NuGradAngle",
	"Source", "SourceNondimer", "SourceNondimer2", "SourceNondimerUNorm", "LogSourceNondimerUNorm",
	"VelGradDet", "VelVortOverNorm", "VelNormOverNorm", "TotalVelGradNorm", "VelDetOverNorm", "VelNormOverSpecDiss",
	"NondimSource", "NondimSource2", "NondimSourceUNorm",
	"TurbKinEnergy", "DTurbKinEnergyDX", "DTurbKinEnergyDY",
	"TurbDissipation", "TurbSpecificDissipation", "DTurbSpecificDissipationDX", "DTurbSpecificDissipationDY",
	"TurbKinEnergySourceBudget", "TurbSpecificDissipationSourceBudget",
	"TurbKinEnergySourceNondimer", "NondimTurbKinEnergySource", "TurbSpecificDissipationSourceNondimer",
	"NondimTurbSpecificDissipationSource",
}

// what features of the neighbor should be included in the feature set.
var neighborFeatures = []string{
	"DeltaXLoc", "DeltaYLoc", "DUDX", "DUDY", "DVDX", "DVDY", "UVel", "VVel", "Pressure", "DPDX", "DPDY",
}

var thisLocationFeatures = []string{
	"idx_x", "idx_y", "grid_x", "grid_yx", "NuTilde", "DNuHatDX", "DNuHatDY", "DUDX", "DUDY", "DVDX", "DVDY", "UVel", "VVel", "Pressure", "DPDX", "DPDY", "Source",
}

var (
	RhoIdx             = findStringLocation(newFeatures, "Rho")
	StrainRateMag      = findStringLocation(newFeatures, "StrainRateMag")
	VorticityMag       = findStringLocation(newFeatures, "VorticityMag")
	VorticityMagNondim = findStringLocation(newFeatures, "VorticityMagNondim")
	NuTilde            = findStringLocation(newFeatures, "NuTilde")
	Chi                = findStringLocation(newFeatures, "Chi")

	DNuHatDX          = findStringLocation(newFeatures, "DNuHatDX")
	DNuHatDY          = findStringLocation(newFeatures, "DNuHatDY")
	NuHatGradMag      = findStringLocation(newFeatures, "NuHatGradMag")
	NuHatGradMagBar   = findStringLocation(newFeatures, "NuHatGradMagBar")
	NuHatGradMagUNorm = findStringLocation(newFeatures, "NuHatGradMagUNorm")
	NuGradAngle       = findStringLocation(newFeatures, "NuGradAngle")

	Source                 = findStringLocation(newFeatures, "Source")
	SourceNondimer         = findStringLocation(newFeatures, "SourceNondimer")
	SourceNondimer2        = findStringLocation(newFeatures, "SourceNondimer2")
	SourceNondimerUNorm    = findStringLocation(newFeatures, "SourceNondimerUNorm")
	LogSourceNondimerUNorm = findStringLocation(newFeatures, "LogSourceNondimerUNorm")
	NondimSource           = findStringLocation(newFeatures, "NondimSource")
	NondimSource2          = findStringLocation(newFeatures, "NondimSource2")
	NondimSourceUNorm      = findStringLocation(newFeatures, "NondimSourceUNorm")

	TotalVelGradNorm    = findStringLocation(newFeatures, "TotalVelGradNorm")
	VelGradDet          = findStringLocation(newFeatures, "VelGradDet")
	VelVortOverNorm     = findStringLocation(newFeatures, "VelVortOverNorm")
	VelDetOverNorm      = findStringLocation(newFeatures, "VelDetOverNorm")
	VelNormOverNorm     = findStringLocation(newFeatures, "VelNormOverNorm")
	VelNormOverSpecDiss = findStringLocation(newFeatures, "VelNormOverSpecDiss")

	TurbKinEnergy    = findStringLocation(newFeatures, "TurbKinEnergy")
	DTurbKinEnergyDX = findStringLocation(newFeatures, "DTurbKinEnergyDX")
	DTurbKinEnergyDY = findStringLocation(newFeatures, "DTurbKinEnergyDY")

	TurbDissipation            = findStringLocation(newFeatures, "TurbDissipation")
	TurbSpecificDissipation    = findStringLocation(newFeatures, "TurbSpecificDissipation")
	DTurbSpecificDissipationDX = findStringLocation(newFeatures, "DTurbSpecificDissipationDX")
	DTurbSpecificDissipationDY = findStringLocation(newFeatures, "DTurbSpecificDissipationDY")

	TurbKinEnergySourceBudget   = findStringLocation(newFeatures, "TurbKinEnergySourceBudget")
	TurbKinEnergySourceNondimer = findStringLocation(newFeatures, "TurbKinEnergySourceNondimer")
	NondimTurbKinEnergySource   = findStringLocation(newFeatures, "NondimTurbKinEnergySource")

	TurbSpecificDissipationSourceBudget   = findStringLocation(newFeatures, "TurbSpecificDissipationSourceBudget")
	TurbSpecificDissipationSourceNondimer = findStringLocation(newFeatures, "TurbSpecificDissipationSourceNondimer")
	NondimTurbSpecificDissipationSource   = findStringLocation(newFeatures, "NondimTurbSpecificDissipationSource")
)

// features that shouldn't be appended but are convenient to store
var extraFeatures = []string{"NuSum", "NuSumTimesDX", "NuSumTimesDY", "LHS", "KDiffSstDX",
	"KDiffSstDY", "OmegaDiffSstDX", "OmegaDiffSstDY"}

var (
	NuSum          = findStringLocation(extraFeatures, "NuSum")
	NuSumTimesDX   = findStringLocation(extraFeatures, "NuSumTimesDX")
	NuSumTimesDY   = findStringLocation(extraFeatures, "NuSumTimesDY")
	KDiffSstDX     = findStringLocation(extraFeatures, "KDiffSstDX")
	KDiffSstDY     = findStringLocation(extraFeatures, "KDiffSstDY")
	OmegaDiffSstDX = findStringLocation(extraFeatures, "OmegaDiffSstDX")
	OmegaDiffSstDY = findStringLocation(extraFeatures, "OmegaDiffSstDY")
	LHS            = findStringLocation(extraFeatures, "LHS")
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

// This function reads in the laval data, computes budgets and other interesting
// quantities, and writes out a new data file.
func main() {
	nDim := 2
	nX := 2304
	nY := 385
	_ = nY

	nProcs := runtime.NumCPU() - 2
	runtime.GOMAXPROCS(nProcs)
	dataset := "/Users/brendan/Documents/mygo/data/ransuq/laval/laval_csv.dat"

	// Need to load the data
	data := loadData(dataset, features)

	g := gridSorter(data)
	sort.Sort(g)
	data = g

	// Find all the neighbors of each point. Assumes structured grid. gridSorter
	// makes sure this is true
	neighbors := make([][]int, len(data)) // Slice of neighbor coordinates
	xStencil := 2
	yStencil := 2

	// Data is sorted, first by x then y.
	// Ones next to equal x values are neighbors
	// We know it's on a grid, so adding the number of x points should give
	// the same y index
	//for i, pt := range data {
	for loc, pt := range data {
		// Add all of the points in the mini-box set by xStencil and yStencil.
		// Use all of the points in the box as a hope to get a more regularized
		// gradient box which will smooth out second derivatives.
		for i := -xStencil; i <= xStencil; i++ {
			for j := -yStencil; j <= yStencil; j++ {
				// The current location is not a neighbor of the current location
				if i == 0 && j == 0 {
					continue
				}
				// Check that the new index will be in bounds. Remember that the
				// grid indices are 1-indexed and not zero-indexed.
				currX := int(pt[IdxX])
				if currX+i < 1 || currX+i > nX {
					continue
				}
				currY := int(pt[IdxY])
				if currY+j < 1 || currY+j > nY {
					continue
				}

				// Verify that the new indicies match the expected
				newIdx := loc + i + j*nX
				newX := int(data[newIdx][IdxX])
				newY := int(data[newIdx][IdxY])

				if newX != currX+i {
					panic("x idx mismatch")
				}
				if newY != currY+j {
					panic("y idx mismatch")
				}
				neighbors[loc] = append(neighbors[loc], newIdx)
			}
		}
	}

	// make sure there are no obvious bugs with the neighbors
	for i := range neighbors {
		if len(neighbors[i]) == 0 {
			log.Fatal("point has no neighbors", len(neighbors))
		}
		// quick check, not proof of correctness
		if len(neighbors[i]) > ((2*xStencil+1)*(2*yStencil+1) - 1) {
			log.Fatalf("point %v has %v neighbors", i, len(neighbors[i]))
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

	// First, modify the values of epsilon to multiply by Nu
	for _, pt := range data {
		pt[DissUU] *= pt[Nu]
		pt[DissVV] *= pt[Nu]
		pt[DissUV] *= pt[Nu]
	}

	newData := make([][]float64, len(data))
	extraData := make([][]float64, len(data))
	for i, pt := range data {
		extraData[i] = make([]float64, len(extraFeatures))
		newpt := make([]float64, len(newFeatures))
		newData[i] = newpt

		velGrad := fluid2d.VelGrad{}
		(&velGrad).SetAll(pt[DUDX], pt[DUDY], pt[DVDX], pt[DVDY])

		//fmt.Println("velgrad", velGrad)

		sym, skewsym := velGrad.Split()
		strainRate := fluid2d.StrainRate{sym}
		vorticity := fluid2d.Vorticity{skewsym}

		//fmt.Println("sym", sym, "skewsym", skewsym)

		detVel := velGrad.Det()
		normVel := velGrad.Norm(twod.Frobenius2)

		strainRateMag := strainRate.Norm(twod.Frobenius2)
		vorticityMag := vorticity.Norm(twod.Frobenius2)
		//fmt.Println("strainmag", strainRateMag, "vortmag", vorticityMag)
		newpt[StrainRateMag] = strainRateMag
		newpt[VorticityMag] = vorticityMag
		newpt[VelGradDet] = detVel
		newpt[TotalVelGradNorm] = normVel
		normVelEps := normVel + NormEps
		newpt[VelVortOverNorm] = vorticityMag / normVel
		newpt[VelDetOverNorm] = detVel / normVel
		newpt[VelNormOverNorm] = normVel / normVelEps

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
		(&tau).SetAll(pt[UU], pt[UV], pt[VV])

		nuTilde := fluid2d.TurbKinVisc(tau, strainRate, NuTildeEpsilon)

		if i == 37871 || i == 37870 {
			fmt.Println()
			fmt.Println("i =", i)
			fmt.Println("tau = ", tau)
			fmt.Println("strain rate = ", strainRate)
			fmt.Println("nut =", nuTilde)
		}

		//fmt.Println(tau, strainRate, nuTilde)
		//		fmt.Println("nuTilde = ", nuTilde)

		newpt[NuTilde] = float64(nuTilde)
		newpt[Chi] = newpt[NuTilde] / pt[Nu]

		walldist := pt[WallDistance]
		nusum := pt[Nu] + newpt[NuTilde]

		extraData[i][NuSum] = nusum

		sourceNondimer := (nusum * nusum) / (walldist * walldist)
		sourceNondimer2 := (pt[Nu] * pt[Nu]) / (walldist * walldist)
		omegaNondimer := nusum / (walldist * walldist)
		newpt[VorticityMagNondim] = float64(strainRateMag) / (omegaNondimer)
		newpt[SourceNondimer] = sourceNondimer
		newpt[SourceNondimer2] = sourceNondimer2

		newpt[TurbKinEnergy] = 0.5 * (pt[UU]*pt[UU] + pt[VV]*pt[VV])
		newpt[TurbDissipation] = pt[DissUU] + pt[DissVV]

		//newpt[TurbSpecificDissipation] = newpt[TurbDissipation] / (sst.BetaStar * newpt[TurbKinEnergy])
		newpt[TurbSpecificDissipation] = newpt[TurbDissipation] / newpt[TurbKinEnergy]

		newpt[VelNormOverSpecDiss] = newpt[TotalVelGradNorm] / newpt[TurbSpecificDissipation]
		// See hack below at at the wall

		newpt[RhoIdx] = Rho

		newpt[TurbKinEnergySourceNondimer] = Rho * newpt[TurbSpecificDissipation] * newpt[TurbKinEnergy]
		newpt[TurbSpecificDissipationSourceNondimer] = Rho * newpt[TurbSpecificDissipation] * newpt[TurbSpecificDissipation]
	}
	//os.Exit(1)

	for i, pt := range data {
		// Need to hack at the wall
		if pt[WallDistance] == 0 {

			w := 10 * 6 * pt[Nu] / (sst.Beta1)
			newData[i][TurbSpecificDissipation] = w / (Dx * Dx)

			/*
				// Set to be the value one off the wall
				if pt[IdxY] != 1 && pt[IdxY] != float64(nY) {
					panic("weird 0")
				}
				// Find the index
				nextX := int(pt[IdxX])
				var nextIdx int
				if pt[IdxY] == 1 {
					nextIdx = i + nX
				} else {
					nextIdx = i - nX
				}
				if int(data[nextIdx][IdxX]) != nextX {
					fmt.Println("next x ", data[nextIdx][IdxX])
					panic("wrong x matching")
				}
				omega := newData[nextIdx][TurbSpecificDissipation]
				newData[i][TurbSpecificDissipation] = omega
			*/
		}
	}

	// Need to compute the source term. This is
	// u * d Nuhat/dx - d/dx (nu + nuhat )(dNuhat/dx)

	nancount := 0

	// First, compute the derivatives of nutilde
	deriv := make([]float64, nDim)
	loc := make([]float64, nDim)
	for i, pt := range data {
		newpt := newData[i]
		setPointValues(newData, NuTilde, planePoints[i], neighbors[i])
		//for i := range data {
		loc[0] = data[i][XLoc]
		loc[1] = data[i][YLoc]
		intercept := scattered.Intercept{
			Force: true,
			Value: newData[i][NuTilde],
		}
		scattered.Plane(loc, planePoints[i], intercept, deriv)

		if pt[IdxX] == 1018 && pt[IdxY] == 16 {
			fmt.Println("looking at nu gradient for bad point")
			fmt.Println(loc, intercept.Value)
			for _, neigh := range planePoints[i] {
				fmt.Println(neigh.Location, neigh.Value)
			}
			fmt.Println(deriv)
			newIdx := i + nX
			fmt.Println(newIdx)
			fmt.Println(newData[newIdx][NuTilde])
			//os.Exit(1)
		}

		dNuHatDX := deriv[0]
		dNuHatDY := deriv[1]
		newpt[DNuHatDX] = dNuHatDX
		newpt[DNuHatDY] = dNuHatDY

		//fmt.Println("nugrad", dNuHatDX, dNuHatDY)

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
		sourceNondimerUNorm := math.Abs(newData[i][NuTilde])*newData[i][TotalVelGradNorm] + NormEps
		newData[i][SourceNondimerUNorm] = sourceNondimerUNorm
		newData[i][LogSourceNondimerUNorm] = math.Log(sourceNondimerUNorm)
		//fmt.Println("sourcenondimer unorm", sourceNondimerUNorm)
		newData[i][NuHatGradMagUNorm] = nuHatGradMag / sourceNondimerUNorm
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

		if math.Abs(source) > 10 {
			fmt.Println()
			fmt.Println("Idxs ", data[i][IdxX], data[i][IdxY])
			fmt.Println("nutilde", newData[i][NuTilde])
			fmt.Println("nutilde grad", newData[i][DNuHatDX], newData[i][DNuHatDY])
			fmt.Println("nusum grad", extraData[i][NuSumTimesDX], extraData[i][NuSumTimesDY])
			fmt.Println("second deriv", nuSumDXX, nuSumDXY, nuSumDYX, nuSumDYY)
			fmt.Println("conv diff", extraData[i][LHS], rhs)
			fmt.Println("source = ", source)
		}
		/*
			if math.Abs(source) > 400 {
				fmt.Println("crazy source")
				fmt.Println("Nu grad ", nuSumDXX, nuSumDXY, nuSumDYX, nuSumDYY)
				fmt.Println("Vel grad = ", pt[DUDX], pt[DUDY], pt[DVDX], pt[DVDY])
				fmt.Println("Wall dist = ", pt[WallDistance])
				os.Exit(1)
			}
		*/
		newpt[Source] = source
		newpt[NondimSource] = source / newpt[SourceNondimer]
		newpt[NondimSource2] = source / newpt[SourceNondimer2]
		newpt[NondimSourceUNorm] = source / newData[i][SourceNondimerUNorm]

		/*
			// Look at the comparison between velocity gradients and estimated velocity gradients
			setPointValues(data, VVel, planePoints[i], neighbors[i])
			intercept = scattered.Intercept{
				Force: true,
				Value: data[i][VVel],
			}
			scattered.Plane(loc, planePoints[i], intercept, deriv)
			//fmt.Println("dudx", data[i][DUDX], "dudy", data[i][DUDY])
			//fmt.Println("dudx", deriv[0], "dudy", deriv[1])
			//fmt.Println("diff:", data[i][DUDX]-deriv[0], data[i][DUDY]-deriv[1])
			u := []float64{data[i][DVDX], data[i][DVDY]}
			fu := []float64{deriv[0], deriv[1]}
			if !floats.EqualApprox(u, fu, 1e-1) && len(planePoints[i]) == 4 {
				if int(pt[IdxX]) < 4 || int(pt[IdxX]) > nX-3 {
					continue
				}
				if int(pt[IdxY]) < 4 || int(pt[IdxY]) > nY-3 {
					continue
				}
				fmt.Println("real = ", u)
				fmt.Println("est = ", fu)
				fmt.Println("diff = ", u[0]-fu[0], u[1]-fu[1])
				fmt.Println(pt[IdxX], pt[IdxY])
				fmt.Println(loc, intercept.Value)
				for j := range planePoints[i] {
					fmt.Println(planePoints[i][j].Location, planePoints[i][j].Value)

				}
				if (u[0] > 0 && fu[0] < 0) || (u[0] < 0 && fu[0] > 0) {
					panic("x diff")
				}
				if (u[1] > 0 && fu[1] < 0) || (u[1] < 0 && fu[1] > 0) {
					panic("y diff")
				}
			}
		*/
	}

	// Compute the k and omega derivatives.
	for i, pt := range data {
		loc[0] = data[i][XLoc]
		loc[1] = data[i][YLoc]
		setPointValues(newData, TurbKinEnergy, planePoints[i], neighbors[i])
		intercept := scattered.Intercept{
			Force: true,
			Value: newData[i][TurbKinEnergy],
		}
		scattered.Plane(loc, planePoints[i], intercept, deriv)
		newData[i][DTurbKinEnergyDX] = deriv[0]
		newData[i][DTurbKinEnergyDY] = deriv[1]
		dkdx := [2]float64{deriv[0], deriv[1]}

		setPointValues(newData, TurbSpecificDissipation, planePoints[i], neighbors[i])
		intercept = scattered.Intercept{
			Force: true,
			Value: newData[i][TurbSpecificDissipation],
		}

		scattered.Plane(loc, planePoints[i], intercept, deriv)
		// Hack at the wall
		/*
			newPlanePoints := make([]*scattered.PointMV, 0)
			if pt[WallDistance] != 0 {
				for j := range planePoints[i] {
					v := planePoints[i][j].Value
					if math.IsNaN(v) || math.IsInf(v, 0) {
						continue
					}
					newPlanePoints = append(newPlanePoints, planePoints[i][j])
				}
			}
			scattered.Plane(loc, newPlanePoints, intercept, deriv)
		*/

		//fmt.Println(planePoints)

		newData[i][DTurbSpecificDissipationDX] = deriv[0]
		newData[i][DTurbSpecificDissipationDY] = deriv[1]
		dOmegaDX := [2]float64{deriv[0], deriv[1]}

		velGrad := &fluid2d.VelGrad{}
		velGrad.SetAll(pt[DUDX], pt[DUDY], pt[DVDX], pt[DVDY])
		// Compute diffusion
		mu := pt[Nu] * Rho

		wd := pt[WallDistance]
		// Hack at wall so gradients work.
		if wd == 0 {
			wd = 1e-10
		}
		turb := &sst.SST2Cache{
			VelGrad:   *velGrad,
			WallDist:  unit.Length(wd),
			K:         fluid.KineticEnergy(newData[i][TurbKinEnergy]),
			Nu:        fluid.KinematicViscosity(pt[Nu]),
			Rho:       fluid.Density(Rho),
			OmegaDiss: fluid.SpecificDissipation(newData[i][TurbSpecificDissipation]),
			DKDX:      dkdx,
			DOmegaDX:  dOmegaDX,
		}
		turb.Compute()
		sigmaK := sst.Blend(turb.F1, sst.SigmaK1, sst.SigmaK2)
		mup := mu + sigmaK*float64(turb.MuT)
		extraData[i][KDiffSstDX] = mup * newData[i][DTurbKinEnergyDX]
		extraData[i][KDiffSstDY] = mup * newData[i][DTurbKinEnergyDY]

		sigmaW := sst.Blend(turb.F1, sst.SigmaW1, sst.SigmaW2)
		mup = mu + sigmaW*float64(turb.MuT)
		extraData[i][OmegaDiffSstDX] = mup * newData[i][DTurbSpecificDissipationDX]
		extraData[i][OmegaDiffSstDY] = mup * newData[i][DTurbSpecificDissipationDY]

		//if (math.IsNaN(mup) || math.IsInf(mup, 0)) && data[i][WallDistance] != 0 {
		//if i == 2304 {
		//if math.IsNaN(mup) {
		//if math.IsInf(extraData[i][OmegaDiffSstDX], 0) || math.IsNaN(extraData[i][OmegaDiffSstDX]) {
		if (math.IsInf(extraData[i][OmegaDiffSstDX], 0) || math.IsNaN(extraData[i][OmegaDiffSstDX])) && pt[WallDistance] != 0 {
			fmt.Println("mup is nan")
			fmt.Println("i = ", i)
			fmt.Println("wall dist = ", data[i][WallDistance])
			fmt.Println("k = ", newData[i][TurbKinEnergy])
			fmt.Println("eps = ", newData[i][TurbDissipation])
			fmt.Println("omega", newData[i][TurbSpecificDissipation])
			fmt.Println("mu = ", mu)
			fmt.Println("rho", Rho)
			fmt.Println("turb omega", turb.OmegaDiss)
			fmt.Println(turb.DKDX)
			fmt.Println(turb.DOmegaDX)
			fmt.Println("cdkw", turb.CDkw)
			fmt.Println("arg1", turb.Arg1)
			fmt.Println("f1 ", turb.F1)
			fmt.Println("mup", mup)
			fmt.Println(newData[i][DTurbSpecificDissipationDX], newData[i][DTurbSpecificDissipationDY])
			fmt.Println(extraData[i][OmegaDiffSstDX], extraData[i][OmegaDiffSstDY])
			fmt.Println(sigmaW)
			fmt.Println(turb.MuT)
			os.Exit(1)
		}
	}

	// Do the budget for the K and Omega source terms
	for i, pt := range data {
		loc[0] = data[i][XLoc]
		loc[1] = data[i][YLoc]
		setPointValues(extraData, KDiffSstDX, planePoints[i], neighbors[i])
		intercept := scattered.Intercept{
			Force: true,
			Value: extraData[i][KDiffSstDX],
		}
		scattered.Plane(loc, planePoints[i], intercept, deriv)

		dissKXX := deriv[0]

		setPointValues(extraData, KDiffSstDY, planePoints[i], neighbors[i])
		intercept = scattered.Intercept{
			Force: true,
			Value: extraData[i][KDiffSstDY],
		}
		scattered.Plane(loc, planePoints[i], intercept, deriv)
		dissKYY := deriv[1]

		dissK := dissKXX + dissKYY
		convK := Rho * (pt[UVel]*newData[i][DTurbKinEnergyDX] + pt[VVel]*newData[i][DTurbKinEnergyDY])

		newData[i][TurbKinEnergySourceBudget] = convK - dissK

		setPointValues(extraData, OmegaDiffSstDX, planePoints[i], neighbors[i])
		intercept = scattered.Intercept{
			Force: true,
			Value: extraData[i][OmegaDiffSstDX],
		}
		scattered.Plane(loc, planePoints[i], intercept, deriv)

		dissOmegaXX := deriv[0]

		setPointValues(extraData, OmegaDiffSstDY, planePoints[i], neighbors[i])
		intercept = scattered.Intercept{
			Force: true,
			Value: extraData[i][OmegaDiffSstDY],
		}
		scattered.Plane(loc, planePoints[i], intercept, deriv)

		dissOmegaYY := deriv[0]

		dissOmega := dissOmegaXX + dissOmegaYY
		convOmega := Rho * (pt[UVel]*newData[i][DTurbSpecificDissipationDX] + pt[VVel]*newData[i][DTurbSpecificDissipationDY])
		newData[i][TurbSpecificDissipationSourceBudget] = convOmega - dissOmega

		//fmt.Println("K = ", newData[i][TurbKinEnergy], "Rho = ", Rho, "Omega = ", newData[i][TurbSpecificDissipation], "Nondim=", newData[i][TurbKinEnergySourceNondimer])

		newData[i][NondimTurbKinEnergySource] = newData[i][TurbKinEnergySourceBudget] / newData[i][TurbKinEnergySourceNondimer]
		newData[i][NondimTurbSpecificDissipationSource] = newData[i][TurbSpecificDissipationSourceBudget] / newData[i][TurbSpecificDissipationSourceNondimer]

		/*
			if newData[i][TurbKinEnergySourceBudget] > 60000 {
				fmt.Println("xIdx", data[i][IdxX], "idxy", data[i][IdxY], "WallDist", data[i][WallDistance])
				fmt.Println("convK", convK, "dissK", dissK)
				fmt.Println("sourcek", newData[i][TurbKinEnergySourceBudget], "nondim", newData[i][NondimTurbKinEnergySource])
			}
		*/

		/*
			if newData[i][TurbSpecificDissipationSourceBudget] > 1e15 {
				fmt.Println("xIdx", data[i][IdxX], "idxy", data[i][IdxY], "WallDist", data[i][WallDistance])
				fmt.Println("conv", convOmega, "dissOmega", dissOmega)
				//fmt.Println("sourcek", newData[i][TurbKinEnergySourceBudget], "nondim", newData[i][NondimTurbKinEnergySource])
			}
		*/

		//fmt.Println(newData[i][TurbKinEnergySourceBudget] - turb.SourceK)

		if math.IsNaN(newData[i][TurbSpecificDissipationSourceBudget]) && pt[WallDistance] != 0 {
			fmt.Println(pt[WallDistance])
			fmt.Println(i)
			fmt.Println("diss omega", dissOmega, "conv omega", convOmega)
			fmt.Println(dissOmegaXX, dissOmegaYY)
			os.Exit(1)
		}
	}

	// do a quick check to make sure all the fields of newData got set
	for j := 0; j < len(newData[0]); j++ {
		fmt.Println(j)
		var hasNonzero bool
		for i, pt := range newData {
			if pt[j] != 0 {
				hasNonzero = true
			}
			if math.IsInf(pt[j], 0) && data[i][WallDistance] != 0 {
				fmt.Println("i = ", i, "j = ", j, " is inf, name is ", newFeatures[j], "WallDist", data[i][WallDistance])
			}

			if math.IsNaN(pt[j]) && data[i][WallDistance] != 0 {
				fmt.Println("i = ", i, "j = ", j, " is nan, name is ", newFeatures[j], "WallDist", data[i][WallDistance])
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

	err, allHeadings, allData := appendCSV(r, newFeatures, newData, w)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Finished")
	fmt.Println(len(allHeadings))
	ra, ca := allData.Dims()
	fmt.Println(ra, ca)

	// Construct an ML dataset with one feature per column

	// Ignore all of the points without the maximum number of neighbors
	var maxNeighbors int
	for i := range neighbors {
		if len(neighbors[i]) > maxNeighbors {
			maxNeighbors = len(neighbors[i])
		}
	}

	mldata := make([][]float64, 0, ra)
	mlHeadings := make([]string, 0)
	var headingAdded bool
	for i := 0; i < ra; i++ {
		if len(neighbors[i]) != maxNeighbors {
			// Ignore the points that don't have a sufficient number of neighbors
			continue
		}
		d := make([]float64, 0)
		// Append the data at this location
		for _, str := range thisLocationFeatures {
			var v float64
			if idx := findStringLocation(features, str); idx != -1 {
				v = data[i][idx]
			} else if idx := findStringLocation(newFeatures, str); idx != -1 {
				v = newData[i][idx]
			} else {
				panic("Didn't find str " + str)
			}
			d = append(d, v)
			if !headingAdded {
				mlHeadings = append(mlHeadings, "this_"+str)
			}
		}
		var neighborCount int
		for _, neighborIdx := range neighbors[i] {
			neighborCount++
			for _, str := range neighborFeatures {
				var v float64
				if idx := findStringLocation(features, str); idx != -1 {
					v = data[neighborIdx][idx]
				} else if idx := findStringLocation(newFeatures, str); idx != -1 {
					v = newData[neighborIdx][idx]
				} else if str == "DeltaXLoc" {
					v = data[i][XLoc] - data[neighborIdx][XLoc]
				} else if str == "DeltaYLoc" {
					v = data[i][YLoc] - data[neighborIdx][YLoc]
				} else {
					panic("Didn't find str " + str)
				}
				d = append(d, v)
				if !headingAdded {
					fstr := fmt.Sprintf("neighbor_%v_%v", neighborCount, str)
					mlHeadings = append(mlHeadings, fstr)
				}
			}
		}
		mldata = append(mldata, d)
		headingAdded = true
	}
	rml := len(mldata)
	cml := len(mldata[0])
	if len(mlHeadings) != cml {
		panic("wrong number of headings")
	}
	mlMatrix := mat64.NewDense(rml, cml, nil)
	for i := 0; i < rml; i++ {
		for j := 0; j < cml; j++ {
			mlMatrix.Set(i, j, mldata[i][j])
		}
	}

	setname := pre + "_mldata_withloc" + ext
	fml, err := os.Create(setname)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("mlheadings = ", mlHeadings)
	w = numcsv.NewWriter(fml)
	err = w.WriteAll(mlHeadings, mlMatrix)
	if err != nil {
		log.Fatal(err)
	}
}

func appendCSV(r *numcsv.Reader, newHeadings []string, newData [][]float64, w *numcsv.Writer) (err error, allheadings []string, alldata *mat64.Dense) {
	// Read all of the headings
	headings, err := r.ReadHeading()
	if err != nil {
		return err, nil, nil
	}
	data, err := r.ReadAll()
	if err != nil {
		return err, nil, nil
	}

	rows, cols := data.Dims()

	if len(newData) != rows {
		return errors.New("nData mismatch"), nil, nil
	}
	dim := len(newData[0])
	for _, pt := range newData {
		if len(pt) != dim {
			return errors.New("dim mismatch"), nil, nil
		}
	}
	if len(newHeadings) != dim {
		fmt.Println(newHeadings)
		fmt.Println(dim)
		return errors.New("Heading length mismatch"), nil, nil
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

	return w.WriteAll(headings, m), headings, m

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
