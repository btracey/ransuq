package main

import (
	"fmt"
	"log"
	"math"

	"code.google.com/p/plotinum/plot"
	"code.google.com/p/plotinum/plotter"

	"github.com/btracey/fluid"
	"github.com/btracey/fluid/fluid2d"
	"github.com/btracey/ransuq"
	"github.com/btracey/ransuq/settings"
	"github.com/btracey/su2tools/driver"
	"github.com/btracey/turbulence/sa"
	"github.com/btracey/turbulence/sst"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
	"github.com/gonum/unit"
)

var (
	inputFeatures = []string{"idx_x", "idx_y", "DUDX", "DUDY", "DVDX", "DVDY", "WallDistance",
		"TurbKinEnergy", "Nu", "Rho", "TurbSpecificDissipation", "TurbDissipation",
		"DTurbKinEnergyDX", "DTurbKinEnergyDY", "DTurbSpecificDissipationDX",
		"DTurbSpecificDissipationDY", "NuTilde", "DNuHatDX", "DNuHatDY", "SourceNondimer",
		"TotalVelGradNorm",
	}

	outputFeatures = []string{"TurbKinEnergySourceBudget", "TurbSpecificDissipationSourceBudget",
		"Source", "NondimTurbKinEnergySource", "NondimTurbSpecificDissipationSource",
		"NondimSource", "NondimSource2", "NondimSourceUNorm",
	}
)

var (
	IdxX                       = findStringLocation(inputFeatures, "idx_x")
	IdxY                       = findStringLocation(inputFeatures, "idx_y")
	DUDX                       = findStringLocation(inputFeatures, "DUDX")
	DUDY                       = findStringLocation(inputFeatures, "DUDY")
	DVDX                       = findStringLocation(inputFeatures, "DVDX")
	DVDY                       = findStringLocation(inputFeatures, "DVDY")
	Nu                         = findStringLocation(inputFeatures, "Nu")
	UU                         = findStringLocation(inputFeatures, "TauUU")
	UV                         = findStringLocation(inputFeatures, "TauUV")
	VV                         = findStringLocation(inputFeatures, "TauVV")
	WallDistance               = findStringLocation(inputFeatures, "WallDistance")
	UVel                       = findStringLocation(inputFeatures, "UVel")
	VVel                       = findStringLocation(inputFeatures, "VVel")
	DissUU                     = findStringLocation(inputFeatures, "DissUU")
	DissVV                     = findStringLocation(inputFeatures, "DissVV")
	Rho                        = findStringLocation(inputFeatures, "Rho")
	StrainRateMag              = findStringLocation(inputFeatures, "StrainRateMag")
	VorticityMag               = findStringLocation(inputFeatures, "VorticityMag")
	VorticityMagNondim         = findStringLocation(inputFeatures, "VorticityMagNondim")
	NuTilde                    = findStringLocation(inputFeatures, "NuTilde")
	Chi                        = findStringLocation(inputFeatures, "Chi")
	SourceNondimer             = findStringLocation(inputFeatures, "SourceNondimer")
	SourceNondimer2            = findStringLocation(inputFeatures, "SourceNondimer2")
	DNuHatDX                   = findStringLocation(inputFeatures, "DNuHatDX")
	DNuHatDY                   = findStringLocation(inputFeatures, "DNuHatDY")
	NuHatGradMag               = findStringLocation(inputFeatures, "NuHatGradMag")
	NuHatGradMagBar            = findStringLocation(inputFeatures, "NuHatGradMagBar")
	TotalVelGradNorm           = findStringLocation(inputFeatures, "TotalVelGradNorm")
	VelGradDet                 = findStringLocation(inputFeatures, "VelGradDet")
	VelVortOverNorm            = findStringLocation(inputFeatures, "VelVortOverNorm")
	VelDetOverNorm             = findStringLocation(inputFeatures, "VelDetOverNorm")
	NuGradAngle                = findStringLocation(inputFeatures, "NuGradAngle")
	SourceNondimerUNorm        = findStringLocation(inputFeatures, "SourceNondimerUNorm")
	NuVelGradNormRatio         = findStringLocation(inputFeatures, "NuVelGradNormRatio")
	TurbKinEnergy              = findStringLocation(inputFeatures, "TurbKinEnergy")
	DTurbKinEnergyDX           = findStringLocation(inputFeatures, "DTurbKinEnergyDX")
	DTurbKinEnergyDY           = findStringLocation(inputFeatures, "DTurbKinEnergyDY")
	TurbDissipation            = findStringLocation(inputFeatures, "TurbDissipation")
	TurbSpecificDissipation    = findStringLocation(inputFeatures, "TurbSpecificDissipation")
	DTurbSpecificDissipationDX = findStringLocation(inputFeatures, "DTurbSpecificDissipationDX")
	DTurbSpecificDissipationDY = findStringLocation(inputFeatures, "DTurbSpecificDissipationDY")

	Source                              = findStringLocation(outputFeatures, "Source")
	TurbKinEnergySourceBudget           = findStringLocation(outputFeatures, "TurbKinEnergySourceBudget")
	TurbSpecificDissipationSourceBudget = findStringLocation(outputFeatures, "TurbSpecificDissipationSourceBudget")
	NondimTurbKinEnergySource           = findStringLocation(outputFeatures, "NondimTurbKinEnergySource")
	NondimTurbSpecificDissipationSource = findStringLocation(outputFeatures, "NondimTurbSpecificDissipationSource")
	NondimSource                        = findStringLocation(outputFeatures, "NondimSource")
	NondimSource2                       = findStringLocation(outputFeatures, "NondimSource2")
	NondimSourceUNorm                   = findStringLocation(outputFeatures, "NondimSourceUNorm")
)

func main() {
	datasetStr := "laval_dns_sa"

	/********** Load the data ************/
	datasets, err := settings.GetDatasets(datasetStr, driver.Serial{})
	if err != nil {
		log.Fatal(err)
	}

	inputDataMat, outputDataMat, weights, err := ransuq.DenseLoadAll(datasets, inputFeatures, outputFeatures, nil, nil)

	if err != nil {
		log.Fatal(err)
	}

	if weights != nil {
		log.Fatal("not coded for weighted data")
	}

	inputData := inputDataMat.(*mat64.Dense)
	outputData := outputDataMat.(*mat64.Dense)

	nSamples, _ := inputData.Dims()

	fmt.Println("nSamples = ", nSamples)
	//outputData.Dims()

	sstSourceKPred := make([]float64, nSamples)
	sstSourceKBudget := make([]float64, nSamples)
	sstSourceOmegaPred := make([]float64, nSamples)
	sstSourceOmegaBudget := make([]float64, nSamples)
	sstNondimSourceKPred := make([]float64, nSamples)
	sstNondimSourceOmegaPred := make([]float64, nSamples)
	sstNondimSourceKBudget := make([]float64, nSamples)
	sstNondimSourceOmegaBudget := make([]float64, nSamples)

	saSourceBudget := make([]float64, nSamples)
	saSourcePred := make([]float64, nSamples)
	saNondimSourceBudget := make([]float64, nSamples)
	saNondimSourceBudget2 := make([]float64, nSamples)
	saNondimSourcePred := make([]float64, nSamples)
	saNondimSourcePred2 := make([]float64, nSamples)
	saNondimSourceBudget3 := make([]float64, nSamples)
	saNondimSourcePred3 := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		pt := inputData.RowView(i)
		//fmt.Println("x idx", pt[IdxX], "y idx", pt[IdxY])
		velGrad := &fluid2d.VelGrad{}
		velGrad.SetAll(pt[DUDX], pt[DUDY], pt[DVDX], pt[DVDY])
		wd := pt[WallDistance]
		if wd < 1e-6 {
			panic("small wall distance")
		}
		dkdx := [2]float64{pt[DTurbKinEnergyDX], pt[DTurbKinEnergyDY]}
		dOmegaDX := [2]float64{pt[DTurbSpecificDissipationDX], pt[DTurbSpecificDissipationDY]}

		turb := &sst.SST2Cache{
			VelGrad:   *velGrad,
			WallDist:  unit.Length(wd),
			K:         fluid.KineticEnergy(pt[TurbKinEnergy]),
			Nu:        fluid.KinematicViscosity(pt[Nu]),
			Rho:       fluid.Density(pt[Rho]),
			OmegaDiss: fluid.SpecificDissipation(pt[TurbSpecificDissipation]),
			DKDX:      dkdx,
			DOmegaDX:  dOmegaDX,
		}
		turb.Compute()
		sstSourceKPred[i] = turb.SourceK
		sstSourceKBudget[i] = outputData.At(i, TurbKinEnergySourceBudget)

		sstSourceOmegaPred[i] = turb.SourceOmega
		sstSourceOmegaBudget[i] = outputData.At(i, TurbSpecificDissipationSourceBudget)

		sstNondimSourceKBudget[i] = outputData.At(i, NondimTurbKinEnergySource)
		sstNondimSourceKPred[i] = turb.NondimSourceK
		sstNondimSourceOmegaBudget[i] = outputData.At(i, NondimTurbSpecificDissipationSource)
		sstNondimSourceOmegaPred[i] = turb.NondimSourceOmega

		SA := &sa.SA{
			NDim:         2,
			Nu:           pt[Nu],
			NuHat:        pt[NuTilde],
			DNuHatDX:     []float64{pt[DNuHatDX], pt[DNuHatDY]},
			DUIdXJ:       [][]float64{{pt[DUDX], pt[DVDX]}, {pt[DUDY], pt[DVDY]}},
			WallDistance: wd,
		}

		source := SA.Source()
		if source > 9.7e6 {
			fmt.Printf("%#v\n", SA)
		}
		saSourcePred[i] = source
		budgetSource := outputData.At(i, Source)
		saSourceBudget[i] = budgetSource
		/*
			if math.Abs(budgetSource) > 300 {
				saSourceBudget[i] = 0
			}
			if math.Abs(source) > 300 {
				saSourcePred[i] = 0
			}
		*/
		nusum := pt[Nu] + pt[NuTilde]
		nondimer := (nusum * nusum) / (pt[WallDistance] * pt[WallDistance])
		nondimer2 := (pt[NuTilde] / pt[WallDistance]) * (pt[NuTilde] / pt[WallDistance])
		nondimer3 := math.Abs(pt[NuTilde]) * pt[TotalVelGradNorm]
		saNondimSourcePred[i] = source / nondimer
		saNondimSourcePred2[i] = source / nondimer2
		saNondimSourcePred3[i] = source / nondimer3
		//saNondimSourceBudget[i] = outputData.At(i, NondimSource)
		//saNondimSourceBudget[i] = outputData.At(i, Source) / outputData.At(i, SourceNondimerUNorm)
		saNondimSourceBudget[i] = outputData.At(i, NondimSource)
		saNondimSourceBudget2[i] = outputData.At(i, NondimSource2)
		saNondimSourceBudget3[i] = outputData.At(i, NondimSourceUNorm)

		//fmt.Println("k = ", pt[TurbKinEnergy], " eps = ", pt[TurbDissipation], "omega = ", pt[TurbSpecificDissipation])
	}

	meanX := stat.Mean(sstSourceKPred, nil)
	meanY := stat.Mean(sstSourceKBudget, nil)
	stdX := stat.StdDev(sstSourceKPred, meanX, nil)
	stdY := stat.StdDev(sstSourceKBudget, meanY, nil)
	corr := stat.Correlation(sstSourceKPred, meanX, stdX, sstSourceKBudget, meanY, stdY, nil)
	//cov := stat.Covariance(sstSourceKPred, meanX, sstSourceKBudget, meanY, nil)
	fmt.Println(corr)

	makePlot(sstSourceKBudget, sstSourceKPred, "sstKSourceComp.jpg")
	makePlot(sstSourceOmegaBudget, sstSourceOmegaPred, "sstOmegaSourceComp.jpg")
	makePlot(sstNondimSourceKBudget, sstNondimSourceKPred, "sstNondimKComp.jpg")
	makePlot(sstNondimSourceOmegaBudget, sstNondimSourceOmegaPred, "sstNondimOmegaComp.jpg")
	makePlot(saSourceBudget, saSourcePred, "saSourceComp.jpg")
	makePlot(saNondimSourceBudget, saNondimSourcePred, "saNondimSourceComp.jpg")
	makePlot(saNondimSourceBudget2, saNondimSourcePred2, "saNondimSourceComp2.jpg")
	makePlot(saNondimSourceBudget3, saNondimSourcePred3, "saNondimSourceCompUNorm.jpg")
}

func findStringLocation(s []string, str string) int {
	for i, v := range s {
		if v == str {
			return i
		}
	}
	return -1
}

func makePlot(x, y []float64, name string) {
	if len(x) != len(y) {
		panic("length mismatch")
	}
	f := plotter.NewFunction(func(x float64) float64 { return x })

	pts := make(plotter.XYs, len(x))
	for i := range pts {
		pts[i].X = x[i]
		pts[i].Y = y[i]
	}

	p, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}
	scatter, err := plotter.NewScatter(pts)
	if err != nil {
		log.Fatal(err)
	}
	p.Add(scatter)
	p.Add(f)
	p.Save(4, 4, name)
}
