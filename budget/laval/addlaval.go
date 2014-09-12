package main

import (
	"fmt"
	"log"

	"github.com/btracey/fluid/fluid2d"
	"github.com/btracey/matrix/twod"
	"github.com/btracey/ransuq/dataloader"
	"github.com/btracey/ransuq/datawrapper"
)

var features = []string{"DUDX", "DUDY", "DVDX", "DVDY", "UU", "UV", "VV"}

var (
	DUDX = findStringLocation(features, "DUDX")
	DUDY = findStringLocation(features, "DUDX")
	DVDX = findStringLocation(features, "DUDX")
	DVDY = findStringLocation(features, "DUDX")
	Nu   = findStringLocation(features, "Nu")
	UU   = findStringLocation(features, "UU")
	UV   = findStringLocation(features, "UV")
	VV   = findStringLocation(features, "VV")
)

var newFeatures = []string{"StrainRateMag", "VorticityMag", "NuTilde", "Chi"}

var (
	StrainRateMag = findStringLocation(newFeatures, "StrainRateMag")
	VorticityMag  = findStringLocation(newFeatures, "VorticityMag")
	NuTilde       = findStringLocation(newFeatures, "NuTilde")
	Chi           = findStringLocation(newFeatures, "Chi")
)

const NuTildeEpsilon = 1e-10

func main() {
	dataset := "/Users/brendan/Documents/mygo/data/ransuq/laval/laval_csv.dat"

	// Need to load the data
	data := loadData(dataset, features)

	fmt.Println(data)

	newFeatures := make([][]float64, len(data))
	for i, pt := range data {
		newpt := make([]float64, len(pt))
		newFeatures[i] = newpt
		velGrad := fluid2d.VelGrad{}
		(&velGrad).Set(pt[DUDX], pt[DUDY], pt[DVDX], pt[DVDY])
		fmt.Println("Vel grad = ", velGrad)

		sym, skewsym := velGrad.Split()
		strainRate := fluid2d.StrainRate{sym}
		vorticity := fluid2d.Vorticity{skewsym}

		newpt[StrainRateMag] = strainRate.Norm(twod.Frobenius2)
		newpt[VorticityMag] = vorticity.Norm(twod.Frobenius2)

		tau := fluid2d.ReynoldsStress{}
		(&tau).Set(pt[UU], pt[UV], pt[VV])

		nuTilde := fluid2d.TurbKinVisc(tau, strainRate, NuTildeEpsilon)

		newpt[NuTilde] = float64(nuTilde)
		newpt[Chi] = newpt[NuTilde] / newpt[Nu]
	}
	fmt.Println(newpt)
}

func loadData(dataset string, features []string) [][]float64 {

	set := &dataloader.Dataset{
		Filename: dataset,
		Format:   datawrapper.Laval,
	}

	allData, err := dataloader.Load(features, []*dataloader.Dataset{set})
	if err != nil {
		log.Fatal(err)
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
