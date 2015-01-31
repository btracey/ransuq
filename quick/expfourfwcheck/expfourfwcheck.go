package main

import (
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"

	"code.google.com/p/plotinum/plot"
	"code.google.com/p/plotinum/plotter"

	"github.com/btracey/numcsv"
	"github.com/btracey/ransuq/internal/util"
	"github.com/btracey/turbulence/sa"
)

var gopath string

func init() {
	gopath = os.Getenv("GOPATH")
	if gopath == "" {
		panic("gopath not set")
	}
}

func main() {
	filename := filepath.Join(gopath, "data", "ransuq", "HiFi", "exp4_mod.txt")
	f, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	r := numcsv.NewReader(f)
	r.Comma = " "
	headings, err := r.ReadHeading()
	if err != nil {
		log.Fatal(err)
	}
	data, err := r.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	Chi := util.FindStringLocation(headings, "Chi")
	OmegaBar := util.FindStringLocation(headings, "OmegaBar")
	Fw := util.FindStringLocation(headings, "Fw")

	rows, _ := data.Dims()
	truth := make([]float64, rows)
	calc := make([]float64, rows)
	for i := 0; i < rows; i++ {
		chi := data.At(i, Chi)
		omegaBar := data.At(i, OmegaBar)
		fv1 := sa.Fv1(chi)
		fv2 := sa.Fv2(chi, fv1)
		shat := omegaBar + 1/sa.K2*fv2
		r := math.Min(1/(shat*sa.K2), 10)
		g := sa.G(r)
		fwcalc := sa.Fw(g)
		fw := data.At(i, Fw)
		if fw > 4 {
			fmt.Println(i)
		}
		truth = append(truth, fw)
		calc = append(calc, fwcalc)
	}
	fmt.Println("done calc")
	scatter, err := plotter.NewScatter(VecXY{truth, calc})
	if err != nil {
		log.Fatal(err)
	}
	p, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}
	p.Add(scatter)
	err = p.Save(4, 4, "predvtruth.jpg")
	if err != nil {
		log.Fatal(err)
	}
}

type VecXY struct {
	X []float64
	Y []float64
}

func (v VecXY) Len() int {
	if len(v.X) != len(v.Y) {
		panic("length mismatch")
	}
	return len(v.X)
}

func (v VecXY) XY(i int) (x, y float64) {
	return v.X[i], v.Y[i]
}
