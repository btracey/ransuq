package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"

	"code.google.com/p/plotinum/plot"

	"github.com/btracey/myplot"
	"github.com/btracey/numcsv"
	"github.com/btracey/ransuq/internal/util"
	"github.com/gonum/stat"
)

func main() {
	lavalpath := "/Users/brendan/Documents/mygo/data"
	lavalpath = filepath.Join(lavalpath, "ransuq", "laval")
	dataname := filepath.Join(lavalpath, "laval_csv_computed.dat")
	f, err := os.Open(dataname)
	if err != nil {
		log.Fatal(err)
	}
	r := numcsv.NewReader(f)
	headings, err := r.ReadHeading()
	if err != nil {
		log.Fatal(err)
	}
	_ = headings
	fmt.Println(headings)
	data, err := r.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	rows, cols := data.Dims()
	fmt.Println(rows, cols)

	XLoc := util.FindStringLocation(headings, "grid_x")
	YLoc := util.FindStringLocation(headings, "grid_yx")
	DUDX := util.FindStringLocation(headings, "dx_mean_u_xyz")
	DUDY := util.FindStringLocation(headings, "dy_mean_u_xyz")
	DVDX := util.FindStringLocation(headings, "dx_mean_v_xyz")
	DVDY := util.FindStringLocation(headings, "dy_mean_v_xyz")
	UU := util.FindStringLocation(headings, "reynolds_stress_uu_xyz")
	UV := util.FindStringLocation(headings, "reynolds_stress_uv_xyz")
	VV := util.FindStringLocation(headings, "reynolds_stress_vv_xyz")
	VorticityMag := util.FindStringLocation(headings, "VorticityMag")
	StrainRateMag := util.FindStringLocation(headings, "StrainRateMag")
	Source := util.FindStringLocation(headings, "Source")

	xLoc := data.Col(nil, XLoc)
	yLoc := data.Col(nil, YLoc)
	dudx := data.Col(nil, DUDX)
	dudy := data.Col(nil, DUDY)
	dvdx := data.Col(nil, DVDX)
	dvdy := data.Col(nil, DVDY)
	uu := data.Col(nil, UU)
	uv := data.Col(nil, UV)
	vv := data.Col(nil, VV)
	vorticityMag := data.Col(nil, VorticityMag)
	strainRateMag := data.Col(nil, StrainRateMag)
	source := data.Col(nil, Source)

	makePlot(xLoc, yLoc, dudx, "dudx.jpg")
	makePlot(xLoc, yLoc, dudy, "dudy.jpg")
	makePlot(xLoc, yLoc, dvdx, "dvdx.jpg")
	makePlot(xLoc, yLoc, dvdy, "dvdy.jpg")
	makePlot(xLoc, yLoc, uu, "uu.jpg")
	makePlot(xLoc, yLoc, uv, "uv.jpg")
	makePlot(xLoc, yLoc, vv, "vv.jpg")
	makePlot(xLoc, yLoc, vorticityMag, "vorticityMag.jpg")
	makePlot(xLoc, yLoc, strainRateMag, "strainRateMag.jpg")
	makePlot(xLoc, yLoc, source, "source.jpg")
}

func makePlot(x, y, z []float64, name string) {
	z2 := make([]float64, len(z))
	copy(z2, z)
	sort.Float64s(z2)

	lb := stat.Quantile(0.05, stat.Empirical, z2, nil)
	ub := stat.Quantile(0.95, stat.Empirical, z2, nil)
	pts := myplot.VecXYZ{X: x, Y: y, Z: z}
	contour, err := myplot.NewColoredScatter(pts)
	contour.SetColormap(&myplot.Jet{})
	if err != nil {
		log.Fatal(err)
	}
	contour.SetScale(lb, ub)
	p, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}
	contour.GlyphStyle.Radius = 2
	p.Add(contour)
	p.Save(8, 8, filepath.Join("plots", name))
}
