package datawrapper

import (
	"fmt"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/btracey/ransuq/dataloader"
	//"github.com/btracey/ransuq/mldriver"

	"github.com/btracey/su2tools/driver"

	"code.google.com/p/plotinum/plot"
	"code.google.com/p/plotinum/plotter"
)

var _ = fmt.Println

var colorwheel []color.RGBA = []color.RGBA{{R: 255, B: 128, A: 255}, {B: 255, A: 255}, {G: 255, A: 255}, {R: 0, G: 0, B: 0, A: 255}}

// FlatplateCompare compares the predictions of the drivers. The first driver
// is used as the base
func flatplateCompare(drivers []*driver.Driver, resultLocation string) error {
	// comment
	inputNames := []string{
		"XLoc",
		"YLoc",
		"UVel",
		"VVel",
		"Nu",
		"NuHat",
		"Source",
		"WallDistance",
		"Density",
		"Viscosity",
		"DUDX",
		"DUDY",
		"DVDX",
		"DVDY",
		"YPlus",
		"Production",
		"Destruction",
		"CrossProduction",
		"NondimProduction",
		"NondimDestruction",
		"NondimCrossProduction",
		"NondimDestructionMod",
		"NondimSourceMod",
	}
	fields := mapInputs(inputNames)

	// Load in the data

	datasets := make([]*dataloader.Dataset, len(drivers))
	for i, drive := range drivers {
		datasets[i] = &dataloader.Dataset{
			Name:     drive.Name,
			Filename: drive.Fullpath(drive.Options.SolutionFlowFilename),
			Format:   &dataloader.SU2_restart_2dturb{},
		}
	}

	//datasets := mldriver.ConstructDataloaders(drivers)
	data, err := dataloader.Load(inputNames, datasets)
	if err != nil {
		return err
	}

	/*
		gamma := drivers[0].Options.GammaValue
		R := drivers[0].Options.GasConstant
		T := drivers[0].Options.FreestreamTemperature
		uInf := drivers[0].Options.MachNumber * math.Sqrt(gamma*R*T)
	*/
	uInf := drivers[0].Options.MachNumber
	rhoInf := drivers[0].Options.RefDensity

	wallData := findZeroLocations(data, fields, 1e-13)
	xlocs, cfs := calculateSkinFrictionCoefficient(wallData, fields, rhoInf, uInf)
	//labels := make([]string, 0, len(datasets))
	labels := make([]string, len(datasets))
	//for i := range drivers {
	//	labels = append(labels, drivers[i].FancyName)
	//}
	if len(labels) != 2 {
		panic("wrong logic")
	}
	labels[0] = "True"
	labels[1] = "Predicted"
	title := "Skin Friction Coefficient Comparison"
	p, err := plotSkinFrictionCoefficient(xlocs, cfs, title, labels)
	if err != nil {
		return err
	}
	err = os.MkdirAll(resultLocation, 0700)
	if err != nil {
		return err
	}
	err = p.Save(4, 4, filepath.Join(resultLocation, "cfplot.pdf"))
	if err != nil {
		return err
	}

	xLocs := []float64{4.0192531261400000e-01, 7.9681583412099999e-01, 1.2267475635900000, 1.6992907191700000, 2.0}
	//xLocs := []float64{2.0}
	profileData := findVelocityPlotLocations(data, xLocs, uInf, 1e-5, fields)
	err = plotProfileData(profileData, xLocs, fields, labels, resultLocation, uInf)
	if err != nil {
		return err
	}
	//PlotVelocityProfiles(profileData, xLocs, fields, labels, resultLocation)
	return nil
}

// mapInputs
func mapInputs(s []string) map[string]int {
	m := make(map[string]int)
	for i, str := range s {
		_, ok := m[str]
		if ok {
			panic("string " + str + " present multiple times ")
		}
		m[str] = i
	}
	return m
}

// Return all the data at the points where the Y location is within tol of zero and the X location is > 0
func findZeroLocations(data [][][]float64, fields map[string]int, tol float64) (zeroData [][][]float64) {
	zeroData = make([][][]float64, len(data))
	for i := range data {
		zeroData[i] = make([][]float64, 0)
	}
	for i := range data {
		for j := range data[i] {
			if data[i][j][fields["XLoc"]] > 0 && data[i][j][fields["YLoc"]] < tol {
				zeroData[i] = append(zeroData[i], data[i][j])
			}
		}
	}
	return zeroData
}

func findVelocityPlotLocations(data [][][]float64, xLocs []float64, uInf float64, tol float64, fields map[string]int) (pts [][][][]float64) {
	// For each data set, need to collect teh points for each x location
	pts = make([][][][]float64, len(xLocs))
	tmpPts := make([][][][]float64, len(xLocs))
	for i := range pts {
		pts[i] = make([][][]float64, len(data))
		tmpPts[i] = make([][][]float64, len(data))
		for j := range data {
			pts[i][j] = make([][]float64, 0)
			tmpPts[i][j] = make([][]float64, 0)
		}
	}
	xind := fields["XLoc"]
	yind := fields["YLoc"]
	velind := fields["UVel"]
	yplusInd := fields["YPlus"]

	const lastPointDist = 1e-11

	for i, loc := range xLocs {
		// Loop over the data to find the points in that dataset that are close
		for k := range data[0] {
			//tmpPts := make([][][]float64, len(data))

			// See if that point is close enough to the xLoc
			//for l := range data {
			//	fmt.Println("data xind = ", l, data[l][k][xind])
			//	fmt.Println("data yind = ", l, data[l][k][yind])
			//}
			if !(math.Abs(data[0][k][xind]-loc) < tol) {
				continue
			}
			// append the point to the data
			for j := range data {
				tmpPts[i][j] = append(tmpPts[i][j], data[j][k])
			}
		}
	}

	ypluses := make([]float64, len(xLocs))

	for i, _ := range xLocs {
		// Sort the base data set
		l := lengthSorter{data: tmpPts[i], yind: yind}
		sort.Sort(l)
		// Used a pointer to tmpPts, so should be sorted

		// Find the y length at the end of the boundary layer
		yDist := 0.0
		for k := range tmpPts[i][0] {
			if tmpPts[i][0][k][velind] > 0.99*uInf {
				break
			}
			yDist = tmpPts[i][0][k][yind]
		}

		// Find the y location of the first y point
		//TODO: THIS IS A TERRIBLE WAY TO DO IT
		yFirst := tmpPts[i][0][1][yind]

		// Double the length
		yDist *= 10
		// Append the appropriate points
		// SU2 stores this at the wall and does weird things elsewhere. Store the correct yplus
		for k := range tmpPts[i][0] {
			//fmt.Println("yloc is ", tmpPts[i][0][k][yind])
			//fmt.Println("u is ", tmpPts[i][0][k][velind])
			//fmt.Println("tmp y is ", tmpPts[i][0][k][yind])
			if tmpPts[i][0][k][yind] > yDist || tmpPts[i][0][k][yind] < lastPointDist {
				if tmpPts[i][0][k][yind] < lastPointDist {
					ypluses[i] = tmpPts[i][0][k][yplusInd]
				}
				//fmt.Println("Not appended")
				continue
			}
			//fmt.Println("yloc is ", tmpPts[i][0][k][yind])
			//fmt.Println("yplus is ", tmpPts[i][0][k][yind])
			//fmt.Println("u is ", tmpPts[i][0][k][velind])
			for j := range tmpPts[i] {
				//fmt.Println("Yplus = ", ypluses[i])
				//data[j][k][yplusInd] = data[j][k][yind] / ypluses[i]
				tmpPts[i][j][k][yplusInd] = tmpPts[i][j][k][yind] / yFirst * ypluses[i]
				//fmt.Println("data y = ", tmpPts[i][j][k][yind])
				//fmt.Println("unscaled yplus = ", tmpPts[i][j][k][yplusInd])

				pts[i][j] = append(pts[i][j], tmpPts[i][j][k])
			}
		}

	}
	return pts
}

type lengthSorter struct {
	data [][][]float64
	yind int
}

func (l lengthSorter) Len() int {
	return len(l.data)
}

func (l lengthSorter) Swap(i, j int) {
	for k := range l.data {
		l.data[k][i], l.data[k][j] = l.data[k][j], l.data[k][i]
	}
}

func (l lengthSorter) Less(i, j int) bool {
	return l.data[i][0][l.yind] < l.data[i][0][l.yind]
}

// Returns the x locations along the surface and the local cf for every data set
func calculateSkinFrictionCoefficient(data [][][]float64, fields map[string]int, rhoInf, uinf float64) (x [][]float64, cfs [][]float64) {
	xind := fields["XLoc"]
	dUdYInd := fields["DUDY"]
	viscInd := fields["Viscosity"]
	x = make([][]float64, len(data))
	cfs = make([][]float64, len(data))
	for i := range data {
		x[i] = make([]float64, len(data[i]))
		cfs[i] = make([]float64, len(data[i]))
	}
	for i := range data {
		for j := range data[i] {
			wallShearStress := data[i][j][dUdYInd] * data[i][j][viscInd]
			cfs[i][j] = wallShearStress / (0.5 * rhoInf * uinf * uinf)
			x[i][j] = data[i][j][xind]
		}
	}
	return x, cfs
}

func plotSkinFrictionCoefficient(xs, cfs [][]float64, title string, labels []string) (*plot.Plot, error) {
	// Create a series of lines
	p, _ := plot.New()
	p.Title.Text = " "
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Cf"
	for i := range xs {
		pts := make(plotter.XYs, len(xs[i]))
		for j := range pts {
			pts[j].X = xs[i][j]
			pts[j].Y = cfs[i][j]
		}
		//plotutil.AddLinePoints(p, labels[i], pts)
		lpLine, lpPoints, err := plotter.NewLinePoints(pts)
		if err != nil {
			return nil, err
		}
		lpLine.Color = colorwheel[i]
		lpPoints.Color = colorwheel[i]
		p.Add(lpLine)
		p.Legend.Add(labels[i], lpLine)

	}
	p.Legend.Top = true
	return p, nil
}

func plotProfileData(profileData [][][][]float64, xLoc []float64, fields map[string]int, labels []string, baseloc string, uInf float64) error {

	uVelInd := fields["UVel"]
	vVelInd := fields["VVel"]
	nuhatInd := fields["NuHat"]
	yPlusInd := fields["YPlus"]
	sourceInd := fields["Source"]
	distind := fields["WallDistance"]
	nuind := fields["Viscosity"]
	kinviscind := fields["Nu"]
	yind := fields["YLoc"]
	productionInd := fields["Production"]
	destructionInd := fields["Destruction"]
	crossproductionInd := fields["CrossProduction"]
	nondimProductionInd := fields["NondimProduction"]
	nondimDestructionInd := fields["NondimDestruction"]
	nondimDestructionModInd := fields["NondimDestructionMod"]
	nondimCrossproductionInd := fields["NondimCrossProduction"]
	nondimSourceModInd := fields["NondimSourceMod"]

	// initialize data
	yData := make([][][]float64, len(profileData))
	xData := make([][][]float64, len(profileData))
	logYPlusData := make([][][]float64, len(profileData))
	// loop over x locations
	for i := range profileData {
		xData[i] = make([][]float64, len(profileData[i]))
		yData[i] = make([][]float64, len(profileData[i]))
		logYPlusData[i] = make([][]float64, len(profileData[i]))
		// loop over data sets
		for j := range profileData[i] {
			xData[i][j] = make([]float64, len(profileData[i][j]))
			yData[i][j] = make([]float64, len(profileData[i][j]))
			logYPlusData[i][j] = make([]float64, len(profileData[i][j]))
			for k := range profileData[i][j] {
				yData[i][j][k] = profileData[i][j][k][distind]
				//fmt.Println("y+ = ", profileData[i][j][k][yPlusInd])
				logYPlusData[i][j][k] = math.Log10(profileData[i][j][k][yPlusInd])
			}
		}
	}

	// plot u vel
	ufunc := func(this []float64, base []float64) float64 { return this[uVelInd] / uInf }
	setData(profileData, ufunc, xData)
	err := MakeProfilePlot(xLoc, xData, yData, "Normalized X - Velocity Profile", "U / U∞", "Y", labels, baseloc, "uvel", false, false)
	if err != nil {
		return err
	}
	udiff := func(this []float64, base []float64) float64 { return (this[uVelInd] - base[uVelInd]) / uInf }
	setData(profileData, udiff, xData)
	err = MakeProfilePlot(xLoc, xData, yData, "Normalized Difference in X - Velocity Profile", "Difference in U / U∞", "Y", labels, baseloc, "udiff", false, false)
	if err != nil {
		return err
	}

	uplusfunc := func(this []float64, base []float64) float64 {
		return this[uVelInd] * this[yind] / (this[kinviscind] * this[yPlusInd])
	}
	setData(profileData, uplusfunc, xData)
	err = MakeProfilePlot(xLoc, logYPlusData, xData, " ", "Log Y+", "U+", labels, baseloc, "uplus_vs_y", false, false)
	if err != nil {
		return err
	}
	err = MakeProfilePlot(xLoc, logYPlusData, xData, " ", "log10(Y+)", "U+", labels, baseloc, "uplus_vs_y_special", false, false)
	if err != nil {
		return err
	}

	vfunc := func(this []float64, base []float64) float64 { return this[vVelInd] / uInf }
	setData(profileData, vfunc, xData)
	err = MakeProfilePlot(xLoc, logYPlusData, xData, " ", "Log Y+", "V+", labels, baseloc, "vplus_vs_y", true, true)
	if err != nil {
		return err
	}

	vdiff := func(this []float64, base []float64) float64 { return (this[vVelInd] - base[vVelInd]) / uInf }
	setData(profileData, vdiff, xData)
	err = MakeProfilePlot(xLoc, xData, yData, "Normalized Difference in Y - Velocity Profile", "Difference in V / U∞", "Y", labels, baseloc, "vdiff", false, false)
	if err != nil {
		return err
	}
	vplusfunc := func(this []float64, base []float64) float64 {
		return this[vVelInd] * this[yind] / (this[kinviscind] * this[yPlusInd])
	}
	setData(profileData, vplusfunc, xData)
	err = MakeProfilePlot(xLoc, xData, yData, "Normalized Y - Velocity Profile", "V / U∞", "Y", labels, baseloc, "vvel", false, false)
	if err != nil {
		return err
	}

	production := func(this []float64, base []float64) float64 { return this[productionInd] }
	setData(profileData, production, xData)
	err = MakeProfilePlot(xLoc, xData, yData, " ", "Production", "Y", labels, baseloc, "production", false, false)
	if err != nil {
		return err
	}
	err = MakeProfilePlot(xLoc, logYPlusData, xData, " ", "Log Y+", "Production", labels, baseloc, "production_vs_yplus", false, false)
	if err != nil {
		return err
	}

	nondimproduction := func(this []float64, base []float64) float64 { return this[nondimProductionInd] }
	setData(profileData, nondimproduction, xData)
	err = MakeProfilePlot(xLoc, xData, yData, " ", "Nondimensional Production", "Y", labels, baseloc, "nondimproduction", false, false)
	if err != nil {
		return err
	}
	err = MakeProfilePlot(xLoc, logYPlusData, xData, " ", "Log Y+", "Nondimensional Production", labels, baseloc, "nondimproduction_vs_yplus", true, true)
	if err != nil {
		return err
	}

	destruction := func(this []float64, base []float64) float64 { return this[destructionInd] }
	setData(profileData, destruction, xData)
	err = MakeProfilePlot(xLoc, xData, yData, " ", "Destruction", "Y", labels, baseloc, "destruction", false, false)
	if err != nil {
		return err
	}
	err = MakeProfilePlot(xLoc, logYPlusData, xData, " ", "Log Y+", "Destruction", labels, baseloc, "destruction_vs_yplus", false, false)
	if err != nil {
		return err
	}

	nondimdestruction := func(this []float64, base []float64) float64 { return this[nondimDestructionInd] }
	setData(profileData, nondimdestruction, xData)
	err = MakeProfilePlot(xLoc, xData, yData, " ", "Nondimensional Destruction", "Y", labels, baseloc, "nondimdestruction", false, false)
	if err != nil {
		return err
	}
	err = MakeProfilePlot(xLoc, logYPlusData, xData, " ", "Log Y+", "Nondimensional Destruction", labels, baseloc, "nondimdestruction_vs_yplus", true, true)
	if err != nil {
		return err
	}

	nondimdestructionmod := func(this []float64, base []float64) float64 { return this[nondimDestructionModInd] }
	setData(profileData, nondimdestructionmod, xData)
	err = MakeProfilePlot(xLoc, xData, yData, " ", "Nondimensional Destruction", "Y", labels, baseloc, "nondimdestructionmod", false, false)
	if err != nil {
		return err
	}
	err = MakeProfilePlot(xLoc, logYPlusData, xData, " ", "Log Y+", "Nondimensional Destruction", labels, baseloc, "nondimdestructionmod_vs_yplus", false, false)
	if err != nil {
		return err
	}

	crossproduction := func(this []float64, base []float64) float64 { return this[crossproductionInd] }
	setData(profileData, crossproduction, xData)
	err = MakeProfilePlot(xLoc, xData, yData, " ", "Cross Production", "Y", labels, baseloc, "crossproduction", false, false)
	if err != nil {
		return err
	}
	err = MakeProfilePlot(xLoc, logYPlusData, xData, " ", "Log Y+", "Cross Production", labels, baseloc, "crossproduction_vs_yplus", true, true)
	if err != nil {
		return err
	}

	nondimcrossproduction := func(this []float64, base []float64) float64 { return this[nondimCrossproductionInd] }
	setData(profileData, nondimcrossproduction, xData)
	err = MakeProfilePlot(xLoc, xData, yData, " ", "Nondimensional Cross Production", "Y", labels, baseloc, "nondimcrossproduction", false, false)
	if err != nil {
		return err
	}
	err = MakeProfilePlot(xLoc, logYPlusData, xData, " ", "Log Y+", "Nondimensional Cross Production", labels, baseloc, "nondimcrossproduction_vs_yplus", true, true)
	if err != nil {
		return err
	}

	source := func(this []float64, base []float64) float64 { return this[sourceInd] }
	setData(profileData, source, xData)
	err = MakeProfilePlot(xLoc, xData, yData, "Source Term", "Source", "Y", labels, baseloc, "source", false, false)
	if err != nil {
		return err
	}
	err = MakeProfilePlot(xLoc, xData, logYPlusData, "Source Term", "Source", "Log Y+", labels, baseloc, "source_vs_yplus", false, false)
	if err != nil {
		return err
	}
	sourcediff := func(this []float64, base []float64) float64 { return (this[sourceInd] - base[sourceInd]) }
	setData(profileData, sourcediff, xData)
	err = MakeProfilePlot(xLoc, xData, yData, "Difference in Source Term", "Difference in Source", "Y", labels, baseloc, "sourcediff", false, false)
	if err != nil {
		return err
	}
	nondimsource := func(this []float64, base []float64) float64 {
		return this[sourceInd] * this[yind] * this[yind] / (this[nuhatInd] * this[nuhatInd])
	}
	setData(profileData, nondimsource, xData)
	err = MakeProfilePlot(xLoc, xData, yData, "Non-Dimensionalized Source Term", "Non-Dimensionalized Source", "Y", labels, baseloc, "source_nondim", false, false)
	if err != nil {
		return err
	}
	err = MakeProfilePlot(xLoc, logYPlusData, xData, " ", "Log Y+", "Non-Dimensionalized Source Term", labels, baseloc, "nondimsource_vs_yplus", true, true)
	if err != nil {
		return err
	}

	nondimsourcemod := func(this []float64, base []float64) float64 { return this[nondimSourceModInd] }
	setData(profileData, nondimsourcemod, xData)
	err = MakeProfilePlot(xLoc, xData, yData, " ", "Nondimensional Source", "Y", labels, baseloc, "nondimsourcemod", true, true)
	if err != nil {
		return err
	}
	err = MakeProfilePlot(xLoc, logYPlusData, xData, " ", "Log Y+", "Nondimensional Source", labels, baseloc, "nondimsourcemod_vs_yplus", false, false)
	if err != nil {
		return err
	}

	nuhat := func(this []float64, base []float64) float64 { return this[nuhatInd] }
	setData(profileData, nuhat, xData)
	err = MakeProfilePlot(xLoc, xData, yData, " ", "Nu Tilde", "Y", labels, baseloc, "nutilde", false, false)
	if err != nil {
		return err
	}
	nuhatdiff := func(this []float64, base []float64) float64 { return (this[nuhatInd] - base[nuhatInd]) }
	setData(profileData, nuhatdiff, xData)
	err = MakeProfilePlot(xLoc, xData, yData, "Difference in Nu Tilde", "Difference in Nu Tilde", "Y", labels, baseloc, "nudiff", false, false)
	if err != nil {
		return err
	}

	nondimer := func(this []float64, base []float64) float64 {
		nuPnuHat := this[nuhatInd] + this[nuind]
		d := this[distind]
		return nuPnuHat * nuPnuHat / (d * d)
	}
	setData(profileData, nondimer, xData)
	err = MakeProfilePlot(xLoc, xData, yData, " ", "Source non-dimer", "Y", labels, baseloc, "sourcenondimer", false, false)
	if err != nil {
		return err
	}
	err = MakeProfilePlot(xLoc, logYPlusData, xData, " ", "Log Y+", "Source non-dimer", labels, baseloc, "sourcenondimer_vs_yplus", false, false)
	if err != nil {
		return err
	}

	invnondimer := func(this []float64, base []float64) float64 {
		return 1 / nondimer(this, base)
	}
	setData(profileData, invnondimer, xData)
	err = MakeProfilePlot(xLoc, xData, yData, " ", "Inv Source non-dimer", "Y", labels, baseloc, "invsourcenondimer", false, false)
	if err != nil {
		return err
	}
	err = MakeProfilePlot(xLoc, logYPlusData, xData, " ", "Log Y+", "Inv Source non-dimer", labels, baseloc, "invsourcenondimer_vs_yplus", false, false)
	if err != nil {
		return err
	}

	dist := func(this []float64, base []float64) float64 {
		return this[distind]
	}
	setData(profileData, dist, xData)
	err = MakeProfilePlot(xLoc, xData, yData, " ", "Wall distance", "Y", labels, baseloc, "walldist", false, false)
	if err != nil {
		return err
	}
	err = MakeProfilePlot(xLoc, logYPlusData, xData, " ", "Log Y+", "Wall Dist", labels, baseloc, "walldist_vs_yplus", false, false)
	if err != nil {
		return err
	}
	nuPNuTilde := func(this []float64, base []float64) float64 {
		return this[nuind] + this[nuhatInd]
	}
	setData(profileData, nuPNuTilde, xData)
	err = MakeProfilePlot(xLoc, xData, yData, " ", "Nu plus Nu Tilde", "Y", labels, baseloc, "nu_p_nut", false, false)
	if err != nil {
		return err
	}
	err = MakeProfilePlot(xLoc, logYPlusData, xData, " ", "Log Y+", "Nu plus Nu Tilde", labels, baseloc, "nu_p_nut_vs_yplus", false, false)
	if err != nil {
		return err
	}
	nu := func(this []float64, base []float64) float64 {
		return this[nuind]
	}
	setData(profileData, nu, xData)
	err = MakeProfilePlot(xLoc, xData, yData, " ", "Nu", "Y", labels, baseloc, "nu", false, false)
	if err != nil {
		return err
	}
	err = MakeProfilePlot(xLoc, logYPlusData, xData, " ", "Log Y+", "Nu", labels, baseloc, "nu_vs_yplus", false, false)
	if err != nil {
		return err
	}

	return nil
}

func setData(profileData [][][][]float64, f func([]float64, []float64) float64, outputData [][][]float64) {
	// loop over x locations
	for i := range profileData {
		// loop over datasets
		for j := range profileData[i] {
			// loop over points in the dataset
			for k := range profileData[i][j] {
				outputData[i][j][k] = f(profileData[i][j][k], profileData[i][0][k])
			}
		}
	}
	return
}

func MakeProfilePlot(xLoc []float64, xData [][][]float64, yData [][][]float64, title, xLabel, ylabel string, legendLables []string, fileloc string, name string, legendTop, legendLeft bool) error {
	// Range over the x locations
	for i := range xLoc {
		p, _ := plot.New()
		p.Title.Text = title
		p.X.Label.Text = xLabel
		p.Y.Label.Text = ylabel

		xstr := strconv.FormatFloat(xLoc[i], 'g', 4, 64)
		xstr = strings.Replace(xstr, ".", "_", -1)

		minX := math.Inf(1)
		maxX := math.Inf(-1)
		minY := math.Inf(1)
		maxY := math.Inf(-1)
		// Range over the different data sets to plot
		for j, set := range xData[i] {
			pts := make(plotter.XYs, len(set))
			//ptsLog := make(plotter.XYs, len(set))
			// Range over points in that plot
			for k := range set {
				//fmt.Println(name)
				//fmt.Println("x = ", xData[i][j][k])
				//fmt.Println("y = ", yData[i][j][k])
				pts[k].X = xData[i][j][k]
				pts[k].Y = yData[i][j][k]
				if pts[k].X < minX {
					minX = pts[k].X
				}
				if pts[k].X > maxX {
					maxX = pts[k].X
				}
				if pts[k].Y < minY {
					minY = pts[k].Y
				}
				if pts[k].Y > maxY {
					maxY = pts[k].Y
				}
			}
			//ppts := profilePts(pts)
			//sort.Sort(ppts)
			//pts = plotter.XYs(ppts)

			if j == 0 {
				lpLine, err := plotter.NewLine(pts)
				if err != nil {
					panic(err)
				}
				lpLine.Color = colorwheel[j]
				p.Add(lpLine)
				//p.Legend.Add(legendLables[j], lpLine)
				p.Legend.Add(legendLables[j], lpLine)

			} else if j == 1 {
				scat, err := plotter.NewScatter(pts)
				if err != nil {
					panic(err)
				}
				scat.Color = colorwheel[j]
				p.Add(scat)
				p.Legend.Add(legendLables[j], scat)
			}
			//lpLine.Color = colorwheel[j]
			//lpPoints.Color = colorwheel[j]
			//p.Add(lpLine, lpPoints)
			//p.Legend.Add(legendLables[j], lpLine, lpPoints)
		}
		p.X.Min = minX
		p.X.Max = maxX
		p.Y.Min = minY
		p.Y.Max = maxY
		// HACK
		if name == "uplus_vs_y_special" {
			uplusEqualsYPlus := plotter.NewFunction(func(x float64) float64 { return math.Pow(10, x) })
			//fmt.Println("Len color wheel = ", len(colorwheel))
			//fmt.Println("Len xData = ", len(xData[0]))
			uplusEqualsYPlus.Color = colorwheel[len(xData[0])]
			p.Add(uplusEqualsYPlus)
			p.Legend.Add("U+ = Y+", uplusEqualsYPlus)

			logLayer := plotter.NewFunction(func(x float64) float64 {
				yp := math.Pow(10, x)
				logyp := math.Log(yp)
				return 1/0.41*logyp + 5.0
			})
			logLayer.Color = colorwheel[len(xData[0])+1]
			p.Add(logLayer)
			p.Legend.Add("U+ = 1/0.41 * ln(Y+) + 5.0", logLayer)
		}
		if xLabel == "Log Y+" || xLabel == "log10(Y+)" {
			//fmt.Println("x label = ", xLabel)
			//fmt.Println("in pnominal thing")
			prewidth := p.X.Width
			pretickwidth := p.X.Tick.Width
			preticklength := p.X.Tick.Length
			p.NominalX("1", "10", "100", "1000")
			p.X.Label.Text = "Y+"
			p.X.Width = prewidth
			p.X.Tick.Width = pretickwidth
			p.X.Tick.Length = preticklength
		}

		p.Legend.Top = legendTop
		p.Legend.Left = legendLeft
		if err := p.Save(4, 4, filepath.Join(fileloc, name+"_"+xstr+".pdf")); err != nil {
			return err
		}
	}
	return nil
}

type profilePts plotter.XYs

func (p profilePts) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}

func (p profilePts) Less(i, j int) bool {
	if p[i].Y < p[j].Y {
		return true
	}
	return false
}

func (p profilePts) Len() int {
	return len(p)
}
