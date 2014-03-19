package sa

import (
	"fmt"
	"testing"

	"github.com/gonum/floats"
)

var colmap = map[string]int{
	"PointID":                   0,
	"x":                         1,
	"y":                         2,
	"Conservative_1":            3,
	"Conservative_2":            4,
	"Conservative_3":            5,
	"Conservative_4":            6,
	"Conservative_5":            7,
	"Pressure":                  8,
	"Pressure_Coefficient":      9,
	"Mach":                      10,
	"Temperature":               11,
	"Laminar_Viscosity":         12,
	"Skin_Friction_Coefficient": 13,
	"Heat_Transfer":             14,
	"Y_Plus":                    15,
	"Eddy_Viscosity":            16,
	"Sharp_Edge_Dist":           17,
	"Residual_0":                18,
	"SAResidual_0":              19,
	"ResidualDiff_0":            20,
	"NondimResidual_0":          21,
	"SANondimResidual_0":        22,
	"NondimResidualDiff_0":      23,
	"Residual_1":                24,
	"SAResidual_1":              25,
	"ResidualDiff_1":            26,
	"NondimResidual_1":          27,
	"SANondimResidual_1":        28,
	"NondimResidualDiff_1":      29,
	"Residual_2":                30,
	"SAResidual_2":              31,
	"ResidualDiff_2":            32,
	"NondimResidual_2":          33,
	"SANondimResidual_2":        34,
	"NondimResidualDiff_2":      35,
	"Residual_3":                36,
	"SAResidual_3":              37,
	"ResidualDiff_3":            38,
	"NondimResidual_3":          39,
	"SANondimResidual_3":        40,
	"NondimResidualDiff_3":      41,
	"Chi":                42,
	"OmegaBar":           43,
	"DNuHatDXBar_0":      44,
	"DNuHatDXBar_1":      45,
	"NuHatGradNorm":      46,
	"NuHatGradNormBar":   47,
	"KinematicViscosity": 48,
	"NuTilde":            49,
	"WallDist":           50,
	"NuGradNondimer":     51,
	"OmegaNondimer":      52,
	"SourceNondimer":     53,
	"DNuTildeDX_0":       54,
	"DNuTildeDX_1":       55,
	"DU_0DX_0":           56,
	"DU_0DX_1":           57,
	"DU_1DX_0":           58,
	"DU_1DX_1":           59,
}

// Test points taken from SU2 solutions
var testPts = [][]float64{
	// Flatplate Re4 line 12185
	{12183, 1.613373781790000e+00, 3.768177062400002e-01, 1.000000505729188e+00, 2.366399879963204e-01, 3.929545196580346e-04, 2.528001074847583e+00, 1.774580319597089e-07, 1.000000707936629e+00, 2.528345102329307e-05, 1.999974470704609e-01, 1.000000202207338e+00, 5.916080702828719e-08, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.244323895666110e-08, 5.398809279307257e-01, -2.629986313768127e-13, -2.629986313768127e-13, 0.000000000000000e+00, -6.669872979782278e-01, -6.669872979782278e-01, 0.000000000000000e+00, 1.440459278858110e-12, 1.440459278858110e-12, 0.000000000000000e+00, 3.653129437303770e+00, 3.653129437303770e+00, 0.000000000000000e+00, 7.721299043179667e-21, 7.721299043179667e-21, 0.000000000000000e+00, 1.958188283616419e-08, 1.958188283616419e-08, 0.000000000000000e+00, -1.703457902513624e-12, -1.703457902513624e-12, 0.000000000000000e+00, -4.320116715700115e+00, -4.320116715700115e+00, 0.000000000000000e+00, 2.999589265545268e+00, 3.206956479498472e-02, -1.261607545539431e-05, 1.443222752647707e-04, 8.275776037705966e-21, 2.098808449749645e-08, 5.916077710895542e-08, 1.774580319573389e-07, 3.768177062400000e-01, 6.279397309307669e-07, 1.666428409632174e-06, 3.943083056814040e-13, -7.922135026862556e-12, 9.062569069707620e-11, 1.729890685570709e-05, -8.707240416897273e-06, -8.760682050756176e-06, -1.648918970260289e-05},
	{11977, 1.645732411200000e-01, 3.335398386080002e-01, 1.000041160174288e+00, 2.364090884932946e-01, 3.116050559929193e-04, 2.528087588602288e+00, 1.774750671623764e-07, 1.000057624661102e+00, 2.058023610799050e-03, 1.997924523315016e-01, 1.000016463809162e+00, 5.916154667545386e-08, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.244924990863736e-08, 5.992966389497372e-01, -2.914659208146620e-13, -2.914659208146620e-13, 0.000000000000000e+00, -5.790662653720975e-01, -5.790662653720975e-01, 0.000000000000000e+00, 1.838872156539931e-12, 1.838872156539931e-12, 0.000000000000000e+00, 3.653356211278707e+00, 3.653356211278707e+00, 0.000000000000000e+00, 1.182981585560686e-21, 1.182981585560686e-21, 0.000000000000000e+00, 2.350273839356280e-09, 2.350273839356280e-09, 0.000000000000000e+00, -2.130338076171611e-12, -2.130338076171611e-12, 0.000000000000000e+00, -4.232422474300530e+00, -4.232422474300530e+00, 0.000000000000000e+00, 2.999961665000203e+00, 8.981008952979341e-01, -3.279509403716871e-05, 3.799384173846432e-05, 1.267933103494841e-21, 2.519050202954212e-09, 5.915911167610657e-08, 1.774750671637856e-07, 3.335398386080000e-01, 7.094630129566072e-07, 2.127071284550266e-06, 5.033377667534669e-13, -2.326690622580498e-11, 2.695522543356739e-11, 4.459204620284923e-04, 4.043640988442539e-04, 4.062744234692710e-04, -4.292484621635022e-04},
}

func TestNondimProduction(t *testing.T) {
	for i, test := range testPts {
		chi := test[colmap["Chi"]]
		omegaBar := test[colmap["OmegaBar"]]
		fmt.Println("omegaBar", omegaBar, "Chi", chi)
		ans := NondimProduction(chi, omegaBar)
		nondimProduction := test[colmap["NondimResidual_0"]]
		if !floats.EqualWithinAbsOrRel(nondimProduction, ans, 1e-14, 1e-14) {
			t.Errorf("mismatch case %v. Found %v, expected %v", i, ans, nondimProduction)
		}
		sourceNondim := test[colmap["SourceNondimer"]]
		production := test[colmap["Residual_0"]]
		ans = Production(chi, omegaBar, sourceNondim)
		if !floats.EqualWithinAbsOrRel(production, ans, 1e-14, 1e-14) {
			t.Errorf("mismatch production case %v. Found %v, expected %v", i, ans, production)
		}
	}
}
