package sa

func NondimProduction(chi, omegaBar float64) (nondimProduction float64) {
	//return cb1 * (1 - ft2(chi)) * (omegaBar + fv2(chi)/kappa) * (chi / (chi + 1))
	chiRat := chi / (chi + 1)
	shatBar := omegaBar + chiRat/(kappa*kappa)*fv2(chi)
	return cb1 * (1 - ft2(chi)) * shatBar * chiRat
}

const (
	kappa = 0.41
	cb1   = 0.1355
	cv1   = 7.1
	cv1_3 = cv1 * cv1 * cv1
)

func fv2(chi float64) float64 {
	return 1 - chi/(1+chi*fv1(chi))
}

func fv1(chi float64) float64 {
	chi_3 := chi * chi * chi
	return chi_3 / (chi_3 + cv1_3)
}

func ft2(chi float64) float64 {
	// SU^2 ignores ft2
	return 0
}

func Production(chi, omega, sourceNondim float64) (production float64) {
	return sourceNondim * NondimProduction(chi, omega)
}

/*
func Destruction(chi, omega, sourceNondim float64) (destruction float64) {
	return sourceNondim * NondimDestruction(chi, omega)
}
*/

/*
func NondimDestruction(chi, omega float64) (nondimDestruction float64) {

}

func Fw(chi float64) float64 {

}

func R_Nondim()
*/
