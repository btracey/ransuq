package settings

import (
	"sort"
)

func init() {
	sortedFeatureset = append(sortedFeatureset, NondimProduction)
	sortedFeatureset = append(sortedFeatureset, NondimDestruction)
	sortedFeatureset = append(sortedFeatureset, NondimCrossProduction)
	sortedFeatureset = append(sortedFeatureset, NondimSource)

	sort.Strings(sortedFeatureset)
}

const (
	NondimProduction       = "nondim_production"
	Production             = "production"
	NondimProductionLog    = "nondim_production_log"
	NondimProductionLogChi = "nondim_production_logchi"
	Destruction            = "destruction"
	CrossProduction        = "cross_production"
	NondimDestruction      = "nondim_destruction"
	NondimCrossProduction  = "nondim_crossproduction"
	NondimSource           = "nondim_source"
	Source                 = "source"
	SourceAll              = "source_all" // Source with all of the variables
	SourceOmegaNNondim     = "source_omega_n_nondim"
	FwHiFi                 = "fw_hifi"
	FwLES2                 = "fw_les_2"
	Fw                     = "fw"
)

var sortedFeatureset []string

func GetFeatures(features string) (inputs, outputs []string, err error) {
	switch features {
	default:
		return nil, nil, Missing{
			Prefix:  "feature setting not found",
			Options: sortedFeatureset,
		}
	case NondimProduction:
		inputs = []string{"Chi", "OmegaBar"}
		outputs = []string{"NondimProduction"}
	case Production:
		inputs = []string{"SourceNondimer", "Chi", "OmegaBar"}
		outputs = []string{"Production"}
	case NondimProductionLog:
		inputs = []string{"Chi_Log", "OmegaBar_Log"}
		outputs = []string{"NondimProduction"}
	case NondimProductionLogChi:
		inputs = []string{"Chi_Log", "OmegaBar"}
		outputs = []string{"NondimProduction"}
	case NondimDestruction:
		inputs = []string{"Chi", "OmegaBar"}
		outputs = []string{"NondimDestruction"}
	case Destruction:
		inputs = []string{"SourceNondimer", "Chi", "OmegaBar"}
		outputs = []string{"Destruction"}
	case NondimCrossProduction:
		inputs = []string{"Chi", "NuGradMagBar"}
		outputs = []string{"NondimCrossProduction"}
	case CrossProduction:
		inputs = []string{"SourceNondimer", "Chi", "NuGradMagBar"}
		outputs = []string{"CrossProduction"}
	case NondimSource:
		inputs = []string{"Chi", "OmegaBar", "NuGradMagBar"}
		outputs = []string{"NondimSource"}
	case Source:
		inputs = []string{"SourceNondimer", "Chi", "OmegaBar", "NuGradMagBar"}
		outputs = []string{"Source"}
	case SourceOmegaNNondim:
		inputs = []string{"SourceOmegaNNondim", "Omega_OmegaNNondim", "Chi", "NuGradMag_OmegaNNondim"}
		outputs = []string{"Source"}
	case SourceAll:
		inputs = []string{"SourceNondimer", "Chi", "DNuHatDXBar", "DNuHatDYBar", "DUDXBar", "DUDYBar", "DVDXBar", "DVDYBar"}
		outputs = []string{"Source"}
	case FwHiFi:
		inputs = []string{"Chi", "OmegaBar", "StrainRateMagBar"}
		outputs = []string{"FwRealMinusFwRans"}
	case FwLES2:
		inputs = []string{"Chi", "OmegaBar"}
		outputs = []string{"Fw"}
	}
	return inputs, outputs, nil
}
