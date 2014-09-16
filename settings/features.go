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
	MulProduction          = "mul_production"
	NondimProductionLog    = "nondim_production_log"
	NondimProductionLogChi = "nondim_production_logchi"
	Destruction            = "destruction"
	CrossProduction        = "cross_production"
	MulDestruction         = "mul_destruction"
	NondimDestruction      = "nondim_destruction"
	NondimCrossProduction  = "nondim_crossproduction"
	MulCrossproduction     = "mul_crossproduction"
	NondimSource           = "nondim_source"
	Source                 = "source"
	Source2DNS             = "source_2_dns"
	SourceAll              = "source_all" // Source with all of the variables
	SourceComputed         = "source_computed"
	SourceOmegaNNondim     = "source_omega_n_nondim"
	FwHiFi                 = "fw_hifi"
	FwHiFi2                = "fw_hifi_2"
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
	case MulProduction: // Multiplier of Omega NuHat
		inputs = []string{"Chi", "OmegaBar"}
		outputs = []string{"MulProduction"}
	case NondimProductionLogChi:
		inputs = []string{"Chi_Log", "OmegaBar"}
		outputs = []string{"NondimProduction"}
	case NondimDestruction:
		inputs = []string{"Chi", "OmegaBar"}
		outputs = []string{"NondimDestruction"}
	case Destruction:
		inputs = []string{"SourceNondimer", "Chi", "OmegaBar"}
		outputs = []string{"Destruction"}
	case MulDestruction:
		inputs = []string{"Chi", "OmegaBar"}
		outputs = []string{"MulDestruction"}
	case NondimCrossProduction:
		inputs = []string{"Chi", "NuGradMagBar"}
		outputs = []string{"NondimCrossProduction"}
	case CrossProduction:
		inputs = []string{"SourceNondimer", "Chi", "NuGradMagBar"}
		outputs = []string{"CrossProduction"}
	case MulCrossproduction:
		inputs = []string{"Chi", "NuGradMagBar"}
		outputs = []string{"MulCrossProduction"}
	case NondimSource:
		inputs = []string{"Chi", "OmegaBar", "NuGradMagBar"}
		outputs = []string{"NondimSource"}
	case Source:
		inputs = []string{"SourceNondimer", "Chi", "OmegaBar", "NuGradMagBar"}
		outputs = []string{"Source"}
	case Source2DNS:
		inputs = []string{"SourceNondimer", "Chi", "OmegaBar"}
		outputs = []string{"Source"}
	case SourceOmegaNNondim:
		inputs = []string{"SourceOmegaNNondim", "Omega_OmegaNNondim", "Chi", "NuGradMag_OmegaNNondim"}
		outputs = []string{"Source"}
	case SourceAll:
		inputs = []string{"SourceNondimer", "Chi", "DNuHatDXBar", "DNuHatDYBar", "DUDXBar", "DUDYBar", "DVDXBar", "DVDYBar"}
		outputs = []string{"Source"}
		/*
			case SourceComputed:
				inputs = []string{"SourceNondimer", "Chi", "OmegaBar", "NuGradMagBar"}
				outputs = []string{"Computed_Source"}
		*/
	case Fw:
		inputs = []string{"Chi", "OmegaBar"}
		outputs = []string{"Fw"}
	case FwHiFi:
		inputs = []string{"Chi", "OmegaBar", "StrainRateMagBar"}
		outputs = []string{"FwRealMinusFwRans"}
	case FwHiFi2:
		inputs = []string{"Chi", "OmegaBar"}
		outputs = []string{"FwRealMinusFwRans"}
	case FwLES2:
		inputs = []string{"Chi", "OmegaBar"}
		outputs = []string{"FwLes2"}
	}
	return inputs, outputs, nil
}
