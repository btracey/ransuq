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
	NondimProduction         = "nondim_production"
	Production               = "production"
	MulProduction            = "mul_production"
	NondimProductionLog      = "nondim_production_log"
	NondimProductionLogChi   = "nondim_production_logchi"
	Destruction              = "destruction"
	CrossProduction          = "cross_production"
	MulDestruction           = "mul_destruction"
	NondimDestruction        = "nondim_destruction"
	NondimCrossProduction    = "nondim_crossproduction"
	MulCrossproduction       = "mul_crossproduction"
	NondimSource             = "nondim_source"
	NondimSource2            = "nondim_source2"
	Source                   = "source"
	Source2DNS               = "source_2_dns"
	SourceAll                = "source_all"           // Source with all of the variables
	SourceAllDNSFirst        = "source_all_dns_first" // Source with all of the variables
	SourceComputed           = "source_computed"
	SourceOmegaNNondim       = "source_omega_n_nondim"
	FwHiFi                   = "fw_hifi"
	FwHiFi2                  = "fw_hifi_2"
	FwLES2                   = "fw_les_2"
	Fw                       = "fw"
	FwAlt                    = "fw_alt"
	SourceIrrotational       = "source_irrotational"
	NondimSourceIrrotational = "nondim_source_irrotational"
	NondimTurbKinSource      = "nondim_turb_kin_source"
	NondimTurbSpecDissSource = "nondim_turb_spec_diss_source"
	TurbKinEnergySA          = "turb_kin_energy_sa"
	TurbDissipation          = "turb_dissipation"
	ShivajiFw                = "shivaji_fw"
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
	case NondimSource2:
		inputs = []string{"Chi", "OmegaBar", "NuGradMagBar"}
		outputs = []string{"NondimSource2"}
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
	case SourceAllDNSFirst:
		inputs = []string{"Chi", "DNuHatDX", "DNuHatDY", "DUDX", "DUDY", "DVDX", "DVDY", "DPressDX", "DPressDY"}
		outputs = []string{"Source"}
		/*
			case SourceComputed:
				inputs = []string{"SourceNondimer", "Chi", "OmegaBar", "NuGradMagBar"}
				outputs = []string{"Computed_Source"}
		*/
	case Fw:
		inputs = []string{"Chi", "OmegaBar"}
		outputs = []string{"Fw"}
	case FwAlt:
		inputs = []string{"Chi", "OmegaBarAlt"}
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
		/*
			case SourceIrrotational:
				inputs = []string{"SourceNondimerUNorm", "NuGradAngle", "Chi", "NuVelGradNormRatio", "VelVortOverNorm", "VelDetOverNorm"}
				outputs = []string{"Source"}
		*/
	case NondimSourceIrrotational:
		inputs = []string{"Chi", "VelVortOverNorm", "VelDetOverNorm", "NuHatGradMagUNorm"}
		outputs = []string{"NondimSourceUNorm"}
	case NondimTurbKinSource:
		inputs = []string{"Chi", "VelVortOverNorm", "VelDetOverNorm", "VelNormOverSpecDiss"}
		outputs = []string{"NondimTurbKinEnergySource"}
	case NondimTurbSpecDissSource:
		inputs = []string{"Chi", "VelVortOverNorm", "VelDetOverNorm", "VelNormOverSpecDiss"}
		outputs = []string{"NondimTurbSpecificDissipationSource"}
		/*
			case TurbKinEnergy:
				//inputs = []string{"NuGradAngle", "Chi", "NuVelGradNormRatio", "VelVortOverNorm", "VelDetOverNorm"}
				inputs = []string{"VelVortOverNorm", "VelDetOverNorm", "TotalVelGradNorm"}
				outputs = []string{"TurbKinEnergy"}
			case TurbKinEnergySST:
				inputs = []string{"NuGradAngle", "Chi", "NuVelGradNormRatio", "VelVortOverNorm", "VelDetOverNorm"}
				outputs = []string{"TurbKinEnergy"}
			case TurbDissipation:
				inputs = []string{"NuGradAngle", "Chi", "NuVelGradNormRatio", "VelVortOverNorm", "VelDetOverNorm"}
				//inputs = []string{"VelVortOverNorm", "VelDetOverNorm", "TotalVelGradNorm"}
				outputs = []string{"TurbDissipation"}
		*/
	case ShivajiFw:
		inputs = []string{"Chi", "Vorticity", "StrainRate", "StressRatio", "WallDistance"}
		outputs = []string{"Fw"}
	}
	return inputs, outputs, nil
}
