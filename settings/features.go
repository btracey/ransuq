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
	NondimProduction      = "nondim_production"
	Production            = "production"
	NondimProductionLog   = "nondim_production_log"
	Destruction            = "destruction"
	NondimDestruction     = "nondim_destruction"
	NondimCrossProduction = "nondim_crossproduction"
	NondimSource          = "nondim_source"
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
		inputs = []string{"Chi", "OmegaBar", "SourceNondimer"}
		outputs = []string{"Production"}
	case NondimProductionLog:
		inputs = []string{"Chi_Log", "OmegaBar_Log"}
		outputs = []string{"NondimProduction"}
	case NondimDestruction:
		inputs = []string{"Chi", "OmegaBar"}
		outputs = []string{"NondimDestruction"}
	case Destruction:
		inputs = []string{"Chi", "OmegaBar", "SourceNondimer"}
		outputs = []string{"Destruction"}
	case NondimCrossProduction:
		inputs = []string{"Chi", "NuGradMagBar"}
		outputs = []string{"NondimCrossProduction"}
	case NondimSource:
		inputs = []string{"Chi", "OmegaBar", "NuGradMagBar"}
		outputs = []string{"NondimSource"}
	}
	return inputs, outputs, nil
}
