package ransuq_test

import (
	"testing"

	. "github.com/btracey/ransuq"
	"github.com/btracey/ransuq/settings"
	"github.com/btracey/su2tools/driver"
)

func TestMulti(t *testing.T) {
	set, err := settings.GetSettings(
		"single_flatplate",
		"flatplate_sweep",
		"nondim_production",
		"none",
		"net_2_50",
		settings.StandardTraining,
		driver.Serial{true},
	)
	if err != nil {
		t.Errorf(err)
	}

	err = MultiTurb([]*Settings{set})
	if err != nil {
		t.Error(err)
	}
}
