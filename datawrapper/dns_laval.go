package datawrapper

import (
	"github.com/btracey/ransuq/dataloader"
)

var Laval = &dataloader.NaiveCSV{
	Delimiter: ",",
	FieldMap: map[string]string{
		"XLoc":  "grid_x",
		"YLoc":  "grid_yx",
		"UVel":  "mean_u_xyz",
		"VVel":  "mean_v_xyz",
		"WVel":  "mean_w_xyz",
		"DUDX":  "dx_mean_u_xyz",
		"DVDX":  "dx_mean_v_xyz",
		"DWDX":  "dx_mean_w_xyz",
		"DUDY":  "dy_mean_u_xyz",
		"DVDY":  "dy_mean_v_xyz",
		"DWDY":  "dy_mean_w_xyz",
		"DUDZ":  "dz_mean_u_xyz",
		"DVDZ":  "dz_mean_v_xyz",
		"DWDZ":  "dz_mean_w_xyz",
		"TauUU": "reynolds_stress_uu_xyz",
		"TauUV": "reynolds_stress_uv_xyz",
		"TauUW": "reynolds_stress_uw_xyz",
		"TauVV": "reynolds_stress_vv_xyz",
		"TauVW": "reynolds_stress_vw_xyz",
		"TauWW": "reynolds_stress_ww_xyz",
	},
}
