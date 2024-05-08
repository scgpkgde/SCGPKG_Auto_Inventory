SELECT 
	Grade AS grades,
	swithing_cost AS switching_cost_new
FROM ods_scgp_project.scgp_auto_inventory.switching_cost
WHERE scd_active = 1
AND [year] = %i
;