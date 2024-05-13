SELECT
	REPLACE(
	CASE 
		WHEN LEFT(grade_gram,3) IN ('CAF','KSF') THEN REPLACE(grade_gram,'F','')
		WHEN LEFT(grade_gram,3) IN ('KAH')  THEN REPLACE(grade_gram,'H','')
		ELSE grade_gram
	END,' ','') AS grade_gram,
	AVG(ton) AS ton,
	AVG(avg_lt) AS avg_lt,
	AVG(sd_lt) AS sd_lt,
	AVG(ton) AS ton,
	AVG(avg_lt * ton)/ AVG(ton) AS production_lt
FROM ods_scgp_project.scgp_auto_inventory.lead_time
WHERE [year] = %i
AND scd_active = 1
AND LEFT(grade_gram,3) NOT IN ('CAF','KSF','KAH')
AND ton > 0
GROUP BY 
	CASE 
		WHEN LEFT(grade_gram,3) IN ('CAF','KSF') THEN REPLACE(grade_gram,'F','')
		WHEN LEFT(grade_gram,3) IN ('KAH')  THEN REPLACE(grade_gram,'H','')
		ELSE grade_gram
	END