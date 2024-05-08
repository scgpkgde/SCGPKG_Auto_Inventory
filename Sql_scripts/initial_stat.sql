SELECT 
	ini.mat_number,
	RTRIM(LTRIM(ini.grade)) AS grade,
	ini.gram,
	ini.dp_date,
	SUM(COALESCE(demand.ton,ini.ton)) AS ton
FROM (
	SELECT DISTINCT *
	FROM skic_dbHistory.dbo.auto_inventory_date_initial ini_sub
) ini

LEFT OUTER JOIN  
(
	SELECT
		mat_number,
		grade,
		gram,
		dp_date,
		SUM(ton) AS ton
	FROM skic_dbHistory.dbo.auto_inventory_demands
	GROUP BY mat_number,
		grade,
		gram,
		dp_date
) demand
ON ini.mat_number = demand.mat_number
AND ini.grade = demand.grade
AND ini.gram = demand.gram
AND ini.dp_date = demand.dp_date

WHERE ini.dp_date BETWEEN '%s' AND'%s'

GROUP BY ini.mat_number,
	ini.grade,
	ini.gram,
	ini.dp_date