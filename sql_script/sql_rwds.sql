SELECT DISTINCT
    RTRIM(grade) AS Grade  ,
	RTRIM(gram) AS Gram ,
	-- [type],
	SUM(loss)  AS loss
FROM ods_scgp_project.scgp_auto_inventory.rwds
WHERE CAST([year] AS VARCHAR(4)) + '_' +  CAST([month] AS VARCHAR(4)) NOT IN (%s) 
AND CAST([year] AS VARCHAR(4)) + '_' NOT IN (%s)
AND [type] NOT IN ('DS')
-- AND material_std = 'NonSTD'
GROUP BY grade,
	     gram 