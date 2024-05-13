SELECT 
	-- [material_no] AS mat_number,
	LTRIM(RTRIM(REPLACE(SUBSTRING([material_no],4,3),'-',''))) AS Grade,
	LTRIM(RTRIM(CAST(SUBSTRING([material_no],7,3) AS INT))) AS Gram,
	(SUM(amount) / SUM(quantity)) * 1000  AS cogs_amt 
FROM vw_auto_inv_cogs 
WHERE CAST([year] AS VARCHAR(4)) + '_' +  CAST([month] AS VARCHAR(4)) NOT IN (%s)
AND  CAST([year] AS VARCHAR(4)) + '_' NOT IN (%s)
GROUP BY 
	-- [material_no],
	REPLACE(SUBSTRING([material_no],4,3),'-','') ,
	SUBSTRING([material_no],7,3);