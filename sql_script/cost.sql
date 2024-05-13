SELECT 
	CTSSKU AS grade_gram,
	CASE WHEN SUM(COALESCE(SalesVol_T,0)) <> 0 THEN
		SUM(
			COALESCE([vcRM_T_B],0) +  
			COALESCE([vcChem_T_B],0) + 
			COALESCE([vcUTL_T_B],0) + 
			COALESCE([vcASG_T_B],0) + 
			COALESCE([vcContractor_T_B],0) + 
			COALESCE([vcOvertime_T_B],0) +
			COALESCE([vcReprocess_T_B],0)
		) / SUM(COALESCE(SalesVol_T,0)) 
	ELSE 0
	END AS vc_per_ton
FROM SC_CTS_SalesDPBilling
WHERE CTSSKU IS NOT NULL
AND CAST([Year] AS VARCHAR(4)) + '_' + LTRIM(Period) IN (%s)
GROUP BY CTSSKU