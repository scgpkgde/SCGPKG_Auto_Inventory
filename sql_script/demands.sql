--====================================================================================================================--
-- Project : Auto-Inventory
-- Description : Demand information
--====================================================================================================================-- 
--version : 1.0.0
--date : 21/11/2022
--description 
	-- 1. Replace 'F' in "material number" 
	-- 2. Replace "gram" of 'KT' 
--====================================================================================================================-- 
WITH dp_billing_main AS
( 
	SELECT 
		dpbilling.year AS data_year,
		dpbilling.month AS data_month,
		dpbilling.do_no,
		CASE 
			WHEN dpbilling.[sales group] in ('756','75P') THEN 'Export' 
			ELSE 'Domestic'
		END AS doex,
		dpbilling.[Mat Number] AS mat_number, 
		dpbilling.customer_code,
		SUBSTRING(dpbilling.[Product Hierachy], 5, 3) AS product, 
		CASE 
			WHEN RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3)) = 'TAW' THEN 'TA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'CSF' THEN 'CS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KAF' THEN 'KA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KAH' THEN 'KA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'WSG' THEN 'WS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) IN ('CAF','KSF') THEN REPLACE(SUBSTRING(dpbilling.[Product Hierachy], 8, 3),'F','')
			WHEN LTRIM(RTRIM((SUBSTRING(dpbilling.[Product Hierachy], 8, 3)))) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) <> 125 THEN 'TS'
			WHEN LTRIM(RTRIM((SUBSTRING(dpbilling.[Product Hierachy], 8, 3)))) = 'KT ' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) <> 125 THEN 'TS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KTB' THEN 'TS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'CAG' THEN 'CA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KSD' THEN 'KS'
			ELSE SUBSTRING(dpbilling.[Product Hierachy], 8, 3)
		END  AS grade, 
		CASE 
			WHEN LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3))) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 150 THEN '140'
			WHEN LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3))) = 'KT ' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 150 THEN '140'
			WHEN LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3))) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 175 THEN '165'
			WHEN LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3))) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 200 THEN '185'
			WHEN LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3))) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 250 THEN '235'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KTB' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 150 THEN '140'
			ELSE LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 11, 3)))
		END AS gram,  
		COALESCE(dpbilling.dp_date,dpbilling.billing_date) AS dp_date,
		dpbilling.IND_CODE AS industry_code,
		dpbilling.[sales group] AS sales_group_code,
		COALESCE(dpbilling.w_ton,0) AS weight_ton
	FROM dbo.Tang_dp_billing2021 dpbilling
	
	WHERE dpbilling.[Billing Type] = 'zf2'
    -- AND   dpbilling.dp_date IS NOT NULL
	-- AND   SUBSTRING(dpbilling.[Mat Number], 10, 1)  NOT IN ('M','P','E')
	AND   (
		(SUBSTRING(dpbilling.[Mat Number], 10, 1) IN ('D','W') AND SUBSTRING(dpbilling.[Product Hierachy], 5, 3) IN ('CP ', 'OTH', 'CM ', 'KLB', 'SK '))
		OR 
		dpbilling.IND_CODE in ('1007','1008')
	)

	UNION ALL 
	
	SELECT 
		dpbilling.year AS data_year,
		dpbilling.month AS data_month,
		dpbilling.do_no,
		CASE 
			WHEN dpbilling.[sales group] in ('756','75P') THEN 'Export' 
			ELSE 'Domestic'
		END AS doex,
		dpbilling.[Mat Number] AS mat_number, 
		dpbilling.customer_code,
		SUBSTRING(dpbilling.[Product Hierachy], 5, 3) AS product, 
		CASE 
			WHEN RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3)) = 'TAW' THEN 'TA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'CSF' THEN 'CS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KAF' THEN 'KA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KAH' THEN 'KA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'WSG' THEN 'WS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) IN ('CAF','KSF') THEN REPLACE(SUBSTRING(dpbilling.[Product Hierachy], 8, 3),'F','')
			WHEN LTRIM(RTRIM((SUBSTRING(dpbilling.[Product Hierachy], 8, 3)))) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) <> 125 THEN 'TS'
			WHEN LTRIM(RTRIM((SUBSTRING(dpbilling.[Product Hierachy], 8, 3)))) = 'KT ' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) <> 125 THEN 'TS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KTB' THEN 'TS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'CAG' THEN 'CA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KSD' THEN 'KS'
			ELSE SUBSTRING(dpbilling.[Product Hierachy], 8, 3)
		END  AS grade, 
		CASE 
			WHEN LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3))) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 150 THEN '140'
			WHEN LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3))) = 'KT ' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 150 THEN '140'
			WHEN LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3))) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 175 THEN '165'
			WHEN LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3))) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 200 THEN '185'
			WHEN LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3))) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 250 THEN '235'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KTB' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 150 THEN '140'
			ELSE LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 11, 3)))
		END AS gram, 
		COALESCE(dpbilling.dp_date,dpbilling.billing_date) AS dp_date,
		dpbilling.IND_CODE AS industry_code,
		dpbilling.[sales group] AS sales_group_code,
		COALESCE(dpbilling.w_ton,0) AS weight_ton
	FROM dbo.Tang_dp_billing2022 dpbilling
	
	WHERE dpbilling.[Billing Type] = 'zf2'
    -- AND   dpbilling.dp_date IS NOT NULL
	-- AND   SUBSTRING(dpbilling.[Mat Number], 10, 1)  NOT IN ('M','P','E')
	AND   (
		(SUBSTRING(dpbilling.[Mat Number], 10, 1) IN ('D','W') AND SUBSTRING(dpbilling.[Product Hierachy], 5, 3) IN ('CP ', 'OTH', 'CM ', 'KLB', 'SK '))
		OR 
		dpbilling.IND_CODE in ('1007','1008')
	)

	UNION ALL 
	
	SELECT 
		dpbilling.year AS data_year,
		dpbilling.month AS data_month,
		dpbilling.do_no,
		CASE 
			WHEN dpbilling.[sales group] in ('756','75P') THEN 'Export' 
			ELSE 'Domestic'
		END AS doex,
		dpbilling.[Mat Number] AS mat_number, 
		dpbilling.customer_code,
		SUBSTRING(dpbilling.[Product Hierachy], 5, 3) AS product, 
		CASE 
			WHEN RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3)) = 'TAW' THEN 'TA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'CSF' THEN 'CS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KAF' THEN 'KA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KAH' THEN 'KA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'WSG' THEN 'WS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) IN ('CAF','KSF') THEN REPLACE(SUBSTRING(dpbilling.[Product Hierachy], 8, 3),'F','')
			WHEN LTRIM(RTRIM((SUBSTRING(dpbilling.[Product Hierachy], 8, 3)))) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) <> 125 THEN 'TS'
			WHEN LTRIM(RTRIM((SUBSTRING(dpbilling.[Product Hierachy], 8, 3)))) = 'KT ' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) <> 125 THEN 'TS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KTB' THEN 'TS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'CAG' THEN 'CA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KSD' THEN 'KS'
			ELSE SUBSTRING(dpbilling.[Product Hierachy], 8, 3)
		END  AS grade, 
		CASE 
			WHEN LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3))) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 150 THEN '140'
			WHEN LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3))) = 'KT ' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 150 THEN '140'
			WHEN LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3))) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 175 THEN '165'
			WHEN LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3))) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 200 THEN '185'
			WHEN LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3))) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 250 THEN '235'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KTB' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 150 THEN '140'
			ELSE LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 11, 3)))
		END AS gram, 
		COALESCE(dpbilling.dp_date,dpbilling.billing_date) AS dp_date,
		dpbilling.IND_CODE AS industry_code,
		dpbilling.[sales group] AS sales_group_code,
		COALESCE(dpbilling.w_ton,0) AS weight_ton
	FROM dbo.Tang_dp_billing2023 dpbilling
	
	WHERE dpbilling.[Billing Type] = 'zf2'
    -- AND   dpbilling.dp_date IS NOT NULL
	-- AND   SUBSTRING(dpbilling.[Mat Number], 10, 1)  NOT IN ('M','P','E')
	AND   (
		(SUBSTRING(dpbilling.[Mat Number], 10, 1) IN ('D','W') AND SUBSTRING(dpbilling.[Product Hierachy], 5, 3) IN ('CP ', 'OTH', 'CM ', 'KLB', 'SK '))
		OR 
		dpbilling.IND_CODE in ('1007','1008')
	)

		UNION ALL 
	
	SELECT 
		dpbilling.year AS data_year,
		dpbilling.month AS data_month,
		dpbilling.do_no,
		CASE 
			WHEN dpbilling.[sales group] in ('756','75P') THEN 'Export' 
			ELSE 'Domestic'
		END AS doex,
		dpbilling.[Mat Number] AS mat_number, 
		dpbilling.customer_code,
		SUBSTRING(dpbilling.[Product Hierachy], 5, 3) AS product, 
		CASE 
			WHEN RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3)) = 'TAW' THEN 'TA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'CSF' THEN 'CS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KAF' THEN 'KA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KAH' THEN 'KA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'WSG' THEN 'WS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) IN ('CAF','KSF') THEN REPLACE(SUBSTRING(dpbilling.[Product Hierachy], 8, 3),'F','')
			WHEN LTRIM(RTRIM((SUBSTRING(dpbilling.[Product Hierachy], 8, 3)))) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) <> 125 THEN 'TS'
			WHEN LTRIM(RTRIM((SUBSTRING(dpbilling.[Product Hierachy], 8, 3)))) = 'KT ' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) <> 125 THEN 'TS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KTB' THEN 'TS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'CAG' THEN 'CA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KSD' THEN 'KS'
			ELSE SUBSTRING(dpbilling.[Product Hierachy], 8, 3)
		END  AS grade, 
		CASE 
			WHEN LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3))) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 150 THEN '140'
			WHEN LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3))) = 'KT ' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 150 THEN '140'
			WHEN LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3))) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 175 THEN '165'
			WHEN LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3))) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 200 THEN '185'
			WHEN LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3))) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 250 THEN '235'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KTB' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 150 THEN '140'
			ELSE LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 11, 3)))
		END AS gram, 
		COALESCE(dpbilling.dp_date,dpbilling.billing_date) AS dp_date,
		dpbilling.IND_CODE AS industry_code,
		dpbilling.[sales group] AS sales_group_code,
		COALESCE(dpbilling.w_ton,0) AS weight_ton
	FROM dbo.Tang_dp_billing2024 dpbilling
	
	WHERE dpbilling.[Billing Type] = 'zf2'
    -- AND   dpbilling.dp_date IS NOT NULL
	-- AND   SUBSTRING(dpbilling.[Mat Number], 10, 1)  NOT IN ('M','P','E')
	AND   (
		(SUBSTRING(dpbilling.[Mat Number], 10, 1) IN ('D','W') AND SUBSTRING(dpbilling.[Product Hierachy], 5, 3) IN ('CP ', 'OTH', 'CM ', 'KLB', 'SK '))
		OR 
		dpbilling.IND_CODE in ('1007','1008')
	)
	
	
	UNION ALL
	
	SELECT 
		dpbilling.year AS data_year,
		dpbilling.month AS data_month,
		dpbilling.do_no,
		CASE 
			WHEN dpbilling.[sales group] in ('756','75P') THEN 'Export' 
			ELSE 'Domestic'
		END AS doex,
		dpbilling.[Mat Number] AS mat_number, 
		dpbilling.customer_code,
		SUBSTRING(dpbilling.[Product Hierachy], 5, 3) AS product, 
		CASE 
			WHEN RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3)) = 'TAW' THEN 'TA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'CSF' THEN 'CS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KAF' THEN 'KA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KAH' THEN 'KA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'WSG' THEN 'WS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) IN ('CAF','KSF') THEN REPLACE(SUBSTRING(dpbilling.[Product Hierachy], 8, 3),'F','')
			WHEN LTRIM(RTRIM((SUBSTRING(dpbilling.[Product Hierachy], 8, 3)))) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) <> 125 THEN 'TS'
			WHEN LTRIM(RTRIM((SUBSTRING(dpbilling.[Product Hierachy], 8, 3)))) = 'KT ' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) <> 125 THEN 'TS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KTB' THEN 'TS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'CAG' THEN 'CA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KSD' THEN 'KS'
			ELSE SUBSTRING(dpbilling.[Product Hierachy], 8, 3)
		END  AS grade, 
		CASE 
			WHEN LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3))) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 150 THEN '140'
			WHEN LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3))) = 'KT ' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 150 THEN '140'
			WHEN LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3))) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 175 THEN '165'
			WHEN LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3))) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 200 THEN '185'
			WHEN LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 8, 3))) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 250 THEN '235'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KTB' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 150 THEN '140'
			ELSE LTRIM(RTRIM(SUBSTRING(dpbilling.[Product Hierachy], 11, 3)))
		END AS gram, 
		COALESCE(dpbilling.dp_date,dpbilling.billing_date) AS dp_date,
		dpbilling.IND_CODE AS industry_code,
		dpbilling.[sales group] AS sales_group_code,
		COALESCE(dpbilling.w_ton,0) AS weight_ton
	FROM skic_db.dbo.dp_billing dpbilling
	
	WHERE dpbilling.[Billing Type] = 'zf2'
    -- AND   dpbilling.dp_date IS NOT NULL
	-- AND   SUBSTRING(dpbilling.[Mat Number], 10, 1)  NOT IN ('M','P','E')
	AND   (
		(
			SUBSTRING(dpbilling.[Mat Number], 10, 1) IN ('D','W') 
			AND 
			SUBSTRING(dpbilling.[Product Hierachy], 5, 3) IN ('CP ', 'OTH', 'CM ', 'KLB', 'SK ')
		)
		OR 
		dpbilling.IND_CODE in ('1007','1008') 
	)
)

--====================================================================================================================--

SELECT
	dpbilling.data_year, 
	dpbilling.data_month, 
--	salesarea.salesgroupForCB,
	dpbilling.doex,
-- dpbilling.mat_number, --21/11/2022
	CASE 
		WHEN CHARINDEX('F',dpbilling.mat_number) = 6 THEN REPLACE(dpbilling.mat_number,'F','-')
		WHEN dpbilling.Grade = 'TS' AND dpbilling.Gram = '140' AND CHARINDEX('KT-150',dpbilling.mat_number) > 0 THEN REPLACE(dpbilling.mat_number,'KT-150',dpbilling.Grade + '-' + dpbilling.Gram)
		WHEN dpbilling.Grade = 'TS' AND dpbilling.Gram = '140' AND CHARINDEX('KTB150',dpbilling.mat_number) > 0 THEN REPLACE(dpbilling.mat_number,'KTB150',dpbilling.Grade + '-' + dpbilling.Gram)
		WHEN dpbilling.Grade = 'TS' AND dpbilling.Gram = '165' THEN REPLACE(dpbilling.mat_number,'KT-175',dpbilling.Grade + '-' + dpbilling.Gram)
		WHEN dpbilling.Grade = 'TS' AND dpbilling.Gram = '185' THEN REPLACE(dpbilling.mat_number,'KT-200',dpbilling.Grade + '-' + dpbilling.Gram)
		WHEN dpbilling.Grade = 'TS' AND dpbilling.Gram = '235' THEN REPLACE(dpbilling.mat_number,'KT-250',dpbilling.Grade + '-' + dpbilling.Gram)
		WHEN dpbilling.Grade = 'KS' AND dpbilling.Gram = '170' AND CHARINDEX('KSD',dpbilling.mat_number) > 0 THEN REPLACE(dpbilling.mat_number,'KSD170',dpbilling.Grade + '-' + dpbilling.Gram)
		WHEN dpbilling.Grade = 'KS' AND dpbilling.Gram = '140' AND CHARINDEX('KSD',dpbilling.mat_number) > 0 THEN REPLACE(dpbilling.mat_number,'KSD140',dpbilling.Grade + '-' + dpbilling.Gram)
		ELSE dpbilling.mat_number
	END AS mat_number, --date : 21/11/2022
	dpbilling.customer_code,
--	customer.name1, 
	dpbilling.Product, 
	dpbilling.Grade, 
	dpbilling.Gram, 
	dpbilling.dp_date,
	-- DATEDIFF(week , '2021-10-01' , dpbilling.dp_date) AS number_of_week,
	-- DATEDIFF(month , '2021-10-01' , dpbilling.dp_date) AS number_of_month,
	dbo.salesdocheader.[create date], 
	SUM(dpbilling.weight_ton) AS ton
FROM dp_billing_main dpbilling

--======================================================================--

LEFT OUTER JOIN ibpdb.dbo.vw_salesarea salesarea 
ON salesarea.industry = dpbilling.industry_code
AND salesarea.salesgroup =  dpbilling.sales_group_code

--======================================================================--

LEFT OUTER JOIN dbo.salesdocheader 
ON dpbilling.do_no = dbo.salesdocheader.[sales document no]

-- WHERE CAST(dpbilling.dp_date AS DATE) BETWEEN '2021-10-01' AND '2022-09-30'
 
AND COALESCE(dpbilling.mat_number,'') <> ''
AND SUBSTRING(dpbilling.Grade,1,1) <> 'G'
AND  UPPER(salesarea.iplan) IN 
(
	'CB1-SKIC','CB1-TCP',
	'CB2-SKIC','CB2-TCP',
	'CIP-SKIC','CIP-TCP',
	'MAP 1 DO','MAP 2 DO',
	'SACK DO','TO-SKIC','TO-TCP'
)

-- Gypsum--
-- AND SUBSTRING(dpbilling.Grade,1,1) == 'G'
--======================================================================--

GROUP BY 
	dpbilling.data_year, 
	dpbilling.data_month, 
	dpbilling.doex,
	dpbilling.mat_number, 
	dpbilling.customer_code,
	dpbilling.Product, 
	dpbilling.Grade, 
	dpbilling.Gram, 
	dpbilling.dp_date,
	dbo.salesdocheader.[create date]
;
--====================================================================================================================--
