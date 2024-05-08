--====================================================================================================================--
-- Project : Auto-Inventory
-- Description : Demand information
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
		REPLACE
		(
			LTRIM
			(
				CASE 
					WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'TAW' THEN 'TA'
					WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'CSF' THEN 'CS'
					WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KAF' THEN 'KA'
					WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KAH' THEN 'KA'
					WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'WSG' THEN 'WS'
					WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) IN ('CAF','KSF') THEN REPLACE(SUBSTRING(dpbilling.[Product Hierachy], 8, 3),'F','')
					WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) <> 125 THEN 'TS'
					WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KTB' THEN 'TS'
					WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'CAG' THEN 'CA'
					ELSE SUBSTRING(dpbilling.[Product Hierachy], 8, 3)
				END
			),' ',''
		)  AS grade, 
		CASE 
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 150 THEN '140'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 175 THEN '165'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 200 THEN '185'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 250 THEN '235'
			ELSE SUBSTRING(dpbilling.[Product Hierachy], 11, 3)
		END AS gram, 
		COALESCE(dpbilling.dp_date,dpbilling.billing_date) AS dp_date,
		dpbilling.IND_CODE AS industry_code,
		dpbilling.[sales group] AS sales_group_code,
		COALESCE(dpbilling.w_ton,0) AS weight_ton
	FROM dbo.Tang_dp_billing2019 dpbilling
	
	WHERE dpbilling.[Billing Type] = 'zf2'
    -- AND   dpbilling.dp_date IS NOT NULL
	-- AND   SUBSTRING(dpbilling.[Mat Number], 10, 1)  NOT IN ('M','P','E')
	AND   MONTH(COALESCE(dpbilling.dp_date,dpbilling.billing_date)) >= 9 
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
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'TAW' THEN 'TA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'CSF' THEN 'CS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KAF' THEN 'KA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KAH' THEN 'KA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'WSG' THEN 'WS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) IN ('CAF','KSF') THEN REPLACE(SUBSTRING(dpbilling.[Product Hierachy], 8, 3),'F','')
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) <> 125 THEN 'TS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KTB' THEN 'TS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'CAG' THEN 'CA'
			ELSE SUBSTRING(dpbilling.[Product Hierachy], 8, 3)
		END  AS grade, 
		CASE 
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 150 THEN '140'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 175 THEN '165'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 200 THEN '185'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 250 THEN '235'
			ELSE SUBSTRING(dpbilling.[Product Hierachy], 11, 3)
		END AS gram, 
		COALESCE(dpbilling.dp_date,dpbilling.billing_date) AS dp_date,
		dpbilling.IND_CODE AS industry_code,
		dpbilling.[sales group] AS sales_group_code,
		COALESCE(dpbilling.w_ton,0) AS weight_ton
	FROM dbo.Tang_dp_billing2020 dpbilling
	
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
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'TAW' THEN 'TA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'CSF' THEN 'CS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KAF' THEN 'KA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KAH' THEN 'KA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'WSG' THEN 'WS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) IN ('CAF','KSF') THEN REPLACE(SUBSTRING(dpbilling.[Product Hierachy], 8, 3),'F','')
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) <> 125 THEN 'TS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KTB' THEN 'TS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'CAG' THEN 'CA'
			ELSE SUBSTRING(dpbilling.[Product Hierachy], 8, 3)
		END  AS grade, 
		CASE 
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 150 THEN '140'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 175 THEN '165'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 200 THEN '185'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 250 THEN '235'
			ELSE SUBSTRING(dpbilling.[Product Hierachy], 11, 3)
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
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'TAW' THEN 'TA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'CSF' THEN 'CS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KAF' THEN 'KA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KAH' THEN 'KA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'WSG' THEN 'WS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) IN ('CAF','KSF') THEN REPLACE(SUBSTRING(dpbilling.[Product Hierachy], 8, 3),'F','')
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) <> 125 THEN 'TS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KTB' THEN 'TS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'CAG' THEN 'CA'
			ELSE SUBSTRING(dpbilling.[Product Hierachy], 8, 3)
		END  AS grade, 
		CASE 
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 150 THEN '140'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 175 THEN '165'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 200 THEN '185'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 250 THEN '235'
			ELSE SUBSTRING(dpbilling.[Product Hierachy], 11, 3)
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
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'TAW' THEN 'TA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'CSF' THEN 'CS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KAF' THEN 'KA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KAH' THEN 'KA'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'WSG' THEN 'WS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) IN ('CAF','KSF') THEN REPLACE(SUBSTRING(dpbilling.[Product Hierachy], 8, 3),'F','')
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) <> 125 THEN 'TS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KTB' THEN 'TS'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'CAG' THEN 'CA'
			ELSE SUBSTRING(dpbilling.[Product Hierachy], 8, 3)
		END  AS grade, 
		CASE 
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 150 THEN '140'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 175 THEN '165'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 200 THEN '185'
			WHEN SUBSTRING(dpbilling.[Product Hierachy], 8, 3) = 'KT' AND SUBSTRING(dpbilling.[Product Hierachy], 11, 3) = 250 THEN '235'
			ELSE SUBSTRING(dpbilling.[Product Hierachy], 11, 3)
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
	dpbilling.mat_number,
	dpbilling.customer_code,
--	customer.name1, 
	dpbilling.Product, 
	dpbilling.Grade, 
	dpbilling.Gram, 
	dpbilling.dp_date,
	DATEDIFF(week , '%s' , dpbilling.dp_date) AS number_of_week,
	DATEDIFF(month , '%s' , dpbilling.dp_date) AS number_of_month,
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

WHERE CAST(dpbilling.dp_date AS DATE) BETWEEN '%s' AND '%s'
AND COALESCE(dpbilling.mat_number,'') <> ''
-- Kraft -- ทำปุ่ม
AND SUBSTRING(dpbilling.Grade,1,1) == 'G'
-- AND  UPPER(salesarea.iplan) IN 
-- (
-- 	'CB1-SKIC','CB1-TCP',
-- 	'CB2-SKIC','CB2-TCP',
-- 	'CIP-SKIC','CIP-TCP',
-- 	'MAP 1 DO','MAP 2 DO',
-- 	'SACK DO','TO-SKIC','TO-TCP'
-- )

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
