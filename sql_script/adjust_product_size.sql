SELECT 
	RTRIM(mat_number) AS  mat_number,
	UPPER(RTRIM(REPLACE(product_type_new,'size',''))) AS product_type
FROM ods_scgp_project.scgp_auto_inventory.adjust_product_type