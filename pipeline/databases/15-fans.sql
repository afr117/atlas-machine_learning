-- Ranks country origins of bands by the number of (non-unique) fans
-- Results display the origin and total number of fans (nb_fans)
SELECT origin, SUM(fans) AS nb_fans
FROM metal_bands
GROUP BY origin
ORDER BY nb_fans DESC;
