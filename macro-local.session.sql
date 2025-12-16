-- Terminate all connections to the database
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE datname = 'macro' 
  AND pid <> pg_backend_pid();

-- Then run your schema