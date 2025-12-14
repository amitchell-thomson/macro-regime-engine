-- Check if schema exists
SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'macro';

-- List all tables in macro schema
SELECT table_name FROM information_schema.tables WHERE table_schema = 'macro';
