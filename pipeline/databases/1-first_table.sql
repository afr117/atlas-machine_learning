-- Creates a table called first_table with id (INT) and name (VARCHAR(256))
-- If the table already exists, the script will not fail
CREATE TABLE IF NOT EXISTS first_table (
    id INT,
    name VARCHAR(256)
);
