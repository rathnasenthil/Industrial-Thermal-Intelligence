-- Ensures the PostGIS extension is available in the industrial_fire_db database.
-- The postgis/postgis image already ships with PostGIS installed; this simply
-- enables the extension on first database initialization.
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
