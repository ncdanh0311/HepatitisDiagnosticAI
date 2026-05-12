-- docker/postgres/init.sql
-- Seed initial roles on DB creation

INSERT INTO roles (name) VALUES ('admin'), ('doctor'), ('researcher')
ON CONFLICT (name) DO NOTHING;
