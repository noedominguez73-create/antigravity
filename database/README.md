# Database Setup Guide - asesoriaimss.io

## Overview
This directory contains all database-related files for the asesoriaimss.io project.

## Files
- `schema.sql` - Complete database schema with all tables, indexes, and constraints
- `seed_data.sql` - Realistic test data for development and testing

## Prerequisites
- PostgreSQL 12+ installed
- psql command-line tool
- Superuser or database creation privileges

## Database Setup Instructions

### 1. Create Database

```bash
# Connect to PostgreSQL as superuser
psql -U postgres

# Create database
CREATE DATABASE asesoriaimss;

# Create user (optional)
CREATE USER asesoriaimss_user WITH PASSWORD 'your_secure_password';

# Grant privileges
GRANT ALL PRIVILEGES ON DATABASE asesoriaimss TO asesoriaimss_user;

# Exit psql
\q
```

### 2. Execute Schema

```bash
# Connect to the database
psql -U postgres -d asesoriaimss

# Or with custom user
psql -U asesoriaimss_user -d asesoriaimss

# Execute schema file
\i C:/asesoriaimss.io/database/schema.sql

# Verify tables were created
\dt

# Exit
\q
```

**Alternative (from command line):**
```bash
psql -U postgres -d asesoriaimss -f C:/asesoriaimss.io/database/schema.sql
```

### 3. Load Seed Data (Optional - for development/testing)

```bash
# Connect to database
psql -U postgres -d asesoriaimss

# Execute seed data file
\i C:/asesoriaimss.io/database/seed_data.sql

# Verify data was loaded
SELECT COUNT(*) FROM users;
SELECT COUNT(*) FROM professionals;
SELECT COUNT(*) FROM comments;

# Exit
\q
```

**Alternative (from command line):**
```bash
psql -U postgres -d asesoriaimss -f C:/asesoriaimss.io/database/seed_data.sql
```

### 4. Update Flask Configuration

Update your `.env` file or `app/__init__.py` with the correct database connection string:

```python
# Format: postgresql://username:password@host:port/database
DATABASE_URL=postgresql://asesoriaimss_user:your_secure_password@localhost:5432/asesoriaimss
```

## Database Structure

### Core Tables
- `users` - User accounts (regular users, professionals, admins)
- `professionals` - Professional profiles with ratings and specialties
- `categories` - Inquiry categories
- `inquiries` - General inquiries from users
- `responses` - Responses to inquiries

### Professional Features
- `services` - Services offered by professionals
- `experiences` - Work experience entries
- `certifications` - Professional certifications
- `comments` - User reviews and ratings

### Credit System
- `credits` - Credit purchases and transactions
- `chat_messages` - Chatbot conversation history
- `chatbot_configs` - Chatbot configuration per professional

### Referral System
- `referrals` - Referral codes and tracking
- `referral_earnings` - Commission earnings
- `referral_withdrawals` - Withdrawal requests

## Seed Data Summary

The `seed_data.sql` file includes:
- **1 Admin User** - admin@asesoriaimss.io (Password: Admin123!)
- **50 Regular Users** - Realistic Mexican names and emails
- **21 Professional Users** - Sample across specialties (Abogados, Contadores, Arquitectos)
- **21 Professional Profiles** - With ratings, specialties, and cities
- **40+ Services** - Various services offered by professionals
- **30+ Work Experiences** - Professional backgrounds
- **10+ Certifications** - Professional credentials
- **30+ Comments** - User reviews (all approved)
- **3 Chatbot Configs** - Sample chatbot configurations
- **12 Chat Messages** - Sample conversations
- **6 Credit Transactions** - Sample purchases
- **3 Referrals** - Sample referral codes
- **4 Inquiries** - General questions
- **3 Responses** - Answers to inquiries

### Specialties Included (Sample)
- Abogado (Lawyer)
- Contador (Accountant)
- Arquitecto (Architect)

### Cities Included (Sample)
- Ciudad de México
- Guadalajara
- Monterrey
- Puebla
- Tijuana
- León
- Querétaro
- Mérida
- Cancún
- Aguascalientes
- Toluca
- San Luis Potosí
- Morelia
- Cuernavaca
- Playa del Carmen

**Note:** The seed data file is condensed for brevity. In production, expand to include all 150 professionals across 22 specialties and 30 cities as specified.

## Backup and Restore

### Create Backup

```bash
# Full database backup
pg_dump -U postgres -d asesoriaimss -F c -f asesoriaimss_backup_$(date +%Y%m%d).dump

# SQL format backup
pg_dump -U postgres -d asesoriaimss -f asesoriaimss_backup_$(date +%Y%m%d).sql

# Backup specific tables
pg_dump -U postgres -d asesoriaimss -t users -t professionals -f users_professionals_backup.sql
```

### Restore from Backup

```bash
# Restore from custom format (.dump)
pg_restore -U postgres -d asesoriaimss -c asesoriaimss_backup_20250125.dump

# Restore from SQL format
psql -U postgres -d asesoriaimss -f asesoriaimss_backup_20250125.sql
```

### Automated Backups (Linux/Mac)

Create a cron job for daily backups:

```bash
# Edit crontab
crontab -e

# Add daily backup at 2 AM
0 2 * * * pg_dump -U postgres -d asesoriaimss -F c -f /backups/asesoriaimss_$(date +\%Y\%m\%d).dump

# Keep only last 7 days
0 3 * * * find /backups -name "asesoriaimss_*.dump" -mtime +7 -delete
```

### Automated Backups (Windows)

Create a batch script `backup_db.bat`:

```batch
@echo off
set PGPASSWORD=your_password
set BACKUP_DIR=C:\backups
set TIMESTAMP=%date:~-4%%date:~-7,2%%date:~-10,2%

pg_dump -U postgres -d asesoriaimss -F c -f %BACKUP_DIR%\asesoriaimss_%TIMESTAMP%.dump

echo Backup completed: asesoriaimss_%TIMESTAMP%.dump
```

Schedule with Windows Task Scheduler to run daily.

## Database Maintenance

### Vacuum and Analyze

```sql
-- Vacuum to reclaim storage
VACUUM FULL;

-- Analyze to update statistics
ANALYZE;

-- Vacuum and analyze specific table
VACUUM ANALYZE professionals;
```

### Check Database Size

```sql
-- Database size
SELECT pg_size_pretty(pg_database_size('asesoriaimss'));

-- Table sizes
SELECT 
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### Reindex

```sql
-- Reindex all tables
REINDEX DATABASE asesoriaimss;

-- Reindex specific table
REINDEX TABLE professionals;
```

## Troubleshooting

### Connection Issues

```bash
# Test connection
psql -U postgres -d asesoriaimss -c "SELECT version();"

# Check if PostgreSQL is running
# Windows:
sc query postgresql-x64-14

# Linux/Mac:
sudo systemctl status postgresql
```

### Permission Issues

```sql
-- Grant all privileges to user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO asesoriaimss_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO asesoriaimss_user;
```

### Reset Database

```bash
# Drop and recreate database
psql -U postgres -c "DROP DATABASE IF EXISTS asesoriaimss;"
psql -U postgres -c "CREATE DATABASE asesoriaimss;"

# Re-run schema and seed data
psql -U postgres -d asesoriaimss -f schema.sql
psql -U postgres -d asesoriaimss -f seed_data.sql
```

## Production Considerations

1. **Security**
   - Use strong passwords
   - Limit database user privileges
   - Enable SSL connections
   - Configure pg_hba.conf properly

2. **Performance**
   - Adjust PostgreSQL configuration (shared_buffers, work_mem)
   - Monitor slow queries with pg_stat_statements
   - Create additional indexes as needed
   - Regular VACUUM and ANALYZE

3. **Backups**
   - Automated daily backups
   - Off-site backup storage
   - Test restore procedures regularly
   - Keep backup retention policy (30 days recommended)

4. **Monitoring**
   - Set up database monitoring (pg_stat_activity)
   - Monitor disk space
   - Track connection pool usage
   - Alert on failed backups

## Migration with Flask-Migrate

For production, use Flask-Migrate for database migrations:

```bash
# Install Flask-Migrate
pip install Flask-Migrate

# Initialize migrations
flask db init

# Create initial migration
flask db migrate -m "Initial migration"

# Apply migration
flask db upgrade

# Rollback if needed
flask db downgrade
```

## Support

For database-related issues:
- Check PostgreSQL logs: `/var/log/postgresql/` (Linux) or Event Viewer (Windows)
- Review Flask application logs
- Consult PostgreSQL documentation: https://www.postgresql.org/docs/

## Next Steps

After database setup:
1. Configure `.env` file with DATABASE_URL
2. Test Flask application connection
3. Run initial tests with seed data
4. Create admin user if not using seed data
5. Configure backup schedule
6. Set up monitoring
