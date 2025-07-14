import logging
import alembic.config
import os
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine

from src.app.health import set_dependency_healthy, set_dependency_unhealthy
from src.app.configs.config import get_config


logger = logging.getLogger(__name__)


def check_migration_status():
    """
    Check if migrations are needed by comparing current database version with available migrations.
    
    Returns:
        tuple: (needs_migration: bool, current_revision: str, head_revision: str)
    """
    try:
        config = get_config()
        engine = create_engine(config.database_url)
        
        # Get current database revision
        with engine.connect() as connection:
            context = MigrationContext.configure(connection)
            current_revision = context.get_current_revision()
        
        # Get the alembic.ini path
        alembic_ini_path = os.path.join(os.getcwd(), "alembic.ini")
        if not os.path.exists(alembic_ini_path):
            logger.warning(f"Alembic configuration file not found at: {alembic_ini_path}")
            return True, current_revision, None  # Assume migration needed if config missing
        
        # Get head revision from migration scripts
        alembic_cfg = alembic.config.Config(alembic_ini_path)
        script_dir = ScriptDirectory.from_config(alembic_cfg)
        head_revision = script_dir.get_current_head()
        
        needs_migration = current_revision != head_revision
        
        logger.debug(f"Migration status check - Current: {current_revision}, Head: {head_revision}, Needs migration: {needs_migration}")
        
        return needs_migration, current_revision, head_revision
        
    except Exception as e:
        logger.warning(f"Could not check migration status: {str(e)}. Assuming migration needed.")
        return True, None, None


def run_migrations():
    """
    Run Alembic migrations and track the status using the dependency health system.
    Only runs migrations if they are actually needed.
    
    Raises:
        Exception: If migrations fail, the exception is re-raised after logging.
    """
    try:
        # First check if migrations are actually needed
        needs_migration, current_rev, head_rev = check_migration_status()
        
        if not needs_migration:
            logger.info(f"Database is already up to date (revision: {current_rev})")
            set_dependency_healthy("database_migrations", {
                "last_migration": current_rev or "unknown",
                "migration_tool": "alembic",
                "status": "up_to_date"
            })
            return
        
        logger.info(f"Running migrations from {current_rev} to {head_rev}")
        
        # Get the path to the alembic.ini file
        alembic_ini_path = os.path.join(os.getcwd(), "alembic.ini")
        
        # Check if alembic.ini exists
        if not os.path.exists(alembic_ini_path):
            logger.error(f"Alembic configuration file not found at: {alembic_ini_path}")
            logger.error(f"Current working directory: {os.getcwd()}")
            logger.error(f"Directory contents: {os.listdir(os.getcwd())}")
            raise FileNotFoundError(f"alembic.ini not found at {alembic_ini_path}")
        
        alembicArgs = [
            '--raiseerr',
            '-c', alembic_ini_path,  # Specify the config file path
            'upgrade', 'head',
        ]
        
        logger.info(f"Starting database migrations using config: {alembic_ini_path}")
        alembic.config.main(argv=alembicArgs)
        set_dependency_healthy("database_migrations", {
            "last_migration": "head",
            "migration_tool": "alembic",
            "status": "completed"
        })
        logger.info("Database migrations completed successfully")
        
    except Exception as e:
        error_msg = f"Database migration failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        set_dependency_unhealthy("database_migrations", error_msg, {
            "migration_tool": "alembic",
            "attempted_target": "head"
        })
        raise  # Re-raise the exception to maintain existing behavior