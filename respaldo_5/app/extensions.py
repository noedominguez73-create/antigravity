from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Initialize Limiter with default settings
# We use key_func=get_remote_address to identify users by IP
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)
