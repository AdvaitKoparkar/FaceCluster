import os

from .common_config import COMMON_CONFIG

DATABASE_CONFIG = \
{
    "db_name": "FaceCluster_DB",
    "db_path": os.path.join(COMMON_CONFIG["root"], "data"),
}
