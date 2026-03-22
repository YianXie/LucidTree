"""
conftest.py for api/ directory.

Sets up the Django environment so that pytest can discover and run the
Django-based tests inside api/game_api/tests.py.
"""

import os
import sys

# Ensure the api/ package is on sys.path (for imports like 'from common...')
api_dir = os.path.dirname(__file__)
if api_dir not in sys.path:
    sys.path.insert(0, api_dir)

# Ensure the lucidtree source package is also importable
src_dir = os.path.join(os.path.dirname(api_dir), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "api.settings")

import django  # noqa: E402

django.setup()
