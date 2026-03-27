import os
from pathlib import Path

import dotenv

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

dotenv.load_dotenv()


ENVIRONMENT = os.environ.get("ENVIRONMENT", "production")

if ENVIRONMENT == "development":
    SECRET_KEY = os.environ.get(
        "SECRET_KEY", "django-insecure-dev-fallback-key-do-not-use-in-production"
    )
else:
    SECRET_KEY = os.environ.get("SECRET_KEY")  # type: ignore
    if not SECRET_KEY:
        raise RuntimeError(
            "SECRET_KEY environment variable must be set in non-development environments."
        )

if ENVIRONMENT == "development":
    DEBUG = True
else:
    DEBUG = False

ALLOWED_HOSTS: list[str] = [
    "localhost",
    "127.0.0.1",
    "lucidtree.onrender.com",
    "api.lucidgo.org",
]


# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "corsheaders",
    "api.game_api",
    "api.common",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
]

STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
    },
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    },
}

ROOT_URLCONF = "api.api.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "api.api.wsgi.application"


# Database
# https://docs.djangoproject.com/en/6.0/ref/settings/#databases

if ENVIRONMENT == "development":
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }
else:
    import dj_database_url

    DATABASES = {
        "default": dj_database_url.parse(  # type: ignore
            os.environ.get("DATABASE_URL") or "",
            conn_max_age=600,
            conn_health_checks=True,
        )
    }


# Password validation
# https://docs.djangoproject.com/en/6.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/6.0/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "Asia/Singapore"

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/6.0/howto/static-files/

STATIC_URL = "/static/"

STATIC_ROOT = BASE_DIR / "staticfiles"

STATICFILES_DIRS = [BASE_DIR / "static"]


# CORS Settings

CORS_ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://lucidtree.onrender.com",
    "https://api.lucidgo.org",
]

CORS_ALLOW_CREDENTIALS = True
