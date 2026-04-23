import asyncio
import logging
import random
import uuid

from app.db.supabase_client import get_supabase

logger = logging.getLogger(__name__)
BUCKET = "provas"


async def _storage_retry(fn, *, label: str, max_attempts: int = 3):
    """Run a synchronous Supabase Storage callable with exponential backoff + jitter.

    fn must be a zero-arg callable (lambda/partial). Its return value is passed through.
    """
    for attempt in range(max_attempts):
        try:
            return await asyncio.to_thread(fn)
        except Exception as exc:
            if attempt == max_attempts - 1:
                raise
            delay = random.uniform(0, 2 ** attempt)  # full jitter: 0-1s, 0-2s
            logger.warning(
                "Storage %s falhou (tentativa %d/%d) — retry em %.1fs: %s",
                label, attempt + 1, max_attempts, delay, exc,
                extra={"attempt": attempt + 1, "label": label},
            )
            await asyncio.sleep(delay)


async def upload_file(
    content: bytes,
    filename: str,
    content_type: str,
    atividade_id: str,
) -> str:
    """Upload a file to Supabase Storage and return its storage path."""
    supabase = get_supabase()
    ext = filename.rsplit(".", 1)[-1] if "." in filename else "bin"
    storage_path = f"{atividade_id}/{uuid.uuid4()}.{ext}"

    await _storage_retry(
        lambda: supabase.storage.from_(BUCKET).upload(
            path=storage_path,
            file=content,
            file_options={"content-type": content_type},
        ),
        label=f"upload/{storage_path}",
    )
    return storage_path


async def download_file(storage_path: str) -> bytes:
    """Download a file from Supabase Storage without blocking the event loop."""
    supabase = get_supabase()
    return await _storage_retry(
        lambda: supabase.storage.from_(BUCKET).download(storage_path),
        label=f"download/{storage_path}",
    )
