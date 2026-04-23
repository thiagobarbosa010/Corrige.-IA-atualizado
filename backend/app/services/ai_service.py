"""
AI Service — uses GPT-4o Vision to read handwritten / scanned proofs
and GPT-4o to grade each student's answers question by question.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
import traceback
from difflib import get_close_matches
from io import BytesIO

import openai
from openai import AsyncOpenAI
from pypdf import PdfReader

from app.config import settings
from app.db.supabase_client import get_supabase
from app.services.detection_service import detectar_copias
from app.services.storage_service import download_file

logger = logging.getLogger(__name__)
client = AsyncOpenAI(
    api_key=settings.openai_api_key,
    timeout=90.0,    # GPT-4o Vision em imagens grandes pode levar ~60s
    max_retries=0,   # Desabilitado — backoff manual com jitter abaixo
)

_OPENAI_RETRYABLE = (openai.RateLimitError, openai.APIStatusError, openai.APIConnectionError)


async def _openai_call(coro_factory, *, max_attempts: int = 4):
    """Exponential backoff with full jitter for OpenAI API calls.

    coro_factory must be a zero-arg callable that returns a fresh coroutine each
    call — the coroutine is consumed on the first attempt and must be recreated
    for each retry.
    """
    for attempt in range(max_attempts):
        try:
            return await coro_factory()
        except _OPENAI_RETRYABLE as exc:
            if attempt == max_attempts - 1:
                raise
            # Full jitter: sleep between 0 and cap seconds where cap doubles each attempt
            cap = 2 ** attempt  # 1s, 2s, 4s
            delay = random.uniform(0, cap)
            logger.warning(
                "OpenAI %s (attempt %d/%d) — retry em %.2fs",
                type(exc).__name__, attempt + 1, max_attempts, delay,
                extra={"attempt": attempt + 1, "delay": delay},
            )
            await asyncio.sleep(delay)


# ─── Public entrypoint ────────────────────────────────────────────────────────

async def corrigir_atividade(atividade_id: str) -> None:
    """Background task: grade all uploaded files for an activity."""
    supabase = get_supabase()
    try:
        ativ_resp = await asyncio.to_thread(
            supabase.table("atividades").select("*, questoes(*)")
            .eq("id", atividade_id).single().execute
        )
        ativ = ativ_resp.data
        questoes = sorted(ativ.get("questoes", []), key=lambda q: q["ordem"])

        uploads_resp = await asyncio.to_thread(
            supabase.table("uploads").select("*").eq("atividade_id", atividade_id).execute
        )
        uploads = uploads_resp.data

        alunos_resp = await asyncio.to_thread(
            supabase.table("alunos").select("*").eq("turma_id", ativ["turma_id"]).execute
        )
        alunos = alunos_resp.data

        results = await asyncio.gather(
            *[_processar_upload(upload, ativ, questoes, alunos) for upload in uploads],
            return_exceptions=True,
        )
        failures = 0
        for upload, result in zip(uploads, results):
            if isinstance(result, Exception):
                failures += 1
                logger.error(
                    "Erro ao processar upload %s:\n%s",
                    upload["id"],
                    "".join(traceback.format_exception(type(result), result, result.__traceback__)),
                )

        # Se todos os uploads falharam, não marcar como concluída
        if uploads and failures == len(uploads):
            raise RuntimeError(
                f"Nenhum dos {failures} upload(s) pôde ser processado. Verifique os arquivos."
            )

        await detectar_copias(atividade_id)

        await asyncio.to_thread(
            supabase.table("atividades")
            .update({"status": "concluida", "uploads_com_erro": failures})
            .eq("id", atividade_id)
            .execute
        )
        if failures:
            logger.warning(
                "Correcao %s concluida com %d/%d upload(s) com erro",
                atividade_id, failures, len(uploads),
                extra={"atividade_id": atividade_id, "failures": failures, "total": len(uploads)},
            )
        else:
            logger.info(
                "Correcao %s concluida — %d upload(s) processados",
                atividade_id, len(uploads),
                extra={"atividade_id": atividade_id, "total": len(uploads)},
            )

    except Exception:
        logger.error("Erro fatal na correcao %s:\n%s", atividade_id, traceback.format_exc())
        await asyncio.to_thread(
            supabase.table("atividades").update({"status": "erro"}).eq("id", atividade_id).execute
        )


# ─── Internal helpers ─────────────────────────────────────────────────────────

async def _processar_upload(
    upload: dict,
    ativ: dict,
    questoes: list[dict],
    alunos: list[dict],
) -> None:
    supabase = get_supabase()
    content = await download_file(upload["storage_path"])  # non-blocking

    if upload["tipo_arquivo"] == "pdf":
        texto = await _extrair_texto_pdf(content)
    else:
        # Pass the real MIME type so Vision gets the correct data URL prefix
        ct = upload.get("content_type", "image/jpeg")
        texto = await _extrair_texto_imagem(content, content_type=ct)

    aluno_id = upload.get("aluno_id")
    if not aluno_id:
        aluno_id = await _identificar_aluno(texto, alunos)

    if not aluno_id:
        logger.warning("Nao foi possivel identificar aluno para upload %s", upload["id"])
        return

    await asyncio.to_thread(
        supabase.table("uploads").update({"aluno_id": aluno_id}).eq("id", upload["id"]).execute
    )

    respostas_ia = await _corrigir_com_ia(texto, ativ, questoes)
    await asyncio.to_thread(_salvar_resultado, supabase, ativ["id"], aluno_id, questoes, respostas_ia)


async def _extrair_texto_pdf(content: bytes) -> str:
    """Try text extraction first; fall back to Vision with PNG conversion if scanned."""
    # Attempt native text extraction (non-blocking)
    def _extrair_sincrono() -> str:
        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    try:
        texto = await asyncio.to_thread(_extrair_sincrono)
        if texto.strip():
            return texto
    except Exception:
        pass

    # Fallback: render first page to image and send to Vision
    img_bytes, img_mime = await asyncio.to_thread(_pdf_primeira_pagina_png, content)
    return await _extrair_texto_imagem(img_bytes, content_type=img_mime)


def _pdf_primeira_pagina_png(content: bytes) -> tuple[bytes, str]:
    """Render the first page of a PDF to image bytes + MIME type.

    Returns (image_bytes, mime_type). Falls back gracefully through three strategies:
    1. pdf2image full render → always PNG
    2. First embedded image from the page → preserves original MIME type
    3. Raw PDF bytes → Vision will reject with a clear error (mime = application/pdf)
    """
    try:
        reader = PdfReader(BytesIO(content))
        page = reader.pages[0]

        try:
            import pdf2image  # optional dependency
            images = pdf2image.convert_from_bytes(content, first_page=1, last_page=1, dpi=150)
            buf = BytesIO()
            images[0].save(buf, format="PNG")
            return buf.getvalue(), "image/png"
        except ImportError:
            pass

        # Last resort: return first embedded image with its real MIME type
        for img_obj in page.images:
            ext = (img_obj.name or "").rsplit(".", 1)[-1].lower()
            mime = {
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "png": "image/png",
                "webp": "image/webp",
            }.get(ext, "image/jpeg")
            return img_obj.data, mime

    except Exception as exc:
        logger.warning("Nao foi possivel converter PDF para imagem: %s", exc)

    return content, "application/pdf"  # Vision will reject with a clear error


async def _extrair_texto_imagem(content: bytes, content_type: str = "image/jpeg") -> str:
    """Use GPT-4o Vision to transcribe handwritten or printed text from an image."""
    b64 = base64.b64encode(content).decode()

    resp = await _openai_call(
        lambda: client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Transcreva todo o texto desta prova/atividade escolar fielmente, "
                                "incluindo o nome do aluno no topo (se houver) e todas as respostas. "
                                "Separe claramente cada questao. Retorne apenas o texto transcrito, sem comentarios."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{content_type};base64,{b64}"},
                        },
                    ],
                }
            ],
            max_tokens=4096,
        )
    )
    return resp.choices[0].message.content or ""


async def _identificar_aluno(texto: str, alunos: list[dict]) -> str | None:
    """Match student name found in text against the class list.

    Strategy: GPT-4o extracts the name → exact match → fuzzy match (cutoff 0.72).
    Fuzzy match handles accents, typos, and partial names without a second LLM call.
    """
    if not alunos:
        return None

    nomes = [a["nome"] for a in alunos]
    nome_para_id = {a["nome"].lower(): a["id"] for a in alunos}

    prompt = (
        f"No texto abaixo, identifique qual dos seguintes alunos e o autor da prova.\n"
        f"Lista de alunos: {', '.join(nomes)}\n\n"
        f"Texto:\n{texto[:2000]}\n\n"
        f"Responda APENAS com o nome exato de um aluno da lista, ou 'desconhecido' se nao encontrar."
    )
    resp = await _openai_call(
        lambda: client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0,
        )
    )
    nome_encontrado = (resp.choices[0].message.content or "").strip().lower()

    if nome_encontrado == "desconhecido":
        return None

    # Exact match (case-insensitive)
    if nome_encontrado in nome_para_id:
        return nome_para_id[nome_encontrado]

    # Fuzzy match: handles "João" vs "Joao", partial names, transcription artifacts
    matches = get_close_matches(nome_encontrado, nome_para_id.keys(), n=1, cutoff=0.72)
    if matches:
        logger.info(
            "Aluno identificado por fuzzy match: '%s' → '%s'",
            nome_encontrado, matches[0],
        )
        return nome_para_id[matches[0]]

    logger.warning("Aluno nao identificado para nome extraido: '%s'", nome_encontrado)
    return None


async def _corrigir_com_ia(
    texto_respostas: str,
    ativ: dict,
    questoes: list[dict],
) -> list[dict]:
    """Send student responses to GPT-4o for grading and return structured JSON."""
    questoes_fmt = "\n".join(
        f"Q{q['ordem']} (id={q['id']}, peso={q['peso']}): {q['enunciado']}"
        + (f"\nGabarito: {q['gabarito']}" if q.get("gabarito") else "")
        for q in questoes
    )

    gabarito_bloco = ""
    if ativ.get("gabarito_texto"):
        gabarito_bloco = f"Gabarito geral:\n{ativ['gabarito_texto']}\n"

    prompt = f"""Voce e um professor assistente corrigindo a seguinte atividade.

Atividade: {ativ['nome']}
Modo: {ativ['modo_correcao']}
{gabarito_bloco}
Questoes:
{questoes_fmt}

Respostas do aluno:
{texto_respostas}

Retorne um JSON no formato exato abaixo (objeto com chave "respostas"):
{{
  "respostas": [
    {{
      "questao_id": "<id exato da questao>",
      "texto_resposta": "<trecho exato da resposta do aluno para esta questao>",
      "status": "correto" | "parcial" | "errado",
      "nota": <numero entre 0 e o peso da questao>,
      "comentario": "<feedback construtivo em 1-2 frases>",
      "flag": null | "ia" | "plagio"
    }}
  ]
}}

Regra para flag "ia": detecte vocabulario excessivamente formal, estrutura
padronizada e ausencia de erros naturais de escrita a mao.
Nao use flag "copia" — isso e detectado por outro sistema."""

    resp = await _openai_call(
        lambda: client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0,  # determinístico — notas reproduzíveis para a mesma submissão
            response_format={"type": "json_object"},
        )
    )

    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error(
            "GPT retornou JSON invalido para atividade — nao e possivel corrigir: %s", raw[:500]
        )
        raise RuntimeError("GPT retornou JSON inválido — correção abortada.") from exc

    respostas = parsed.get("respostas", [])
    if not isinstance(respostas, list):
        logger.error("GPT retornou formato inesperado (sem chave 'respostas'): %s", raw[:500])
        raise RuntimeError("GPT retornou formato inesperado — correção abortada.")

    return respostas


def _salvar_resultado(
    supabase,
    atividade_id: str,
    aluno_id: str,
    questoes: list[dict],
    respostas_ia: list[dict],
) -> None:
    """Persist grading results to the database (runs in thread pool)."""
    peso_map = {q["id"]: float(q.get("peso", 1)) for q in questoes}
    nota_total = sum(
        max(0.0, min(float(r.get("nota") or 0), peso_map.get(r.get("questao_id"), 0)))
        for r in respostas_ia
    )

    resultado_resp = (
        supabase.table("resultados")
        .upsert(
            {"atividade_id": atividade_id, "aluno_id": aluno_id, "nota_total": nota_total},
            on_conflict="atividade_id,aluno_id",
        )
        .execute()
    )
    resultado_id = resultado_resp.data[0]["id"]

    respostas_rows = [
        {
            "resultado_id": resultado_id,
            "questao_id": r.get("questao_id"),
            "nota": r.get("nota"),
            "status": r.get("status"),
            "comentario_ia": r.get("comentario"),
            "flag_tipo": r.get("flag"),
            "texto_resposta": r.get("texto_resposta"),
        }
        for r in respostas_ia
    ]

    if respostas_rows:
        supabase.table("respostas").delete().eq("resultado_id", resultado_id).execute()
        supabase.table("respostas").insert(respostas_rows).execute()
