import asyncio
from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import TurmaCreate, TurmaOut
from app.db.supabase_client import get_supabase
from app.dependencies import get_current_user

router = APIRouter(prefix="/turmas", tags=["turmas"])


def _extract_counts(t: dict) -> dict:
    t["total_alunos"] = t.pop("alunos", [{}])[0].get("count", 0) if t.get("alunos") else 0
    t["total_atividades"] = t.pop("atividades", [{}])[0].get("count", 0) if t.get("atividades") else 0
    return t


@router.get("", response_model=list[TurmaOut])
async def listar_turmas(current_user: dict = Depends(get_current_user)):
    supabase = get_supabase()
    result = await asyncio.to_thread(
        supabase.table("turmas")
        .select("*, alunos(count), atividades(count)")
        .eq("professor_id", current_user["id"])
        .order("criado_em", desc=True)
        .execute
    )
    return [_extract_counts(t) for t in result.data]


@router.post("", response_model=TurmaOut, status_code=201)
async def criar_turma(body: TurmaCreate, current_user: dict = Depends(get_current_user)):
    supabase = get_supabase()
    data = body.model_dump()
    data["professor_id"] = current_user["id"]
    result = await asyncio.to_thread(supabase.table("turmas").insert(data).execute)
    if not result.data:
        raise HTTPException(status_code=500, detail="Erro ao criar turma.")
    row = result.data[0]
    row["total_alunos"] = 0
    row["total_atividades"] = 0
    return row


@router.get("/{turma_id}", response_model=TurmaOut)
async def detalhe_turma(turma_id: str, current_user: dict = Depends(get_current_user)):
    supabase = get_supabase()
    result = await asyncio.to_thread(
        supabase.table("turmas")
        .select("*, alunos(count), atividades(count)")
        .eq("id", turma_id)
        .eq("professor_id", current_user["id"])
        .single()
        .execute
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Turma não encontrada.")
    return _extract_counts(result.data)


@router.delete("/{turma_id}", status_code=204)
async def deletar_turma(turma_id: str, current_user: dict = Depends(get_current_user)):
    supabase = get_supabase()
    result = await asyncio.to_thread(
        supabase.table("turmas")
        .delete()
        .eq("id", turma_id)
        .eq("professor_id", current_user["id"])
        .execute
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Turma não encontrada.")
