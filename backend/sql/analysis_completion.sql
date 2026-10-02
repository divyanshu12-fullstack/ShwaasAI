-- Run after sessions.sql and embeddings.sql. Safe to rerun with this signature.
-- One transaction updates an owned pending session and stores its embedding.
-- Only the server-side service_role may execute this Data API function.
create or replace function public.complete_screening_result(
    p_session_id uuid,
    p_user_id uuid,
    p_embedding_data real[],
    p_risk_score smallint,
    p_level text,
    p_recommendation text,
    p_model_used text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
    saved public.sessions%rowtype;
begin
    if coalesce(pg_catalog.array_ndims(p_embedding_data), 0) <> 1
       or pg_catalog.cardinality(p_embedding_data) <> 512 then
        raise exception 'Expected a 512-value embedding' using errcode = '22023';
    end if;

    update public.sessions
    set status = 'completed',
        risk_score = p_risk_score,
        level = p_level,
        confidence = null,
        recommendation = p_recommendation,
        model_used = p_model_used
    where session_id = p_session_id
      and user_id = p_user_id
      and status = 'pending'
    returning * into saved;

    if not found then
        return null;
    end if;

    insert into public.embeddings (session_id, embedding_data, embedding_dim)
    values (saved.session_id, p_embedding_data, 512);

    return pg_catalog.jsonb_build_object(
        'session_id', saved.session_id,
        'user_id', saved.user_id,
        'status', saved.status,
        'risk_score', saved.risk_score,
        'level', saved.level
    );
end;
$function$;

revoke all on function public.complete_screening_result(uuid, uuid, real[], smallint, text, text, text)
    from public, anon, authenticated;
grant select, update on public.sessions to service_role;
grant insert on public.embeddings to service_role;
grant execute on function public.complete_screening_result(uuid, uuid, real[], smallint, text, text, text)
    to service_role;

notify pgrst, 'reload schema';
