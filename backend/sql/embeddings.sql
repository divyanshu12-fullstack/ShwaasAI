-- Run after sessions.sql. One optional aggregated embedding per screening.
-- The analysis backend will write these rows after computing an embedding.
-- Mobile clients may read only their own embedding and cannot write vectors.
create table if not exists public.embeddings (
    session_id uuid primary key references public.sessions(session_id) on delete cascade,
    embedding_data real[] not null,
    embedding_dim integer not null check (embedding_dim > 0),
    stored_at timestamptz not null default now(),
    constraint embeddings_dimension_matches_data check (
        array_ndims(embedding_data) = 1
        and cardinality(embedding_data) = embedding_dim
    )
);

alter table public.embeddings enable row level security;
revoke all on public.embeddings from anon, authenticated;
grant usage on schema public to authenticated;
grant select on public.embeddings to authenticated;

do $policy$
begin
    if not exists (
        select 1 from pg_policies
        where schemaname = 'public' and tablename = 'embeddings'
          and policyname = 'embeddings_select_own'
    ) then
        create policy "embeddings_select_own" on public.embeddings
            for select to authenticated
            using (
                exists (
                    select 1 from public.sessions s
                    where s.session_id = embeddings.session_id
                      and s.user_id = (select auth.uid())
                )
            );
    end if;
end;
$policy$;

notify pgrst, 'reload schema';
