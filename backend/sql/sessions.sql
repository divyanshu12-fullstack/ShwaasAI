-- Run in the Supabase SQL Editor before using /api/v1/sessions.
-- Safe to rerun when the existing table has this schema; this is not a
-- general-purpose migration for changing columns on an older schema.
-- The mobile client may create pending sessions, but cannot write model results.
create table if not exists public.sessions (
    session_id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    input_type text not null check (input_type in ('cough', 'breathing')),
    cough_type text check (cough_type in ('passive', 'forced')),
    status text not null default 'pending' check (status in ('pending', 'completed', 'failed')),
    risk_score smallint check (risk_score between 0 and 100),
    level text check (level in ('Low', 'Moderate', 'High')),
    confidence smallint check (confidence between 0 and 100),
    recommendation text,
    model_used text,
    recorded_at timestamptz not null default now(),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint breathing_without_cough_type check (input_type <> 'breathing' or cough_type is null),
    constraint completed_has_result check (
        status <> 'completed' or (risk_score is not null and level is not null and model_used is not null)
    )
);

create index if not exists sessions_user_created_idx
    on public.sessions (user_id, created_at desc, session_id desc);

alter table public.sessions enable row level security;
revoke all on public.sessions from anon, authenticated;
grant usage on schema public to authenticated;
grant select on public.sessions to authenticated;
grant insert (user_id, input_type, cough_type, recorded_at) on public.sessions to authenticated;
grant delete on public.sessions to authenticated;

do $policy$
begin
    if not exists (
        select 1 from pg_policies
        where schemaname = 'public' and tablename = 'sessions' and policyname = 'sessions_select_own'
    ) then
        create policy "sessions_select_own" on public.sessions
            for select to authenticated
            using ((select auth.uid()) = user_id);
    end if;
end;
$policy$;

do $policy$
begin
    if not exists (
        select 1 from pg_policies
        where schemaname = 'public' and tablename = 'sessions' and policyname = 'sessions_insert_own'
    ) then
        create policy "sessions_insert_own" on public.sessions
            for insert to authenticated
            with check ((select auth.uid()) = user_id);
    end if;
end;
$policy$;

do $policy$
begin
    if not exists (
        select 1 from pg_policies
        where schemaname = 'public' and tablename = 'sessions' and policyname = 'sessions_delete_own'
    ) then
        create policy "sessions_delete_own" on public.sessions
            for delete to authenticated
            using ((select auth.uid()) = user_id);
    end if;
end;
$policy$;

create schema if not exists app_private;
revoke all on schema app_private from public, anon, authenticated;

create or replace function app_private.set_session_updated_at()
returns trigger language plpgsql set search_path = '' as $$
begin
    new.updated_at = now();
    return new;
end;
$$;
revoke all on function app_private.set_session_updated_at() from public, anon, authenticated;

do $trigger$
begin
    if not exists (
        select 1 from information_schema.triggers
        where event_object_schema = 'public'
          and event_object_table = 'sessions'
          and trigger_name = 'set_session_updated_at_before_update'
    ) then
        create trigger set_session_updated_at_before_update
            before update on public.sessions
            for each row execute function app_private.set_session_updated_at();
    end if;
end;
$trigger$;
