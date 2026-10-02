-- Run after sessions.sql in the Supabase SQL Editor.
-- One private set of screening metadata per session. Safe to rerun when the
-- existing table has this schema; column changes need a separate migration.
create table if not exists public.session_metadata (
    session_id uuid primary key references public.sessions(session_id) on delete cascade,
    age smallint check (age between 0 and 120),
    sex text check (char_length(sex) <= 40),
    fever boolean,
    smoker boolean,
    cough_duration text check (char_length(cough_duration) <= 80),
    night_sweats boolean,
    weight_loss boolean,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

alter table public.session_metadata enable row level security;
revoke all on public.session_metadata from anon, authenticated;
grant usage on schema public to authenticated;
grant select on public.session_metadata to authenticated;
grant insert (session_id, age, sex, fever, smoker, cough_duration, night_sweats, weight_loss)
    on public.session_metadata to authenticated;
grant update (age, sex, fever, smoker, cough_duration, night_sweats, weight_loss)
    on public.session_metadata to authenticated;

do $policy$
begin
    if not exists (
        select 1 from pg_policies
        where schemaname = 'public' and tablename = 'session_metadata'
          and policyname = 'metadata_select_own'
    ) then
        create policy "metadata_select_own" on public.session_metadata
            for select to authenticated
            using (
                exists (
                    select 1 from public.sessions s
                    where s.session_id = session_metadata.session_id
                      and s.user_id = (select auth.uid())
                )
            );
    end if;
end;
$policy$;

do $policy$
begin
    if not exists (
        select 1 from pg_policies
        where schemaname = 'public' and tablename = 'session_metadata'
          and policyname = 'metadata_insert_own'
    ) then
        create policy "metadata_insert_own" on public.session_metadata
            for insert to authenticated
            with check (
                exists (
                    select 1 from public.sessions s
                    where s.session_id = session_metadata.session_id
                      and s.user_id = (select auth.uid())
                )
            );
    end if;
end;
$policy$;

do $policy$
begin
    if not exists (
        select 1 from pg_policies
        where schemaname = 'public' and tablename = 'session_metadata'
          and policyname = 'metadata_update_own'
    ) then
        create policy "metadata_update_own" on public.session_metadata
            for update to authenticated
            using (
                exists (
                    select 1 from public.sessions s
                    where s.session_id = session_metadata.session_id
                      and s.user_id = (select auth.uid())
                )
            )
            with check (
                exists (
                    select 1 from public.sessions s
                    where s.session_id = session_metadata.session_id
                      and s.user_id = (select auth.uid())
                )
            );
    end if;
end;
$policy$;

create schema if not exists app_private;
revoke all on schema app_private from public, anon, authenticated;

create or replace function app_private.set_metadata_updated_at()
returns trigger language plpgsql set search_path = '' as $$
begin
    new.updated_at = now();
    return new;
end;
$$;
revoke all on function app_private.set_metadata_updated_at() from public, anon, authenticated;

do $trigger$
begin
    if not exists (
        select 1 from information_schema.triggers
        where event_object_schema = 'public'
          and event_object_table = 'session_metadata'
          and trigger_name = 'set_metadata_updated_at_before_update'
    ) then
        create trigger set_metadata_updated_at_before_update
            before update on public.session_metadata
            for each row execute function app_private.set_metadata_updated_at();
    end if;
end;
$trigger$;
