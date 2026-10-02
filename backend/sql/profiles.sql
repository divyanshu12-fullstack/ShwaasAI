-- Run once in the Supabase SQL Editor before using /api/v1/auth/me.
-- Supabase Auth owns credentials; this table stores only app profile data.
create table if not exists public.profiles (
    user_id uuid primary key references auth.users(id) on delete cascade,
    display_name text check (char_length(display_name) <= 80),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

alter table public.profiles enable row level security;
revoke all on public.profiles from anon, authenticated;
grant usage on schema public to authenticated;
grant select on public.profiles to authenticated;
grant update (display_name) on public.profiles to authenticated;

create policy "profiles_select_own" on public.profiles
    for select to authenticated
    using ((select auth.uid()) = user_id);

create policy "profiles_update_own" on public.profiles
    for update to authenticated
    using ((select auth.uid()) = user_id)
    with check ((select auth.uid()) = user_id);

-- Trigger functions are private and have a fixed search path. Application
-- clients cannot insert profile rows or set their owner ID.
create schema if not exists app_private;
revoke all on schema app_private from public, anon, authenticated;

create function app_private.create_profile_for_new_user()
returns trigger language plpgsql security definer set search_path = '' as $$
begin
    insert into public.profiles (user_id) values (new.id);
    return new;
end;
$$;
revoke all on function app_private.create_profile_for_new_user() from public, anon, authenticated;

create trigger create_profile_after_signup
    after insert on auth.users
    for each row execute function app_private.create_profile_for_new_user();

create function app_private.set_profile_updated_at()
returns trigger language plpgsql security definer set search_path = '' as $$
begin
    new.updated_at = now();
    return new;
end;
$$;
revoke all on function app_private.set_profile_updated_at() from public, anon, authenticated;

create trigger set_profile_updated_at_before_update
    before update on public.profiles
    for each row execute function app_private.set_profile_updated_at();

-- Include users who registered before this schema was installed.
insert into public.profiles (user_id)
select id from auth.users
on conflict (user_id) do nothing;
