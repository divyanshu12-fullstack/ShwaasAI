-- Run after sessions.sql. The caller's JWT determines the user; no user ID
-- argument is accepted. SECURITY INVOKER keeps sessions RLS in force.
create index if not exists sessions_user_recorded_idx
    on public.sessions (user_id, recorded_at, session_id);

create or replace function public.get_my_session_stats()
returns jsonb
language sql
stable
security invoker
set search_path = ''
as $stats$
    with own as materialized (
        select input_type, risk_score, level, recorded_at, session_id
        from public.sessions
        where user_id = (select auth.uid())
    ), summary as (
        select count(*) as total_screenings,
               max(recorded_at) as last_screened_at,
               round(avg(risk_score)::numeric, 1) as average_risk_score,
               count(*) filter (where level = 'Low') as low_count,
               count(*) filter (where level = 'Moderate') as moderate_count,
               count(*) filter (where level = 'High') as high_count,
               count(*) filter (where input_type = 'cough') as cough_count,
               count(*) filter (where input_type = 'breathing') as breathing_count
        from own
    ), trend as (
        select coalesce(
            jsonb_agg(
                jsonb_build_object(
                    'date', (recorded_at at time zone 'UTC')::date,
                    'risk_score', risk_score
                ) order by recorded_at, session_id
            ),
            '[]'::jsonb
        ) as points
        from own
        where risk_score is not null
    )
    select jsonb_build_object(
        'total_screenings', summary.total_screenings,
        'last_screened_at', summary.last_screened_at,
        'average_risk_score', summary.average_risk_score,
        'risk_distribution', jsonb_build_object(
            'Low', summary.low_count,
            'Moderate', summary.moderate_count,
            'High', summary.high_count
        ),
        'input_type_breakdown', jsonb_build_object(
            'cough', summary.cough_count,
            'breathing', summary.breathing_count
        ),
        'risk_trend', trend.points
    )
    from summary cross join trend;
$stats$;

revoke all on function public.get_my_session_stats() from public, anon;
grant usage on schema public to authenticated;
grant execute on function public.get_my_session_stats() to authenticated;

notify pgrst, 'reload schema';
